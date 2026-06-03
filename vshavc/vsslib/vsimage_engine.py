"""
------------------------------------------------------------------------------- 
Author: Dan64
Date: 2024-05-09
version: 
LastEditors: Dan64
LastEditTime: 2026-05-19
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
module containing the functions used to colorize single images using DiT models
and deoldify() and/or ddcolor().
"""
import vapoursynth as vs
import math
import os
import io
import uuid
import numpy as np
import xmlrpc.client
from multiprocessing.shared_memory import SharedMemory
from PIL import Image
from vshavc.vsslib.constants import DEF_CALL_TIMEOUT, DEF_CONNECT_TIMEOUT
from vshavc.vsslib.imfilters import image_weighted_merge

"""
-------------------------------------------------------------------------------
Author: Dan64
LastEditTime: 2026-05-09
-------------------------------------------------------------------------------
Description:
-------------------------------------------------------------------------------
HAVCditEngine — Single-frame colorization engine via RPC.

Same public interface as HAVCimageEngine:
   colorize_image(pil_img) -> Colorized PIL Image
   reset() -> Drop connection and reset singleton

Same singleton and lazy loading pattern: instantiating HAVCditEngine() does NOT
open the RPC connection or load the model. Both operations
occur the first time colorize_merged() is used, just as HAVCimageEngine
loads DeOldify/DDColor on first use.

Images travel entirely in RAM via XML-RPC (PNG bytes, base64):
no filesystem access, unlike process_image() in the GUI.

The RPC server (HAVC_colorize_server.py) must already be listening before
the first call. If the model is already loaded on the server (e.g., from the
GUI), load_pipeline is not repeated.

Dependencies: Python stdlib + Pillow only. No dependencies on
HAVC_rpc_client.py, the GUI, VapourSynth, or torch.

Typical usage (identical to HAVCimageEngine):
    engine = HAVCditEngine(
        host="127.0.0.1",
        port=8765,
        full_model_path="./models/svdq-fp4_r128-...safetensors",
    )
    colorized_img = engine.colorize_merged(bw_pil_image)
-------------------------------------------------------------------------------
"""


# ---------------------------------------------------------------------------
# Image conversion helpers (in-memory, no filesystem)
# ---------------------------------------------------------------------------

def _pil_to_bytes(img: Image.Image) -> bytes:
    """Serialize an RGB PIL Image into PNG raw bytes."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _bytes_to_pil(data) -> Image.Image:
    """
    Deserialize PNG raw bytes to PIL Image RGB.
    It handles both native bytes (use_builtin_types=True) and
    xmlrpc.client.Binary (use_builtin_types=False).
    """
    raw = data.data if hasattr(data, "data") else data
    return Image.open(io.BytesIO(raw)).convert("RGB")


# ---------------------------------------------------------------------------
# Transport con timeout configurabile
# (stesso pattern della GUI, inlineato per autonomia)
# ---------------------------------------------------------------------------

class _TimeoutTransport(xmlrpc.client.Transport):
    """
    A Transport subclass that sets the timeout on HTTPConnection.
    The standard Transport does not expose the timeout directly.
    """
    def __init__(self, timeout: float):
        super().__init__()
        self._timeout = timeout

    def make_connection(self, host):
        conn = super().make_connection(host)
        conn.timeout = self._timeout
        return conn


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Singleton RPC client RPC for colorization of single B&W frame using external DiT
models RPC Server
"""
class HAVCditEngine:
    """
    Parameters
    ---------
    host : str
        IP address of the RPC server (default: "127.0.0.1").
    port : int
        RPC server TCP port (default: 8765).
    model_name : str
        Nunchaku model name (default: "nunchaku-qwen").
    model_precision : str
        Quantization precision: "fp4" (RTX 50-Series) or "int4" (RTX 30/40-Series).
        Default: "fp4".
    model_rank : str
        SVD rank: "32" or "128" (default: "32").
    model_inference_steps : str
        Steps used to select the model file to download: "4" (default).
        Inference can be run faster with steps=2 independently of this value.
    cache_dir : str
        HuggingFace cache directory. Leave empty to use the default
        (~/.cache/huggingface).
    full_model_path : str
        Absolute path to a local .safetensors file. When specified,
        model_name / precision / rank / inference_steps are ignored.
    prompt : str
        Text prompt to guide colorization.
    steps : int
        Inference steps per image (default: 2). Independent of model_inference_steps.
    img_size : int
        Maximum long side in pixels before inference.
        0 = original size, no resize (default).

    Transport
    ---------
    When host is 127.0.0.1 / localhost, image data is transferred via shared
    memory (zero-copy, ~23% faster). For remote hosts the standard RPC
    transport (PNG bytes, base64) is used automatically.

    Singleton Notes
    ------------------
    HAVCditEngine() with different parameters after the first instance
    returns the existing instance unchanged.
    To change parameters: call reset() before rebuilding.
    """

    _instance    = None
    _initialized = False

    # Timeout for fast calls (ping, is_pipeline_loaded)
    _CONNECT_TIMEOUT = DEF_CONNECT_TIMEOUT    # secondi

    # Timeout for slow calls (load_pipeline, colorize_frame).
    # load_pipeline() can take more time on slow hardware.
    _CALL_TIMEOUT = DEF_CALL_TIMEOUT  # secondi

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        host: str              = "127.0.0.1",
        port: int              = 8765,
        model_name: str            = "nunchaku-qwen",
        model_precision: str       = "fp4",
        model_rank: str            = "32",
        model_inference_steps: str = "4",
        cache_dir: str             = "",
        full_model_path: str       = "",
        prompt: str = (
            "Colorize this image, natural colors. "
            "Strictly preserve all shapes, edges and background details."
        ),
        steps: int    = 2,
        img_size: int = 0,
    ):
        if self._initialized:
            return

        self._host                  = host
        self._port                  = port
        self._model_name            = model_name
        self._model_precision       = model_precision
        self._model_rank            = model_rank
        self._model_inference_steps = model_inference_steps
        self._cache_dir             = cache_dir
        self._full_model_path       = full_model_path
        self._prompt                = prompt
        self._steps                 = steps
        self._img_size              = img_size

        # Proxy RPC - initialized to the first call to colorize_image()
        self._proxy_fast = None   # ping, is_pipeline_loaded
        self._proxy_slow = None   # load_pipeline, colorize_frame

        self._initialized = True

    # ------------------------------------------------------------------
    # Lazy connection + pipeline load
    # ------------------------------------------------------------------

    @property
    def _use_shm(self) -> bool:
        """True when server and client are on the same host (shared memory available)."""
        return self._host in ("127.0.0.1", "localhost", "::1")

    def _shm_write(self, img: Image.Image):
        """
        Allocate a SharedMemory segment, write PIL Image pixels into it,
        and return (shm, height, width).
        The caller is responsible for shm.close() and shm.unlink().
        """
        arr  = np.array(img)
        h, w = arr.shape[:2]
        name = f"diteng_{uuid.uuid4().hex[:12]}"
        shm  = SharedMemory(name=name, create=True, size=h * w * 3)
        np.ndarray((h, w, 3), dtype=np.uint8, buffer=shm.buf)[:] = arr
        return shm, h, w

    def _shm_read(self, shm: SharedMemory, h: int, w: int) -> Image.Image:
        """Read a PIL Image from an open SharedMemory segment."""
        arr = np.ndarray((h, w, 3), dtype=np.uint8, buffer=shm.buf)
        return Image.fromarray(arr.copy(), mode="RGB")

    def _ensure_connection(self):
        """
        Open RPC proxies and load the pipeline on first use.
        No-op if already connected.

        Both proxies use use_builtin_types=True for correct bool/int handling.
        """
        if self._proxy_fast is not None:
            return

        uri = f"http://{self._host}:{self._port}"

        proxy_fast = xmlrpc.client.ServerProxy(
            uri,
            transport=_TimeoutTransport(self._CONNECT_TIMEOUT),
            allow_none=True,
            use_builtin_types=True,
        )
        proxy_slow = xmlrpc.client.ServerProxy(
            uri,
            transport=_TimeoutTransport(self._CALL_TIMEOUT),
            allow_none=True,
            use_builtin_types=True,
        )

        # Verify connection
        try:
            if proxy_fast.ping() != "pong":
                raise ConnectionError("unexpected ping response")
        except ConnectionRefusedError:
            raise ConnectionError(
                f"HAVCditEngine: RPC server not reachable at "
                f"{self._host}:{self._port} (Connection refused)"
            )
        except TimeoutError:
            raise ConnectionError(
                f"HAVCditEngine: connection timeout to "
                f"{self._host}:{self._port}"
            )
        except Exception as e:
            raise ConnectionError(
                f"HAVCditEngine: connection failed — {e}"
            )

        # Load pipeline only if not already in server memory
        if not proxy_fast.is_pipeline_loaded():
            result = proxy_slow.load_pipeline(
                self._model_name,
                self._model_precision,
                self._model_rank,
                self._model_inference_steps,
                self._cache_dir,
                self._full_model_path,
            )
            if not result.get("ok"):
                raise RuntimeError(
                    f"HAVCditEngine: pipeline load failed — "
                    f"{result.get('msg', 'unknown error')}"
                )

        self._proxy_fast = proxy_fast
        self._proxy_slow = proxy_slow

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def colorize_image(self, pil_img: Image.Image) -> Image.Image:
        """
        Colorize a single B&W frame via the RPC server.

        When the server is on the same host (127.0.0.1 / localhost), image
        data is transferred via shared memory (zero-copy). For remote hosts
        the standard PNG-over-RPC transport is used automatically.

        Parameters
        ----------
        pil_img : PIL.Image.Image
            B&W input frame in RGB mode (L channel replicated 3 times).
            Non-RGB inputs are converted automatically.

        Returns
        -------
        PIL.Image.Image
            Colorized frame, RGB mode, same resolution as input.
            If the frame is too dark (skipped by the server), the input
            is returned unchanged — same behaviour as HAVCimageEngine.

        Raises
        ------
        ConnectionError : server not reachable or pipeline not loadable.
        RuntimeError    : RPC error during colorization.
        ValueError      : input is not a PIL Image.
        """
        if not isinstance(pil_img, Image.Image):
            raise ValueError(
                "HAVCditEngine.colorize_image: input must be a PIL Image"
            )
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        self._ensure_connection()

        if self._use_shm:
            return self._colorize_image_shm(pil_img)

        result = self._proxy_slow.colorize_frame(
            _pil_to_bytes(pil_img),
            self._prompt,
            self._img_size,
            self._steps,
        )
        if not result.get("ok"):
            raise RuntimeError(
                f"HAVCditEngine.colorize_image: RPC error — "
                f"{result.get('msg', 'unknown error')}"
            )
        if result.get("skipped"):
            return pil_img
        return _bytes_to_pil(result["data"])

    def _colorize_image_shm(self, pil_img: Image.Image) -> Image.Image:
        """
        Internal SHM implementation for colorize_image().
        Assumes input is already validated, RGB, and _ensure_connection() called.
        """
        shm_in, h, w = self._shm_write(pil_img)
        shm_out = SharedMemory(
            name=f"diteng_out_{uuid.uuid4().hex[:12]}",
            create=True, size=h * w * 3)
        try:
            result = self._proxy_slow.colorize_frame_shm(
                shm_in.name, shm_out.name, h, w,
                self._prompt, self._img_size, self._steps,
            )
            if not result.get("ok"):
                raise RuntimeError(
                    f"HAVCditEngine.colorize_image: RPC error — "
                    f"{result.get('msg', 'unknown error')}"
                )
            if result.get("skipped"):
                return pil_img
            return self._shm_read(shm_out, h, w)
        finally:
            shm_in.close();  shm_in.unlink()
            shm_out.close(); shm_out.unlink()

    def colorize_merged(self, pil_img: Image.Image) -> Image.Image:
        """Alias for colorize_image() - compatible with HAVCimageEngine interface."""
        return self.colorize_image(pil_img)

    def colorize_image_pair(
        self,
        pil_img1: Image.Image,
        pil_img2: Image.Image,
    ) -> tuple:
        """
        Colorize two B&W frames in a single inference pass.

        The two images are placed side-by-side and processed in one forward
        pass, roughly halving the per-image cost versus two separate calls.

        When the server is on the same host (127.0.0.1 / localhost), image
        data is transferred via shared memory (zero-copy). For remote hosts
        the standard PNG-over-RPC transport is used automatically.

        Parameters
        ----------
        pil_img1, pil_img2 : PIL.Image.Image
            B&W input frames, RGB mode. Non-RGB inputs are converted automatically.

        Returns
        -------
        tuple[PIL.Image.Image, PIL.Image.Image]
            (colorized_img1, colorized_img2). If a frame is too dark it is
            returned unchanged.

        Raises
        ------
        ConnectionError : server not reachable or pipeline not loadable.
        RuntimeError    : RPC error during colorization.
        ValueError      : inputs are not PIL Images.
        """
        for img, name in ((pil_img1, "pil_img1"), (pil_img2, "pil_img2")):
            if not isinstance(img, Image.Image):
                raise ValueError(
                    f"HAVCditEngine.colorize_image_pair: {name} must be a PIL Image")
        if pil_img1.mode != "RGB":
            pil_img1 = pil_img1.convert("RGB")
        if pil_img2.mode != "RGB":
            pil_img2 = pil_img2.convert("RGB")

        self._ensure_connection()

        if self._use_shm:
            return self._colorize_image_pair_shm(pil_img1, pil_img2)

        result = self._proxy_slow.colorize_frame_pair(
            _pil_to_bytes(pil_img1),
            _pil_to_bytes(pil_img2),
            self._prompt,
            8,  # gap_px
        )
        if not result.get("ok"):
            raise RuntimeError(
                f"HAVCditEngine.colorize_image_pair: RPC error — "
                f"{result.get('msg', 'unknown error')}"
            )
        out1 = pil_img1 if result.get("skipped1") else _bytes_to_pil(result["data1"])
        out2 = pil_img2 if result.get("skipped2") else _bytes_to_pil(result["data2"])
        return out1, out2

    def _colorize_image_pair_shm(
        self,
        pil_img1: Image.Image,
        pil_img2: Image.Image,
    ) -> tuple:
        """
        Internal SHM implementation for colorize_image_pair().
        Assumes inputs are already validated, RGB, and _ensure_connection() called.
        """
        shm_in1, h1, w1 = self._shm_write(pil_img1)
        shm_in2, h2, w2 = self._shm_write(pil_img2)
        shm_out1 = SharedMemory(
            name=f"diteng_o1_{uuid.uuid4().hex[:12]}", create=True, size=h1 * w1 * 3)
        shm_out2 = SharedMemory(
            name=f"diteng_o2_{uuid.uuid4().hex[:12]}", create=True, size=h2 * w2 * 3)
        try:
            result = self._proxy_slow.colorize_frame_pair_shm(
                shm_in1.name, shm_out1.name, h1, w1,
                shm_in2.name, shm_out2.name, h2, w2,
                self._prompt,
                8,  # gap_px
            )
            if not result.get("ok"):
                raise RuntimeError(
                    f"HAVCditEngine.colorize_image_pair: RPC error — "
                    f"{result.get('msg', 'unknown error')}"
                )
            out1 = pil_img1 if result.get("skipped1") else self._shm_read(shm_out1, h1, w1)
            out2 = pil_img2 if result.get("skipped2") else self._shm_read(shm_out2, h2, w2)
            return out1, out2
        finally:
            shm_in1.close();  shm_in1.unlink()
            shm_in2.close();  shm_in2.unlink()
            shm_out1.close(); shm_out1.unlink()
            shm_out2.close(); shm_out2.unlink()

    @classmethod
    def reset(cls):
        """
        Closes RPC proxies and resets the singleton.

        It's a @classmethod because it operates on class state (_instance,
        _initialized), not instance state. cls._instance ensures
        that the class variable is modified without shadowing.

        After reset(), the next call to HAVCditEngine(...) creates a
        new instance, possibly with different parameters.
        Use sparingly (pipeline reload takes ~3.5 min).
        """
        instance = cls._instance
        if instance is not None and instance._proxy_fast is not None:
            try:
                instance._proxy_fast.request_stop()
            except Exception:
                pass
            instance._proxy_fast = None
            instance._proxy_slow = None

        cls._instance = None
        cls._initialized = False

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Singleton orchestrator for single-frame colorization combining DeOldify and 
DDColor with a weighted merge.
"""

class HAVCimageEngine:
    """
    Designed for use cases where a SINGLE B&W frame must be colorized on demand
    inside another VapourSynth filter callback (e.g. CMNET2 retry path with a
    fresh reference). For colorizing entire clips, use vs_sc_deoldify /
    vs_sc_ddcolor / vs_combine_models — they have their own model lifecycle
    scoped to the filter.

    Singleton scope: one engine per process. The DeOldify and DDColor models
    are loaded into GPU memory ONCE, lazily, on first use of colorize_merged().
    Calling HAVCimageEngine() with different parameters after the first call
    has NO EFFECT — the existing instance is returned with its original
    configuration. To switch parameters mid-process, call reset() first
    (not recommended in steady-state).

    Lazy loading: instantiating HAVCimageEngine() does NOT load any model. The
    DeOldify model is loaded on the first colorize_merged() call; the DDColor
    model is loaded on the first DDColorEngine() call inside it. If the engine
    is never used (e.g. retry never triggers in the clip), no VRAM is consumed.

    Typical usage:
        engine = HAVCimageEngine(modelname='video', render_factor=32)
        merged_img = engine.colorize_merged(bw_pil_image)

    Notes:
        - Input PIL image must be RGB ('RGB' mode); B&W input must have the L
          channel replicated into the 3 RGB channels by the caller.
        - Output is PIL RGB at the same resolution as input.
        - DDColor model is hardcoded to 'artistic' (model=1) since it tends to
          give richer colors that better complement DeOldify on hard scenes.
        - Merge weight 0.5 follows HAVC's default for DeOldify+DDColor blending.
    """
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, modelname: str = 'video', render_factor: int = 32,
                 ddcolor_model: int = 1, merge_weight: float = 0.5, color_model: int = 0):
        if self._initialized:
            return

        # Validate parameters early — model loading is deferred to first use.
        _DIT_MAP = {1: "fp4", 2: "int4"}
        self._dit_model = _DIT_MAP.get(color_model)  # None for 0 or out-of-range -> HAVC fallback

        if modelname not in ('video', 'stable', 'artistic'):
            raise ValueError(
                f"HAVCimageEngine: modelname must be 'video', 'stable' or 'artistic', "
                f"got '{modelname}'")
        if not (10 <= render_factor <= 44):
            raise ValueError(
                f"HAVCimageEngine: render_factor must be in [10, 44], got {render_factor}")
        if ddcolor_model not in (0, 1):
            raise ValueError(
                f"HAVCimageEngine: ddcolor_model must be 0 or 1, got {ddcolor_model}")
        if not (0.0 <= merge_weight <= 1.0):
            raise ValueError(
                f"HAVCimageEngine: merge_weight must be in [0.0, 1.0], got {merge_weight}")

        self._modelname = modelname
        self._render_factor = render_factor
        self._ddcolor_model = ddcolor_model
        self._ddcolor_input_size = render_factor * 16
        self._merge_weight = merge_weight

        # Resolve package_dir from this file's location: mcomb.py is in vsslib/,
        # so the parent directory is the vshavc package root, which is what
        # ModelImageRender expects.
        self._package_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

        # Lazily-initialised engines.
        self._deoldify_engine = None  # ModelImageRender
        self._ddcolor_engine = None   # vsddcolor.DDColorEngine
        self._dit_engine = None # DiT engine

        self._initialized = True


    def _ensure_engines(self):

        # Try to load first DiT Model
        if self._dit_model is not None:
            try:
                self._dit_engine = HAVCditEngine(model_precision=self._dit_model)
                self._dit_engine._ensure_connection()
                return
            except Exception:
                self._dit_model = None
                self._dit_engine = None
                pass

        # Load DeOldify and DDColor models on first call. No-op afterwards
        if self._deoldify_engine is None and self._merge_weight != 1:
            # Ensure GPU device is properly configured before loading models.
            # Without this, the model may silently load on CPU (~10x slower).
            from vshavc.deoldify import device
            from vshavc.deoldify.device_id import DeviceId
            if not device.is_gpu():
                device.set(DeviceId.GPU0)

            # Imported lazily to keep the import surface minimal at module load.
            from vshavc.deoldify.visualize import ModelImageRender
            self._deoldify_engine = ModelImageRender(
                package_dir=self._package_dir,
                modelname=self._modelname,
                render_factor=self._render_factor,
            )
        if self._ddcolor_engine is None and self._merge_weight != 0:
            import vsddcolor
            self._ddcolor_engine = vsddcolor.DDColorEngine(
                model=self._ddcolor_model,
                input_size=self._ddcolor_input_size,
            )

    def colorize_merged(self, pil_img: Image.Image) -> Image.Image:
        """
        Colorize a single PIL B&W image as a weighted merge of DeOldify and
        DDColor outputs.

        :param pil_img: B&W input as PIL Image, mode 'RGB' (L replicated 3 times).
        :return:        Merged colorized PIL Image, mode 'RGB', same resolution
                        as input.
        """
        if not isinstance(pil_img, Image.Image):
            raise ValueError("HAVCimageEngine.colorize_merged: input must be a PIL Image")
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')

        self._ensure_engines()

        if self._dit_engine is not None:
            return self._dit_engine.colorize_merged(pil_img)

        if self._merge_weight == 0:
            img_m = self._deoldify_engine.get_transformed_image(pil_img, post_process=True)
        elif self._merge_weight == 1:
            img_m = self._ddcolor_engine.colorize_image(pil_img)
        else:
            img_deoldify = self._deoldify_engine.get_transformed_image(pil_img, post_process=True)
            img_ddcolor = self._ddcolor_engine.colorize_image(pil_img)
            # image_weighted_merge: out = a * (1 - w) + b * w
            # With w=0.5, equally weighted blend.
            img_m = image_weighted_merge(img_deoldify, img_ddcolor, self._merge_weight)

        return img_m

    def reset(self):
        """
        Drop the loaded models and reset the singleton. After reset() the next
        HAVCimageEngine(...) call will reload models fresh, possibly with
        different parameters. Use sparingly.
        """
        # Drop DeOldify reference; relies on Python GC + torch.cuda.empty_cache.
        self._deoldify_engine = None

        # DiTEngine has its own reset() that handles its singleton.
        if self._dit_engine is not None:
            try:
                self._dit_engine.reset()
            except Exception:
                pass
            self._dit_engine = None

        # DDColorEngine has its own reset() that handles its singleton.
        if self._ddcolor_engine is not None:
            try:
                self._ddcolor_engine.reset()
            except Exception:
                pass
            self._ddcolor_engine = None

        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

        HAVCimageEngine._instance = None
        HAVCimageEngine._initialized = False
