"""
-------------------------------------------------------------------------------
Author: Dan64
Date: 2025-09-28
version:
LastEditors: Dan64
LastEditTime: 2026-05-21
-------------------------------------------------------------------------------
Description:
-------------------------------------------------------------------------------
CMNET2 frame server class for Vapoursynth.
"""
import os
from os import path
import warnings
import torch
import math
import numpy as np
from PIL import Image
import threading
from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.server import SimpleXMLRPCRequestHandler
import tempfile
import datetime
from vshavc.colormnet2 import ColorMNetRender2
from vshavc.colormnet2.colormnet2_utils import byte_array_to_image, image_to_byte_array

# weights are not duplicated
package_dir = os.path.dirname(os.path.realpath(__file__)).replace("colormnet2", "colormnet")

from vshavc.colormnet2.colormnet2_logbuffer import ServerLogBuffer, log_warning, log_info, log_debug

class ColorMNetRPCServer2:
    server_address: str = None
    server_port: int = None
    server: SimpleXMLRPCServer = None

    # Restrict to a particular path.
    class RequestHandler(SimpleXMLRPCRequestHandler):
        rpc_paths = ('/RPC2',)
        # Use HTTP/1.1 to enable persistent connections (keep-alive). Without
        # this the default HTTP/1.0 closes the TCP connection after every
        # response, causing client-side ephemeral port exhaustion (WinError
        # 10048) on long-running sessions with tens of thousands of RPC calls.
        protocol_version = 'HTTP/1.1'

    def __init__(self, server_address: str = '127.0.0.1', server_port: int = 0):
        self.server_address = server_address
        self.server = SimpleXMLRPCServer(addr=(server_address, server_port), allow_none=True,
                                         requestHandler=self.RequestHandler, use_builtin_types=True, logRequests=False)
        self.server_port = self.server.socket.getsockname()[1]
        self.server.register_introspection_functions()
        self.server.register_instance(self.ColorMNetService(), allow_dotted_names=True)

    def shutdown(self):
        self.server.shutdown()

    class ColorMNetService:
        render: ColorMNetRender2 = None
        _preload_counter: int = 0

        def initialize(self, image_size: int = -1, vid_length: int = 1000, enable_resize: bool = False,
                       encode_mode: int = 0, propagate: bool = False, max_memory_frames: int = None,
                       reset_on_ref_update: bool = True, retry_mmsp_threshold: float = -1.0,
                       retry_perm_share_threshold: float = 0.30, retry_model: int = 0,
                       backbone: str = "dinov3"):

            # Force a fresh render on reinitialization (e.g. VSEdit loop).
            # The render is a singleton and would otherwise keep stale state.
            if self.render is not None:
                log_warning("CMNET2 Render state reset")
                self.render.reset_state()
            self.render = ColorMNetRender2(image_size, vid_length, enable_resize, encode_mode, propagate,
                                           max_memory_frames, reset_on_ref_update=reset_on_ref_update,
                                           retry_mmsp_threshold=retry_mmsp_threshold,
                                           retry_perm_share_threshold=retry_perm_share_threshold,
                                           retry_model = retry_model,
                                           project_dir=package_dir, backbone=backbone)


        def SetRefImage(self, img_byte_array: bytes, frame_propagate: bool = False):
            img = byte_array_to_image(img_byte_array)
            if self.render is not None:
                self.render.set_ref_frame(img, frame_propagate)
            else:
                log_warning("CMNET2 Render is not initialized")

        def SetRefImageNone(self, frame_propagate: bool = False):
            if self.render is not None:
                self.render.set_ref_frame(None, frame_propagate)
            else:
                log_warning("CMNET2 Render is not initialized")

        def IsInitialized(self) -> bool:
            return self.render is not None

        def ColorizeImage(self, img_byte_array: bytes, ti: int = None):
            img = byte_array_to_image(img_byte_array)
            if self.render is not None:
                img_colored = self.render.colorize_frame(ti, img)
                img_byte_array = image_to_byte_array(img_colored)
                return img_byte_array
            else:
                log_warning("CMNET2 Render is not initialized")
                return img_byte_array

        def ColorizeImageWithRetry(self, img_byte_array: bytes, ti: int = None,
                                   retry_blend_weight: float = 0.85,
                                   merge_engine_weight: float = 0.40,
                                   render_factor: int = 24) -> bytes:
            """
            Thin RPC wrapper around ColorMNetRender2.colorize_frame_with_retry().
            All retry logic (HAVCimageEngine instantiation, merge, blend, ref
            injection, recolor) lives in the render. See render docstring.

            Parameters:
                img_byte_array        : image to be colorized
                ti                    : frame number of image to be colorized
                retry_blend_weight    : weight of img_ref in the final blend
                                         image_weighted_merge(img_color, img_ref, w).
                                         Default 0.85 means 15% CMNET2 (bad) +
                                         85% merged-ref. Calibrated empirically.
                merge_engine_weight   : merge_weight for HAVCimageEngine
                                         (0.40 = 60% DeOldify + 40% DDColor).
                render_factor         : DeOldify render factor for HAVCimageEngine.

            Returns the colorized frame as bytes (same format as ColorizeImage).
            If the render is not initialized, returns the input bytes unchanged.
            """
            img = byte_array_to_image(img_byte_array)
            if self.render is None:
                log_warning("CMNET2 Render is not initialized")
                return img_byte_array
            img_colored = self.render.colorize_frame_with_retry(
                ti, img, retry_blend_weight, merge_engine_weight, render_factor)
            return image_to_byte_array(img_colored)

        def GetFrameCount(self) -> int:
            if self.render is not None:
                return self.render.get_frame_count()
            log_warning("CMNET2 Render is not initialized")
            return 0

        def PreloadReference(self, img_byte_array: bytes, frame_idx: int = None):
            if self.render is not None:
                img = byte_array_to_image(img_byte_array)
                self.render.preload_reference(img, frame_idx=frame_idx)
                self._preload_counter += 1
            else:
                log_warning("CMNET2 Render is not initialized")

        def SlidePermanentMemory(self, n_frames: int):
            if self.render is not None:
                self.render.slide_permanent_memory(n_frames)
            else:
                log_warning("CMNET2 Render is not initialized")

        def GetPermMemFrameCount(self) -> int:
            if self.render is not None:
                return self.render.get_perm_mem_frame_count()
            log_warning("CMNET2 Render is not initialized")
            return 0

        def GetLastMatchMetrics(self) -> list:
            """
            Returns the (mmsp, perm_share) tuple from the most recent ColorizeImage
            call. Both values are serialized as None when NaN (XMLRPC does not
            handle NaN natively); the client converts None back to float('nan').

            Returns:
                [mmsp_or_none, perm_share_or_none]  (XMLRPC array of length 2)
            """
            if self.render is None:
                log_warning("CMNET2 Render is not initialized")
                return [None, None]
            mmsp, perm_share = self.render.get_last_match_metrics()
            # Serialize NaN as None for XMLRPC compatibility.
            mmsp_payload = None if math.isnan(mmsp) else mmsp
            perm_share_payload = None if math.isnan(perm_share) else perm_share
            return [mmsp_payload, perm_share_payload]

        def ReferenceFrameMissing(self) -> bool:
            """
            Returns True when the most recent ColorizeImage call indicates that
            an additional reference frame is likely needed for proper
            colorization. Thresholds are the ones configured at initialize().

            Stateless: each call evaluates the latest metrics independently.
            Returns False if the render is not initialized or metrics are NaN.
            """
            if self.render is None:
                log_warning("CMNET2 Render is not initialized")
                return False
            return self.render.reference_frame_missing()

        def PollLogMessages(self) -> list:
            """Return and clear pending server log messages.

            Each item is a 2-element list [level:int, text:str] to keep the
            XML-RPC payload minimal. Level matches MessageType integer values.
            """
            return [list(m) for m in ServerLogBuffer().drain()]

        # ------------------------------------------------------------------
        # Shared-memory variants — zero-copy transport (same-host only).
        #
        # The CLIENT owns and manages all SharedMemory segments (create/unlink).
        # The server only attaches (create=False) and detaches — no cleanup
        # responsibility. This mirrors the protocol used in DiTServerRPC.
        # ------------------------------------------------------------------

        @staticmethod
        def _shm_to_img(shm_name: str, height: int, width: int):
            """Attach to a client-owned SharedMemory segment and return a PIL Image."""
            from multiprocessing.shared_memory import SharedMemory
            shm = SharedMemory(name=shm_name, create=False)
            try:
                arr = np.ndarray((height, width, 3), dtype=np.uint8, buffer=shm.buf)
                return Image.fromarray(arr.copy(), mode="RGB")
            finally:
                shm.close()

        @staticmethod
        def _img_to_shm(shm_name: str, height: int, width: int, img: Image.Image):
            """Write a PIL Image into a client-owned SharedMemory segment."""
            from multiprocessing.shared_memory import SharedMemory
            shm = SharedMemory(name=shm_name, create=False)
            try:
                arr = np.ndarray((height, width, 3), dtype=np.uint8, buffer=shm.buf)
                arr[:] = np.array(img)
            finally:
                shm.close()

        def SetRefImageShm(self, shm_name: str, height: int, width: int,
                           frame_propagate: bool = False):
            """Shared-memory variant of SetRefImage."""
            img = self._shm_to_img(shm_name, height, width)
            if self.render is not None:
                self.render.set_ref_frame(img, frame_propagate)
            else:
                log_warning("CMNET2 Render is not initialized")

        def ColorizeImageShm(self, shm_in_name: str, shm_out_name: str,
                             height: int, width: int, ti: int = None):
            """Shared-memory variant of ColorizeImage."""
            img = self._shm_to_img(shm_in_name, height, width)
            if self.render is not None:
                img_colored = self.render.colorize_frame(ti, img)
                self._img_to_shm(shm_out_name, height, width, img_colored)
            else:
                log_warning("CMNET2 Render is not initialized")
                self._img_to_shm(shm_out_name, height, width, img)

        def ColorizeImageWithRetryShm(self, shm_in_name: str, shm_out_name: str,
                                      height: int, width: int, ti: int = None,
                                      retry_blend_weight: float = 0.85,
                                      merge_engine_weight: float = 0.40,
                                      render_factor: int = 24):
            """Shared-memory variant of ColorizeImageWithRetry."""
            try:
                img = self._shm_to_img(shm_in_name, height, width)
                if self.render is None:
                    log_warning(f"CMNET2 Render is not initialized, return original frame ti={ti}")
                    self._img_to_shm(shm_out_name, height, width, img)
                    return
                img_colored = self.render.colorize_frame_with_retry(
                    ti, img, retry_blend_weight, merge_engine_weight, render_factor)
                self._img_to_shm(shm_out_name, height, width, img_colored)
            except Exception as e:
                log_warning(f"ColorizeImageWithRetryShm failed at ti={ti}: {type(e).__name__}: {e}")
                raise

        def PreloadReferenceShm(self, shm_name: str, height: int, width: int, frame_idx: int = None):
            """Shared-memory variant of PreloadReference."""
            if self.render is not None:
                img = self._shm_to_img(shm_name, height, width)
                self.render.preload_reference(img, frame_idx=frame_idx)
                self._preload_counter += 1
            else:
                log_warning("CMNET2 Render is not initialized")


    def start_server(self):
        log_info("Start CMNET2 server, listening on : " + str(self.server.server_address))
        self.server.serve_forever()


class ColorMNetServer2:
    _instance = None
    _initialized = False
    rpc_server: ColorMNetRPCServer2 = None
    rpc_thread: threading.Thread = None
    context: any = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, server_port: int = 0):
        if not self._initialized:
            try:
                server_address = '127.0.0.1'
                self.rpc_server = ColorMNetRPCServer2(server_address, server_port)
                self.rpc_thread = threading.Thread(target=self.rpc_server.start_server, name="RPCServer-2", daemon=True)
                self._initialized = True
            except Exception as exe:
                self._initialized = False
                raise RuntimeError(f"CMNET2 server error allocating port {server_port}: {exe}")

    def run_server(self):
        if self.rpc_thread is None:
            return None
        if not self.rpc_thread.is_alive():
            self.rpc_thread.start()
        return self

    def get_port(self):
        return self.rpc_server.server_port

    def close_server(self):
        if self.rpc_thread.is_alive():
            log_warning("CMNET2 server is alive, stop it")
            self.rpc_server.shutdown()
            self.rpc_thread.join()
        log_info("CMNET2 server closed")
