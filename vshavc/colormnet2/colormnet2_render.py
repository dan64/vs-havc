"""
-------------------------------------------------------------------------------
Author: Dan64
Date: 2024-09-14
version:
LastEditors: Dan64
LastEditTime: 2026-05-04
-------------------------------------------------------------------------------
Description:
-------------------------------------------------------------------------------
CMNET2 rendering class for Vapoursynth.
"""
import os
from os import path
import torch
import gc
import warnings
import torch.backends.cudnn as cudnn
from skimage import color
from torchvision import transforms
from torchvision.transforms import InterpolationMode
# import torch.nn.functional as Ff
import torch.nn.functional as F
from PIL import Image
import numpy as np
import math
import vapoursynth as vs
from vsdeoldify.vsslib.constants import DEF_MAX_MEMORY_FRAMES

from vsdeoldify.colormnet2.dataset.range_transform import im_normalization, im_rgb2lab_normalization, ToTensor, RGB2Lab
from vsdeoldify.colormnet2.model.network import ColorMNet
from vsdeoldify.colormnet2.inference.inference_core import InferenceCore
from vsdeoldify.colormnet2.util.transforms import lab2rgb_transform_PIL
from vsdeoldify.vsslib.mcomb import HAVCimageEngine
from vsdeoldify.vsslib.imfilters import image_weighted_merge
from vsdeoldify.colormnet2.colormnet2_logbuffer import log_warning as _buf_warning, log_debug as _buf_debug
from vsdeoldify.vsslib.vsutils import MessageType, HAVC_LogMessage

import warnings

warnings.filterwarnings("ignore")

os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ["NUMEXPR_MAX_THREADS"] = "8"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class ColorMNetRender2:
    """
    This class is used to render a frame at a time
    """
    _instance = None
    _initialized = False
    _frame_size = None
    ref_img: Image = None
    ref_img_valid: Image = None
    reset_on_ref_update = False
    img: Image = None
    first_mask_loaded: bool = False
    max_memory_frames: int = None
    frame_count: int = 0
    ref_count: int = 0
    ref_count_prv: int = 0
    total_colored_frames = 0
    processor: InferenceCore = None
    encode_mode: int = None  # 0: remote, 1: async (local)
    _reset_cond_1_was_active = False
    _reset_cond_2_was_active = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, image_size: int = -1, vid_length: int = None, enable_resize: bool = False,
                 encode_mode: int = None, propagate: bool = False, max_memory_frames: int = None,
                 reset_on_ref_update: bool = True, top_k: int = 30, mem_every: int = 5,
                 retry_mmsp_threshold: float = -1.0, retry_perm_share_threshold: float = 0.25,
                 project_dir: str = None):

        self.reset_on_ref_update = reset_on_ref_update  # deprecated with XMem2
        self.top_k = top_k
        self.mem_every = mem_every
        self.enable_resize = enable_resize
        # Edge-triggered state for VRAM reset warnings:
        # we log only on the False -> True transition to avoid spamming
        # the console when the condition stays true for many frames.
        self._reset_cond_1_was_active = False
        self._reset_cond_2_was_active = False
        if project_dir is None:
            project_dir = os.path.dirname(os.path.realpath(__file__))
        self.project_dir = project_dir
        self._frame_size = image_size
        if encode_mode is None:
            self.encode_mode = 0
        else:
            self.encode_mode = encode_mode
        if max_memory_frames is None or max_memory_frames == 0:
            self.max_memory_frames = min(DEF_MAX_MEMORY_FRAMES, vid_length)
        else:
            self.max_memory_frames = min(DEF_MAX_MEMORY_FRAMES, max_memory_frames)
        self.total_colored_frames = 0

        # Retry-trigger thresholds. Stored as attributes so _colorize_config_init
        # can pick them up. See reference_frame_missing() for usage.
        self._retry_mmsp_threshold = retry_mmsp_threshold
        self._retry_perm_share_threshold = retry_perm_share_threshold

        # Lazy init: havc_engine is created on first colorize_frame_with_retry()
        # call to avoid loading DeOldify+DDColor for sessions that don't enable
        # retry. See colorize_frame_with_retry() below.
        self._havc_engine = None

        self._colorize_config_init(image_size, vid_length, propagate)

        if not self._initialized:
            self._colorize_model_init(vid_length)
            self._initialized = True

    def _colorize_config_init(self, image_size: int = -1, vid_length: int = 100, propagate: bool = False):
        """
            size - resize min. side to size. Does nothing if <0.
            Resize the shorter side to this size. -1 to use original resolution.
       """

        cudnn.benchmark = True
        torch.autograd.set_grad_enabled(False)

        self.config = {}
        # model checkpoint location
        self.config['model'] = path.join(self.project_dir, 'weights/DINOv2FeatureV6_LocalAtten_s2_154000.pth')
        # Whether the provided reference frame is exactly the first input frame
        self.config['FirstFrameIsNotExemplar'] = not propagate
        # dataset setting
        # For generic (G) evaluation, point to a folder that contains "JPEGImages" and "Annotations"
        self.config['dataset'] = 'D16_batch'  # D16/D17/Y18/Y19/LV1/LV3/G
        # Long-term memory options
        self.config['max_mid_term_frames'] = min(10, vid_length)  # T_max in paper, decrease to save memory
        self.config['min_mid_term_frames'] = min(5, int(
            self.config['max_mid_term_frames'] / 2))  # T_min in paper, decrease to save memory
        self.config[
            'max_long_term_elements'] = self.max_memory_frames  # LT_max in paper, increase if objects disappear for a long time
        self.config['num_prototypes'] = 128  # P in paper
        self.config['top_k'] = self.top_k
        self.config['mem_every'] = min(self.mem_every, self.config[
            'max_mid_term_frames'])  # r in paper. Increase to improve running speed
        self.config['deep_update_every'] = -1  # Leave -1 normally to synchronize with mem_every
        # Multi-scale options
        self.config['save_scores'] = False
        self.config['size'] = image_size  # Resize the shorter side to this size. -1 to use original resolution
        self.config['disable_long_term'] = False
        self.config['enable_long_term'] = not self.config['disable_long_term']
        # Retry-trigger thresholds for reference_frame_missing(). Set by __init__.
        self.config['retry_mmsp_threshold'] = self._retry_mmsp_threshold
        self.config['retry_perm_share_threshold'] = self._retry_perm_share_threshold
        # Secondary trigger thresholds: capture scenes with NO suitable reference
        # in perm_mem at all (mmsp very negative AND perm_share near zero).
        # Hardcoded — derived empirically, not user-tunable for now.
        self.config['retry_mmsp_low_threshold'] = -3.0
        self.config['retry_perm_share_low_threshold'] = 0.03
        self.config['enable_retry'] = self._retry_perm_share_threshold > 0

        if self.config['enable_retry']:
            _buf_warning(f"ColorMNetRender2(): enabled missing Reference colorization with threshold={self._retry_perm_share_threshold}")

        if image_size < 0:
            self.im_transform = transforms.Compose([
                RGB2Lab(),
                ToTensor(),
                im_rgb2lab_normalization,
            ])
        else:
            self.im_transform = transforms.Compose([
                transforms.ToTensor(),
                im_normalization,
                transforms.Resize(image_size, interpolation=InterpolationMode.BILINEAR),
            ])
        self.size = image_size
        self.ref_img = None
        self.img = None

    def _colorize_model_init(self, vid_length: int = 100):

        cudnn.benchmark = True
        torch.autograd.set_grad_enabled(False)

        # Model setup
        self.network = ColorMNet(self.config, self.config['model']).cuda().eval()
        self.model_weights = torch.load(self.config['model'])
        self.network.load_weights(self.model_weights, init_as_zero_if_needed=True)
        self.vid_length = vid_length
        self.config['enable_long_term_count_usage'] = (
                self.config['enable_long_term'] and
                (self.vid_length
                 / (self.config['max_mid_term_frames'] - self.config['min_mid_term_frames'])
                 * self.config['num_prototypes'])
                >= self.config['max_long_term_elements']
        )
        self.processor = InferenceCore(self.network, config=self.config)

    def set_config(self, param_name: str = None, param_value: any = None):
        self.config[param_name] = param_value
        self.processor.update_config(self.config)

    def set_ref_frame(self, frame_ref: Image = None, frame_propagate: bool = False):
        self.ref_img = frame_ref
        self.config['FirstFrameIsNotExemplar'] = not frame_propagate
        if not (frame_ref is None):
            self.ref_img_valid = frame_ref
            if self.frame_count > 0:
                self.ref_count_prv = self.ref_count
            else:
                self.ref_count_prv = 0
            self.ref_count = self.frame_count

    def _emit_warning(self, msg: str) -> None:
        """Route a warning to the right log sink based on encode_mode."""
        if self.encode_mode == 0:
            # Remote (RPC server): goes through the buffer, the client will
            # forward it to VS's log on the next poll.
            _buf_warning(msg)
        else:
            # async (local): we're already in the VS process, log directly.
            HAVC_LogMessage(MessageType.WARNING, msg)

    def _emit_debug(self, msg: str) -> None:
        """Route a debug message to the right log sink based on encode_mode."""
        if self.encode_mode == 0:
            _buf_debug(msg)
        else:
            # Promote to WARNING for visibility, mirroring the client-side
            # promotion done for the RPC route.
            HAVC_LogMessage(MessageType.WARNING, f"[DEBUG] {msg}")

    def preload_reference(self, ref_img: Image):
        """
        Preloads a reference frame into perm_mem before starting colorization.
        Can be called N times consecutively.
        """
        if self.processor is None:
            return

        img = self.im_transform(ref_img)
        img_l = img[:1, :, :]
        img_lll = img_l.repeat(3, 1, 1).cuda()
        img_ab = img[1:3, :, :].cuda()

        if self.processor.all_labels is None:
            self.processor.set_all_labels(list(range(1, 3)))

        self.processor.load_reference(img_lll, img_ab)
        # Free memory cache
        torch.cuda.empty_cache()

    def slide_permanent_memory(self, n_frames: int):
        """
        Removes the first n_frames reference frames from permanent memory.
        Call when perm_mem exceeds the desired window size.
        """
        if self.processor is not None:
            self.processor.memory.slide_permanent_memory(n_frames)
            # Free memory cache
            torch.cuda.empty_cache()

    def get_perm_mem_frame_count(self) -> int:
        """Returns the number of reference frames currently in perm_mem."""
        if self.processor is not None:
            return self.processor.memory._perm_frame_count
        return 0

    def get_last_match_metrics(self):
        """
        Returns the (mmsp, perm_share) tuple from the most recent colorize_frame call.

        - mmsp           : mean_max_sim_perm. Per-position max similarity restricted
                           to the perm_mem block, averaged over query positions.
                           Closer to 0 = better structural match against perm_mem.
                           NaN if perm_mem was not engaged on the last call (e.g.
                           very first frame, or no reference frames loaded).
        - perm_share     : fraction of post-softmax attention mass placed on
                           perm_mem tokens. In [0, 1]. NaN if perm_mem not engaged.

        Use these metrics to assess whether the last colorized frame may benefit
        from injecting an alternative reference (e.g. a DDColor-colored version
        of the same frame) and re-colorizing.

        Returns:
            tuple[float, float] : (mmsp, perm_share). Both NaN if the processor
                                   has not been initialized or no match_memory
                                   call has happened yet.
        """
        enable_retry = self.config.get('enable_retry', False)

        if self.processor is None or self.processor.memory is None or not enable_retry:
            return float('nan'), float('nan')
        return self.processor.memory._last_mmsp, self.processor.memory._last_perm_share

    def reference_frame_missing(self):
        """
        Decision rule (OR of two regimes):
          Rule 1 — high structural match but low usage:
              mmsp > mmsp_threshold  AND  perm_share < perm_share_threshold
            Captures scenes where perm_mem contains structurally similar but
            contextually wrong references (e.g. day exterior reference for a
            night interior scene).
          Rule 2 — structural match absent and perm_mem ignored:
              mmsp < mmsp_low_threshold  AND  perm_share < perm_share_low_threshold
            Captures scenes that have NO suitable reference in perm_mem at all
            — the model finds nothing structurally similar AND barely uses perm_mem.
            This signals a coverage gap that benefits from injecting a fresh
            reference (e.g. DDColor merged) for the scene.

        Thresholds are read from self.config:
            'retry_mmsp_threshold'             (default -1.0)
            'retry_perm_share_threshold'       (default 0.25)
            'retry_mmsp_low_threshold'         (default -3.0, hardcoded)
            'retry_perm_share_low_threshold'   (default 0.03, hardcoded)
        """
        enable_retry = self.config.get('enable_retry', False)

        if not enable_retry:
            return False

        mmsp, perm_share = self.get_last_match_metrics()
        if math.isnan(mmsp) or math.isnan(perm_share):
            return False

        mmsp_t = self.config.get('retry_mmsp_threshold', -1.0)
        ps_t = self.config.get('retry_perm_share_threshold', 0.25)
        mmsp_low_t = self.config.get('retry_mmsp_low_threshold', -3.0)
        ps_low_t = self.config.get('retry_perm_share_low_threshold', 0.03)

        rule_1 = (mmsp > mmsp_t) and (perm_share < ps_t)
        rule_2 = (mmsp < mmsp_low_t) and (perm_share < ps_low_t)
        return rule_1 or rule_2

    def colorize_frame_with_retry(self, ti: int = None, frame_i: Image = None,
                                  retry_blend_weight: float = 0.85,
                                  merge_engine_weight: float = 0.40,
                                  render_factor: int = 24,
                                  debug: bool = True) -> Image:
        """
        Single-call colorize + auto-retry. Equivalent to:

            img_color = self.colorize_frame(ti, frame_i)
            if self.reference_frame_missing():
                img_ref = havc_engine.colorize_merged(frame_i)
                img_merged = image_weighted_merge(img_color, img_ref, retry_blend_weight)
                self.set_ref_frame(img_merged, frame_propagate=False)
                img_color = self.colorize_frame(ti, frame_i)
            return img_color

        Used by both local and remote callbacks. The HAVCimageEngine is lazily
        instantiated on first retry, so sessions that never trigger retry pay
        no DeOldify/DDColor VRAM cost.

        Parameters:
            retry_blend_weight    : weight of img_ref in the final blend
                                     image_weighted_merge(img_color, img_ref, w).
                                     Default 0.85 means 15% CMNET2 (bad) + 85%
                                     merged-ref. Calibrated empirically.
            merge_engine_weight   : merge_weight for HAVCimageEngine
                                     (0.40 = 60% DeOldify + 40% DDColor).
            render_factor         : DeOldify render factor for HAVCimageEngine.
            debug                 : if true is displayed a debug message.

        Returns the colorized frame as PIL Image.
        """
        if frame_i is None:
            return None

        # First pass: standard colorize.
        img_color = self.colorize_frame(ti, frame_i)

        # Check whether perm_mem coverage is insufficient for this frame.
        if self.reference_frame_missing():
            # Lazy-init the merge engine on first need.
            if self._havc_engine is None:
                self._havc_engine = HAVCimageEngine(
                    render_factor=render_factor,
                    merge_weight=merge_engine_weight,
                )
            if debug:
                mmsp, perm_share = self.get_last_match_metrics()
                _buf_warning(f"Frame {ti} retry: injected merged ref (mmsp={mmsp:.3f}, perm_share={perm_share:.3f})")
            # Generate a clean reference for this frame.
            img_ref = self._havc_engine.colorize_merged(frame_i)
            # Blend with the bad first-pass output for smooth scene transitions.
            img_merged = image_weighted_merge(img_color, img_ref, weight=retry_blend_weight)
            # Inject as new perm_mem reference and re-colorize.
            self.set_ref_frame(img_merged, frame_propagate=False)
            img_color = self.colorize_frame(ti, frame_i)

        return img_color

    def colorize_batch_frames(self, frame_list: list[Image] = None, ref_list: list[Image] = None,
                              frame_propagate: bool = False) -> list[Image]:
        nframes = len(frame_list)
        frames_colored = []
        for i in range(0, nframes, 1):
            frame_i = frame_list[i]
            ref_i = ref_list[i]
            self.set_ref_frame(ref_i, frame_propagate)
            col_i = self.colorize_frame(i, frame_i)
            frames_colored.append(col_i)
        return frames_colored

    def get_frame_count(self) -> int:
        return self.frame_count

    def colorize_frame(self, ti: int = None, frame_i: Image = None, lab_mode: str = "gpu") -> Image:

        self.total_colored_frames += 1

        gpu_mem_free, gpu_mem_total = torch.cuda.mem_get_info()
        gpu_mem_k = round(gpu_mem_free / 1024 / 1024, 1)
        reset_cond_1 = (gpu_mem_k < 500)
        reset_cond_2 = (gpu_mem_k < 1500)
        # if self.frame_count >= self.vid_length or self.frame_count >= self.max_memory_frames:
        reset_cond_1 = (gpu_mem_k < 500)
        reset_cond_2 = (gpu_mem_k < 1500)

        # Edge-triggered logging: warn only on False -> True transitions.
        if reset_cond_1 and not self._reset_cond_1_was_active:
            self._emit_warning(
                f"VRAM critical ({gpu_mem_k:.0f} MB free) at frame "
                f"{self.total_colored_frames}: rebuilding inference core "
                f"(frame_count was {self.frame_count}/{self.max_memory_frames})"
            )
        elif reset_cond_2 and not reset_cond_1 and not self._reset_cond_2_was_active:
            self._emit_debug(
                f"VRAM low ({gpu_mem_k:.0f} MB free) at frame "
                f"{self.total_colored_frames}: sliding 70% of permanent memory"
            )
        self._reset_cond_1_was_active = reset_cond_1
        self._reset_cond_2_was_active = reset_cond_2

        # if self.frame_count >= self.vid_length or self.frame_count >= self.max_memory_frames:
        if reset_cond_1:
            self.frame_count = 0
            del self.processor
            gc.collect()
            torch.cuda.empty_cache()
            self.config['FirstFrameIsNotExemplar'] = True  # because the reference image is the previous colored frame
            self.processor = InferenceCore(self.network, config=self.config)
            data = self.get_image(ti, frame_i, self.ref_img_valid)
        else:
            if reset_cond_2:
                # VRAM below threshold: aggressive slide of 70% of perm_mem
                n = int(self.get_perm_mem_frame_count() * 0.7)
                self.slide_permanent_memory(n)
                torch.cuda.empty_cache()
            data = self.get_image(ti, frame_i, self.ref_img)
            self.frame_count += 1

        rgb = data['rgb'].cuda()[0]

        msk = data.get('mask')
        if not self.config['FirstFrameIsNotExemplar']:
            msk = msk[:, 1:3, :, :] if msk is not None else None

        info = data['info']
        # frame = '{:0>5}'.format(info['frame'])
        shape = info['shape']
        need_resize = info['need_resize']

        if not self.first_mask_loaded:
            if msk is not None:
                self.first_mask_loaded = True
            else:
                # no point to do anything without a mask
                return frame_i

        # Map possibly non-continuous labels to continuous ones
        if msk is not None:
            msk = torch.Tensor(msk[0]).cuda()
            if need_resize:
                msk = self.resize_mask(msk.unsqueeze(0))[0]
            self.processor.set_all_labels(list(range(1, 3)))
            labels = range(1, 3)
        else:
            labels = None

        # Run the model on this frame
        is_last_frame = self.vid_length == self.total_colored_frames - 1  # (ti == (self.vid_length - 1))
        if self.config['FirstFrameIsNotExemplar']:

            if msk is None:
                prob = self.processor.step_AnyExemplar(rgb, None, None, labels, end=is_last_frame)
            else:
                prob = self.processor.step_AnyExemplar(rgb, msk[:1, :, :].repeat(3, 1, 1), msk[1:3, :, :], labels,
                                                       end=is_last_frame)
        else:
            prob = self.processor.step(rgb, msk, labels, end=is_last_frame)

        # Upsample to original size if needed
        if need_resize:
            prob = F.interpolate(prob.unsqueeze(1), shape, mode='bilinear', align_corners=False)[:, 0]

        # return the colored frame
        out_img_final = lab2rgb_transform_PIL(torch.cat([rgb[:1, :, :], prob], dim=0), mode = lab_mode)
        out_img_final = out_img_final * 255
        out_img_final = out_img_final.astype(np.uint8)

        out_pil_img = Image.fromarray(out_img_final)

        self.save_last_image(out_pil_img)

        # empty torch cache
        torch.cuda.empty_cache()

        return out_pil_img

    def get_image(self, idx: int = None, img: Image = None, ref_img: Image = None) -> dict:

        shape = np.array(img).shape[:2]

        img = self.im_transform(img)
        img_l = img[:1, :, :]
        img_lll = img_l.repeat(3, 1, 1)

        data = {}
        info = {}

        if not (ref_img is None):
            mask = self.im_transform(ref_img)

            # keep L channel of reference image in case First frame is not exemplar
            # mask_ab = mask[1:3,:,:]
            # data['mask'] = mask_ab
            data['mask'] = torch.unsqueeze(mask, dim=0)

        info['shape'] = [torch.tensor(shape[0]), torch.tensor(shape[1])]
        info['need_resize'] = not (self.size < 0)
        info['frame'] = idx
        data['rgb'] = torch.unsqueeze(img_lll, dim=0)
        data['info'] = info

        return data

    def save_last_image(self, img: Image = None):
        self.img = img
        self.ref_img_valid = img

    def resize_mask(self, mask):
        # mask transform is applied AFTER mapper, so we need to post-process it in eval.py
        h, w = mask.shape[-2:]
        min_hw = min(h, w)
        return F.interpolate(mask, (int(h / min_hw * self.size), int(w / min_hw * self.size)),
                             mode='nearest')

    @classmethod
    def reset(cls):
        """
        Tear down the singleton instance, releasing GPU memory and clearing all
        accumulated state (perm_mem, work_mem, long_mem, model weights).

        Use this when running multiple unrelated colorization jobs in the same
        Python process (e.g. VapourSynth-Editor job server, which reuses the
        interpreter across jobs). Without reset, perm_mem from job N may
        contaminate job N+1 and produce odd colors at the start of the second
        clip.

        After reset(), the next ColorMNetRender2(...) call will reload the
        CMNET2 model from disk (~5-10 seconds) and start with empty memory.

        Safe to call multiple times. No-op if the singleton is not initialized.
        """
        if cls._instance is None:
            return

        inst = cls._instance

        # Drop the inference processor and its internal MemoryManager
        # (perm_mem, work_mem, long_mem all live inside).
        if hasattr(inst, 'processor'):
            try:
                # Some XMem-style processors expose a clear/dispose hook;
                # if not, just dropping the reference is enough — Python GC
                # will collect once no references remain.
                del inst.processor
            except Exception:
                pass

        # Drop the CMNET2 network weights from GPU.
        if hasattr(inst, 'network'):
            try:
                del inst.network
            except Exception:
                pass

        # Tear down the lazily-loaded merge engine, if present. HAVCimageEngine
        # has its own reset() that handles its DDColor + DeOldify singletons.
        if getattr(inst, '_havc_engine', None) is not None:
            try:
                inst._havc_engine.reset()
            except Exception:
                pass
            inst._havc_engine = None

        # Reset edge-triggered VRAM warning flags so the next session starts
        # with a clean slate.
        if hasattr(inst, '_reset_cond_1_was_active'):
            inst._reset_cond_1_was_active = False
        if hasattr(inst, '_reset_cond_2_was_active'):
            inst._reset_cond_2_was_active = False

        # Force a CUDA cache flush so VRAM is actually released, not just
        # marked as free in PyTorch's allocator.
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        # Force a Python GC pass to drop any lingering tensor references that
        # would otherwise pin GPU memory.
        try:
            import gc
            gc.collect()
        except Exception:
            pass

        # Clear class-level state so the next ColorMNetRender2(...) constructs
        # a fresh instance with model reload.
        cls._instance = None
        cls._initialized = False
