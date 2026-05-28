"""
-------------------------------------------------------------------------------
Author: Dan64
Date: 2025-09-28
version:
LastEditors: Dan64
LastEditTime: 2026-04-30
-------------------------------------------------------------------------------
Description:
-------------------------------------------------------------------------------
CMNET2 frame client class for Vapoursynth.
"""
import os
import math
from PIL import Image
import warnings
import xmlrpc.client
from vshavc.colormnet2.colormnet2_utils import *
from vshavc.vsslib.vsutils import MessageType, HAVC_LogMessage

class ColorMNetClient2:
    _instance = None
    _initialized = False
    server_address: str = None
    server_port: int = None
    server: xmlrpc.client.ServerProxy = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, image_size: int = -1, vid_length: int = 1000, enable_resize: bool = False,
                 encode_mode: int = 0, propagate: bool = False, max_memory_frames: int = None,
                 reset_on_ref_update: bool = True, retry_mmsp_threshold: float = -1.0,
                 retry_perm_share_threshold: float = 0.30, server_port: int = None):
        if not self._initialized:
            server_address = '127.0.0.1'
            if server_port is None:
                HAVC_LogMessage(MessageType.CRITICAL, "CMNET2 Client(): server port is None")
                return
            self.server_address = server_address
            self.server_port = server_port
            # Connect to a RPC instance; all the methods of the instance are
            # published as XML-RPC methods.
            self.uri = f"http://{server_address}:{server_port}"
            try:
                self.server = xmlrpc.client.ServerProxy(uri=self.uri, allow_none=True, use_builtin_types=True)
                self.server.initialize(image_size, vid_length, enable_resize, encode_mode, propagate,
                                       max_memory_frames, reset_on_ref_update,
                                       retry_mmsp_threshold, retry_perm_share_threshold)
                self._initialized = True
            except Exception as exe:
                HAVC_LogMessage(MessageType.CRITICAL,
                                f"CMNET2 Client(): init failed [{type(exe).__name__}]: {exe}")

    def is_initialized(self) -> bool:
        return self.server.IsInitialized()

    def get_frame_count(self) -> int:
        return self.server.GetFrameCount()

    def _safe_remote_call(self, fn, *args, max_attempts=3, base_delay=0.1):
        """Retry an RPC call on OSError with linear backoff."""
        # Retry on transient network errors (e.g. ephemeral port exhaustion
        # on long-running sessions). 3 attempts with linear backoff cover
        # the vast majority of cases. Re-raise on persistent failure.
        for attempt in range(max_attempts):
            try:
                return fn(*args)
            except OSError as exe:
                if attempt == 2:
                    HAVC_LogMessage(MessageType.CRITICAL,
                                    f"CMNET2 Client(): colorize_frame() failed [{type(exe).__name__}]: {exe}")
                import time
                time.sleep(base_delay * (attempt + 1))

    def colorize_frame_with_retry(self, ti: int = None, frame_i: Image = None,
                                  retry_blend_weight: float = 0.85,
                                  merge_engine_weight: float = 0.40,
                                  render_factor: int = 24) -> Image:
        """
        Single-call colorize + auto-retry. Server-side equivalent of:

            img_color = self.colorize_frame(ti, frame_i)
            if self.reference_frame_missing():
                img_ref = havc_engine.colorize_merged(frame_i)
                img_merged = image_weighted_merge(img_color, img_ref, retry_blend_weight)
                self.set_ref_frame(img_merged, frame_propagate=False)
                img_color = self.colorize_frame(ti, frame_i)
            return img_color

        but executed entirely on the server side, in a single RPC call. This
        avoids 3 extra round-trips per retry and keeps the merged-ref
        computation co-located with CMNET2 in the same process.

        Parameters mirror ColorizeImageWithRetry on the server side; defaults
        are tuned empirically for HAVC retry workflow.
        """
        if frame_i is not None:
            img_bytes_i = image_to_byte_array(frame_i)
            frame_bytes = self._safe_remote_call(self.server.ColorizeImageWithRetry,
                img_bytes_i, ti, retry_blend_weight, merge_engine_weight, render_factor)
            result = byte_array_to_image(frame_bytes)
            self._drain_server_logs()
            return result
        else:
            return None

    def set_ref_frame(self, frame_ref: Image = None, frame_propagate: bool = False):
        if frame_ref is None:
            self._safe_remote_call(self.server.SetRefImageNone, frame_propagate)
        else:
            frame_bytes = image_to_byte_array(frame_ref)
            self._safe_remote_call(self.server.SetRefImage, frame_bytes, frame_propagate)

    def colorize_frame(self, ti: int = None, frame_i: Image = None) -> Image:
        if frame_i is not None:
            img_bytes_i = image_to_byte_array(frame_i)
            frame_bytes = self._safe_remote_call(self.server.ColorizeImage, img_bytes_i, ti)
            result = byte_array_to_image(frame_bytes)
            self._drain_server_logs()
            return result
        else:
            return None

    def preload_reference(self, ref_img: Image):
        frame_bytes = image_to_byte_array(ref_img)
        self._safe_remote_call(self.server.PreloadReference, frame_bytes)

    def slide_permanent_memory(self, n_frames: int):
        self._safe_remote_call(self.server.SlidePermanentMemory, n_frames)

    def get_perm_mem_frame_count(self) -> int:
        return self.server.GetPermMemFrameCount()

    def get_last_match_metrics(self) -> tuple:
        """
        Returns the (mmsp, perm_share) tuple from the most recent colorize_frame
        call on the server. Mirrors ColorMNetRender2.get_last_match_metrics().

        XMLRPC serializes NaN as None on the wire; this method converts None
        back to float('nan') so the tuple is always (float, float).

        Returns:
            tuple[float, float] : (mmsp, perm_share). Both NaN if the server
                                   has not yet run a match_memory call.
        """
        try:
            payload = self.server.GetLastMatchMetrics()
        except Exception:
            # Network glitch or server not initialized: cannot decide.
            return float('nan'), float('nan')
        if not payload or len(payload) < 2:
            return float('nan'), float('nan')
        mmsp = float('nan') if payload[0] is None else float(payload[0])
        perm_share = float('nan') if payload[1] is None else float(payload[1])
        return mmsp, perm_share

    def reference_frame_missing(self) -> bool:
        """
        Returns True when the most recent colorize_frame call on the server
        indicates that an additional reference frame is likely needed for
        proper colorization. Thresholds are the ones configured at __init__.

        Stateless: each call evaluates the latest metrics independently.
        Returns False if the server is not reachable.
        """
        try:
            return bool(self.server.ReferenceFrameMissing())
        except Exception:
            # Network glitch or server not initialized: be conservative.
            return False

    def _drain_server_logs(self):
        """Pull log messages from the server and forward them to VS."""
        if self.server is None:
            return
        try:
            messages = self.server.PollLogMessages()
        except Exception:
            # Network glitch or server not yet ready: skip silently,
            # logs are best-effort and must never break inference.
            return
        for item in messages:
            if not item or len(item) < 2:
                continue
            level, text = item[0], item[1]
            try:
                mt = MessageType(int(level))
            except ValueError:
                mt = MessageType.INFORMATION
            # Never escalate server-side messages to EXCEPTION here: we don't
            # want a buffered log to raise vs.Error during frame processing.
            if mt == MessageType.EXCEPTION:
                mt = MessageType.CRITICAL
            # Server-side diagnostic logs are DEBUG/INFORMATION by origin.
            # Many VS host applications filter those levels out of their console,
            # so we promote them to WARNING to make them visible.
            if mt in (MessageType.DEBUG, MessageType.INFORMATION):
                mt = MessageType.WARNING
            HAVC_LogMessage(mt, text)

