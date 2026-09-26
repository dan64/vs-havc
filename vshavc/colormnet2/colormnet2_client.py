"""
-------------------------------------------------------------------------------
Author: Dan64
Date: 2025-09-28
version:
LastEditors: Dan64
LastEditTime: 2026-05-13
-------------------------------------------------------------------------------
Description:
-------------------------------------------------------------------------------
CMNET2 frame client class for Vapoursynth.
"""
import os
import math
import uuid
import numpy as np
from multiprocessing.shared_memory import SharedMemory
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
                 retry_perm_share_threshold: float = 0.30, retry_model: int = 0, server_port: int = None,
                 backbone: str = "dinov3"):
        if server_port is None:
            HAVC_LogMessage(MessageType.CRITICAL, "CMNET2 Client(): server port is None")
            return
        server_address = '127.0.0.1'

        # Handle graph restart (e.g. VSEdit loop): if the server was recreated
        # on a new port, reconnect to it instead of reusing the stale connection.
        if self._initialized:
            if server_port != self.server_port:
                HAVC_LogMessage(MessageType.WARNING,
                                f"CMNET2 Client(): change port from {self.server_port} to {server_port}")
                self.server_port = server_port
                self.uri = f"http://{server_address}:{server_port}"
                self.server = xmlrpc.client.ServerProxy(uri=self.uri, allow_none=True, use_builtin_types=True)
                # Reinitialize the server-side render
                self.server.initialize(image_size, vid_length, enable_resize, encode_mode, propagate,
                                       max_memory_frames, reset_on_ref_update, retry_mmsp_threshold,
                                       retry_perm_share_threshold, retry_model, backbone)
            return

        if not self._initialized:
            self.server_address = server_address
            self.server_port = server_port
            # Connect to a RPC instance; all the methods of the instance are
            # published as XML-RPC methods.
            self.uri = f"http://{server_address}:{server_port}"
            try:
                self.server = xmlrpc.client.ServerProxy(uri=self.uri, allow_none=True, use_builtin_types=True)
                self.server.initialize(image_size, vid_length, enable_resize, encode_mode, propagate,
                                       max_memory_frames, reset_on_ref_update, retry_mmsp_threshold,
                                       retry_perm_share_threshold, retry_model, backbone)
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

    def _shm_write(self, img: Image.Image):
        """
        Allocate a SharedMemory segment, write the PIL Image pixels into it,
        and return (shm, height, width).  The caller is responsible for
        calling shm.close() and shm.unlink() once the RPC call completes.
        """
        arr  = np.array(img)
        h, w = arr.shape[:2]
        name = f"cmnet2_{uuid.uuid4().hex[:12]}"
        shm  = SharedMemory(name=name, create=True, size=h * w * 3)
        np.ndarray((h, w, 3), dtype=np.uint8, buffer=shm.buf)[:] = arr
        return shm, h, w

    def _shm_read(self, shm: SharedMemory, h: int, w: int) -> Image.Image:
        """Read a PIL Image from a SharedMemory segment (must still be open)."""
        arr = np.ndarray((h, w, 3), dtype=np.uint8, buffer=shm.buf)
        return Image.fromarray(arr.copy(), mode="RGB")

    def set_ref_frame(self, frame_ref: Image = None, frame_propagate: bool = False):
        if frame_ref is None:
            self._safe_remote_call(self.server.SetRefImageNone, frame_propagate)
            return
        shm, h, w = self._shm_write(frame_ref)
        try:
            self._safe_remote_call(
                self.server.SetRefImageShm, shm.name, h, w, frame_propagate)
        finally:
            shm.close(); shm.unlink()

    def colorize_frame(self, ti: int = None, frame_i: Image = None) -> Image:
        if frame_i is None:
            return None
        shm_in, h, w = self._shm_write(frame_i)
        shm_out = SharedMemory(
            name=f"cmnet2_out_{uuid.uuid4().hex[:12]}", create=True, size=h * w * 3)
        try:
            self._safe_remote_call(
                self.server.ColorizeImageShm, shm_in.name, shm_out.name, h, w, ti)
            result = self._shm_read(shm_out, h, w)
            self._drain_server_logs()
            return result
        finally:
            shm_in.close();  shm_in.unlink()
            shm_out.close(); shm_out.unlink()

    def colorize_frame_with_retry(self, ti: int = None, frame_i: Image = None,
                                  retry_blend_weight: float = 0.85,
                                  merge_engine_weight: float = 0.40,
                                  render_factor: int = 24,
                                   retry_model: int = 0) -> Image:
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
        if frame_i is None:
            return None
        shm_in, h, w = self._shm_write(frame_i)
        shm_out = SharedMemory(
            name=f"cmnet2_out_{uuid.uuid4().hex[:12]}", create=True, size=h * w * 3)
        try:
            self._safe_remote_call(
                self.server.ColorizeImageWithRetryShm,
                shm_in.name, shm_out.name, h, w,
                ti, retry_blend_weight, merge_engine_weight, render_factor)
            result = self._shm_read(shm_out, h, w)
            self._drain_server_logs()
            return result
        finally:
            shm_in.close();  shm_in.unlink()
            shm_out.close(); shm_out.unlink()

    def preload_reference(self, ref_img: Image, frame_idx: int = None):
        shm, h, w = self._shm_write(ref_img)
        try:
            self._safe_remote_call(
                self.server.PreloadReferenceShm, shm.name, h, w, frame_idx)
        finally:
            shm.close(); shm.unlink()

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
        """
        mmsp, perm_share = self.server.GetLastMatchMetrics()
        mmsp       = float('nan') if mmsp       is None else float(mmsp)
        perm_share = float('nan') if perm_share is None else float(perm_share)
        return mmsp, perm_share

    def _drain_server_logs(self):
        """Pull log messages from the server and forward them to VS."""
        if self.server is None:
            return
        try:
            messages = self.server.PollLogMessages()
        except Exception:
            return
        for item in messages:
            if not item or len(item) < 2:
                continue
            level, text = item[0], item[1]
            try:
                mt = MessageType(int(level))
            except ValueError:
                mt = MessageType.INFORMATION
            if mt == MessageType.EXCEPTION:
                mt = MessageType.CRITICAL
            if mt in (MessageType.DEBUG, MessageType.INFORMATION):
                mt = MessageType.WARNING
            HAVC_LogMessage(mt, text)
