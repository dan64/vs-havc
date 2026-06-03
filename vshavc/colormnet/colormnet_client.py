"""
-------------------------------------------------------------------------------
Author: Dan64
Date: 2024-09-27
version:
LastEditors: Dan64
LastEditTime: 2026-05-21
-------------------------------------------------------------------------------
Description:
-------------------------------------------------------------------------------
ColorMNet frame client class for Vapoursynth.
"""
import os
from PIL import Image
import warnings
import xmlrpc.client
from vshavc.colormnet.colormnet_utils import *
from vshavc.vsslib.vsutils import MessageType, HAVC_LogMessage

class ColorMNetClient:
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
                 reset_on_ref_update: bool = True, server_port: int = None):
        if server_port is None:
            HAVC_LogMessage(MessageType.CRITICAL, "ColorMNet client(): server port is None")
            return
        server_address = '127.0.0.1'

        # Handle graph restart (e.g. VSEdit loop): if the server was recreated
        # on a new port, reconnect to it instead of reusing the stale connection.
        if self._initialized:
            if server_port != self.server_port:
                HAVC_LogMessage(MessageType.WARNING,
                                f"ColorMNet Client(): change port from {self.server_port} to {server_port}")
                self.server_port = server_port
                self.uri = f"http://{server_address}:{server_port}"
                self.server = xmlrpc.client.ServerProxy(uri=self.uri, allow_none=True, use_builtin_types=True)
                # Reinitialize the server-side render
                self.server.initialize(image_size, vid_length, enable_resize, encode_mode, propagate,
                                       max_memory_frames, reset_on_ref_update)
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
                                       max_memory_frames, reset_on_ref_update)
                self._initialized = True
            except Exception as exe:
                HAVC_LogMessage(MessageType.CRITICAL,
                                f"ColorMNet client(): init failed [{type(exe).__name__}]: {exe}")

    def is_initialized(self) -> bool:
        return self.server.IsInitialized()

    def get_frame_count(self) -> int:
        return self.server.GetFrameCount()

    def set_ref_frame(self, frame_ref: Image = None, frame_propagate: bool = False):
        if frame_ref is None:
            self.server.SetRefImageNone(frame_propagate)
        else:
            frame_bytes = image_to_byte_array(frame_ref)
            self.server.SetRefImage(frame_bytes, frame_propagate)

    def colorize_frame(self, ti: int = None, frame_i: Image = None) -> Image:
        if frame_i is not None:
            img_bytes_i = image_to_byte_array(frame_i)
            frame_bytes = self.server.ColorizeImage(img_bytes_i, ti)
            result = byte_array_to_image(frame_bytes)
            self._drain_server_logs()
            return result
        else:
            return None

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
            if mt in (MessageType.DEBUG, MessageType.INFORMATION):
                mt = MessageType.WARNING
            HAVC_LogMessage(mt, text)
