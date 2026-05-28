"""
-------------------------------------------------------------------------------
Author: Dan64
-------------------------------------------------------------------------------
Description:
-------------------------------------------------------------------------------
Thread-safe log buffer used by the CMNET2 RPC server to collect messages
that will be polled by the client and forwarded to the VapourSynth log.

Levels follow vsdeoldify.vsslib.vsutils.MessageType integer values:
  0=DEBUG, 1=INFORMATION, 2=WARNING, 3=CRITICAL, 4=FATAL
"""
from collections import deque
from threading import Lock
import sys

class ServerLogBuffer:
    """Single-instance (per process) append-only log queue."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._lock = Lock()
            # Cap prevents unbounded growth if the client stops polling.
            cls._instance._queue = deque(maxlen=10000)
        return cls._instance

    def log(self, level: int, message: str) -> None:
        with self._lock:
            self._queue.append((int(level), str(message)))

    def drain(self) -> list:
        """Return all pending messages and empty the buffer."""
        with self._lock:
            items = list(self._queue)
            self._queue.clear()
        return items


# Convenience module-level helpers so callers inside the server don't need
# to reach for the singleton every time.
_buf = ServerLogBuffer()

def log_debug(msg: str):       _buf.log(0, msg)
def log_info(msg: str):        _buf.log(1, msg)
def log_warning(msg: str):     _buf.log(2, msg)
def log_critical(msg: str):    _buf.log(3, msg)

