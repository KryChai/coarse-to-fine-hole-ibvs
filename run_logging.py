"""将 Python 标准输出和错误输出同时写入终端及每次运行的 UTF-8 日志。"""

import atexit
from datetime import datetime
import os
from pathlib import Path
import sys
import threading


_active_log_path = None
_active_log_writer = None


def write_run_event(message):
    """Mirror audit data to the active terminal log without flooding the console."""
    if _active_log_writer is not None:
        _active_log_writer(message)


class _TeeStream:
    def __init__(self, console, log_file, lock):
        self._console = console
        self._log_file = log_file
        self._lock = lock

    def write(self, text):
        # stdout/stderr 共用锁，避免多个线程同时写坏文件。
        with self._lock:
            if not self._log_file.closed:
                self._log_file.write(text)
                self._log_file.flush()
            if self._console is not None:
                self._console.write(text)
        return len(text)

    def flush(self):
        with self._lock:
            if not self._log_file.closed:
                self._log_file.flush()
            if self._console is not None:
                self._console.flush()

    def __getattr__(self, name):
        # 保留终端的 encoding、isatty、fileno 等接口，兼容现有库。
        return getattr(self._console, name)


def enable_run_logging(script_path):
    """在脚本入口尽早调用；导入的模块和后台线程共享同一份日志。"""
    global _active_log_path, _active_log_writer
    if _active_log_path is not None:
        return _active_log_path

    script_path = Path(script_path).resolve()
    log_dir = script_path.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now()
    log_path = log_dir / (
        f"{script_path.stem}_{started_at:%Y%m%d_%H%M%S_%f}_{os.getpid()}.log"
    )
    log_file = log_path.open("x", encoding="utf-8")
    lock = threading.RLock()
    def write_event(message):
        with lock:
            if not log_file.closed:
                log_file.write(message + "\n")
                log_file.flush()
    _active_log_writer = write_event
    original_stdout, original_stderr = sys.stdout, sys.stderr
    stdout = _TeeStream(original_stdout, log_file, lock)
    stderr = _TeeStream(original_stderr, log_file, lock)
    sys.stdout, sys.stderr = stdout, stderr
    _active_log_path = log_path

    def close_log():
        # 未捕获异常的 traceback 在 atexit 之前输出，因此也会保留下来。
        with lock:
            try:
                stdout.flush()
                stderr.flush()
            finally:
                if sys.stdout is stdout:
                    sys.stdout = original_stdout
                if sys.stderr is stderr:
                    sys.stderr = original_stderr
                log_file.close()

    atexit.register(close_log)
    print(f"[日志] 运行日志文件: {log_path}", flush=True)
    print(f"[日志] 开始时间: {started_at:%Y-%m-%d %H:%M:%S}", flush=True)
    return log_path
