"""通过独立 Python 进程验证日志留存，不连接机器人或相机。"""

from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RunLoggingTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.script_path = self.directory / "task.py"

    def run_python(self, code):
        return subprocess.run(
            [sys.executable, "-X", "utf8", "-c", textwrap.dedent(code)],
            cwd=ROOT,
            capture_output=True,
            encoding="utf-8",
            timeout=15,
        )

    def run_logged(self, body):
        return self.run_python(
            "from run_logging import enable_run_logging\n"
            f"log_path = enable_run_logging({str(self.script_path)!r})\n"
            + textwrap.dedent(body)
        )

    def read_log(self, stem="task"):
        paths = list((self.directory / "logs").glob(f"{stem}_*.log"))
        self.assertEqual(len(paths), 1)
        return paths[0].read_text(encoding="utf-8")

    def test_console_output_errors_threads_and_cleanup_are_saved(self):
        result = self.run_logged('''
            import atexit
            import sys
            import threading
            import warnings
            print("✅ 图片已保存: example/photo/IMG.jpg")
            print("[IBVS 40] u=1284.1, v=831.2, conf=0.942, error=111.28 px")
            print("错误输出", file=sys.stderr)
            warnings.warn("检测警告")
            worker = threading.Thread(target=lambda: print("[USB相机] 后台线程输出"))
            worker.start()
            worker.join()
            # 不等待进程退出，确认已实时写入，包括没有换行的输出。
            print("伺服微调", end="")
            assert "伺服微调" in log_path.read_text(encoding="utf-8")
            atexit.register(lambda: print("资源清理完成"))
        ''')

        self.assertEqual(result.returncode, 0, result.stderr)
        log = self.read_log()
        self.assertIn("✅ 图片已保存", result.stdout)
        self.assertIn("错误输出", result.stderr)
        # 每个终端流中的所有内容，按原有顺序和换行保存在同一日志中。
        for output in (result.stdout, result.stderr):
            offset = 0
            for line in output.splitlines(keepends=True):
                position = log.find(line, offset)
                self.assertGreaterEqual(position, offset, repr(line))
                offset = position + len(line)
        self.assertIn("[USB相机] 后台线程输出", log)
        self.assertIn("资源清理完成", log)

    def test_uncaught_exception_and_ctrl_c_tracebacks_are_saved(self):
        for exception in ("RuntimeError('运行异常')", "KeyboardInterrupt()"):
            with self.subTest(exception=exception):
                result = self.run_logged(f"print('退出前输出')\nraise {exception}\n")
                self.assertNotEqual(result.returncode, 0)
                paths = list((self.directory / "logs").glob("task_*.log"))
                log = max(paths, key=lambda path: path.name).read_text(encoding="utf-8")
                self.assertIn("退出前输出", log)
                self.assertIn(result.stderr, log)
                self.assertIn("Traceback (most recent call last)", log)

    def test_repeated_setup_is_idempotent_and_runs_have_separate_files(self):
        body = f'''
            assert enable_run_logging({str(self.script_path)!r}) == log_path
            print("只输出一次")
        '''
        for _ in range(2):
            result = self.run_logged(body)
            self.assertEqual(result.returncode, 0, result.stderr)
        paths = list((self.directory / "logs").glob("task_*.log"))
        self.assertEqual(len(paths), 2)
        for path in paths:
            log = path.read_text(encoding="utf-8")
            self.assertEqual(log.count("[日志] 运行日志文件:"), 1)
            self.assertEqual(log.count("只输出一次"), 1)


if __name__ == "__main__":
    unittest.main()
