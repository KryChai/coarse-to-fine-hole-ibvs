"""Autofocus on the capture-owner thread; sharpness is a heuristic, not lens feedback."""
import time
import cv2
import numpy as np


def sharpness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def focus_capture(capture, settings, *, on_frame=None, cancelled=lambda: False):
    """Do not call concurrently with another read/set on this capture.

    Continuous mode leaves AF enabled; legacy node mode inspects post-lock frames. OpenCV
    property acknowledgements cannot establish that hardware accepted a setting.
    Blocking driver calls cannot be interrupted by this software deadline.
    """
    if settings.get("continuous", False):
        return _continuous_focus(capture, settings, on_frame=on_frame, cancelled=cancelled)
    started = time.monotonic()
    timeout = float(settings["timeout_s"])
    wait = float(settings["min_wait_s"])
    count = int(settings["stable_frames"])
    ratio = float(settings["stable_ratio"])
    if not 0 < wait < timeout or count < 2 or not 0 < ratio < 1:
        raise ValueError("invalid autofocus settings")
    result = dict(status="failed", score=None, baseline=None, peak=None,
                  enable_ack=False, lock_ack=False, reason="", elapsed_s=0.)
    locked = False
    previous = None
    stable = 0
    peak = baseline = None

    def read_score():
        if cancelled():
            raise RuntimeError("对焦已取消")
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError("对焦期间无法读取新帧")
        score = sharpness(frame)
        if not np.isfinite(score) or score <= 0:
            raise RuntimeError("清晰度无效或画面没有可判定纹理")
        if on_frame is not None:
            on_frame(frame)
        return score

    try:
        baseline = peak = read_score()
        result["baseline"] = baseline
        capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        result["enable_ack"] = bool(capture.set(cv2.CAP_PROP_AUTOFOCUS, 1))
        while time.monotonic() - started < timeout:
            score = read_score()
            peak = max(peak, score)
            near_peak = score >= peak * (1 - ratio)
            steady = previous is not None and abs(score - previous) <= peak * ratio
            stable = stable + 1 if near_peak and steady else 0
            previous = score
            if stable >= count and time.monotonic() - started >= wait:
                break
            time.sleep(.02)
        else:
            raise RuntimeError("对焦等待超时或清晰度未恢复至峰值附近")
        result["lock_ack"] = bool(capture.set(cv2.CAP_PROP_AUTOFOCUS, 0))
        locked = True
        if result["enable_ack"] and not result["lock_ack"]:
            raise RuntimeError("自动对焦已开启，但关闭/锁焦请求未被后端接受")
        # Discard transition frames, then verify sharpness remains stable with AF off.
        scores = []
        for index in range(count + 2):
            if time.monotonic() - started >= timeout:
                raise RuntimeError("锁焦后验证超时")
            score = read_score()
            if index >= 2:
                scores.append(score)
            time.sleep(.02)
        if min(scores) < peak * (1 - ratio) or max(scores)-min(scores) > peak * ratio:
            raise RuntimeError("关闭自动对焦后清晰度下降或不稳定，不能确认焦点保持")
        result["score"] = float(np.median(scores))
        # A stable image is usable, but never claim unsupported hardware refocused.
        result["status"] = "stable" if result["enable_ack"] else "unsupported"
    except (AttributeError, TypeError, ValueError, RuntimeError, cv2.error) as error:
        result["reason"] = str(error)
    finally:
        if not locked:
            try:
                result["lock_ack"] = bool(capture.set(cv2.CAP_PROP_AUTOFOCUS, 0))
            except (AttributeError, TypeError, ValueError, RuntimeError, cv2.error) as error:
                result["reason"] += "; 锁焦失败: " + str(error)
        result["peak"] = peak
        result["elapsed_s"] = time.monotonic() - started
    return result


def _continuous_focus(capture, settings, *, on_frame=None, cancelled=lambda: False):
    """Leave AF enabled on success, timeout and cancellation; never send AF=0."""
    from collections import deque
    started = time.monotonic()
    timeout, minimum = float(settings["timeout_s"]), float(settings["min_wait_s"])
    count, ratio = int(settings["stable_frames"]), float(settings["stable_ratio"])
    if not 0 < minimum < timeout or count < 2 or not 0 < ratio < 1:
        raise ValueError("invalid continuous autofocus settings")
    result = dict(mode="continuous", status="failed", score=None, baseline=None,
                  peak=None, enable_ack=False, lock_ack=None, reason="", elapsed_s=0., samples=0)
    window = deque(maxlen=count)
    peak = 0.
    try:
        # The owner supplies the acknowledgement from startup, so key nodes do
        # not toggle the property or restart a sweep while taking a photo.
        ack = settings.get("enabled_ack")
        result["enable_ack"] = bool(capture.set(cv2.CAP_PROP_AUTOFOCUS, 1)) if ack is None else bool(ack)
        while time.monotonic() - started < timeout:
            if cancelled():
                raise RuntimeError("连续自动对焦等待已取消")
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError("连续自动对焦期间读取新帧失败")
            score = sharpness(frame)
            if not np.isfinite(score) or score <= 0:
                raise RuntimeError("清晰度无效或画面没有可判定纹理")
            if on_frame is not None:
                on_frame(frame)
            if result["baseline"] is None:
                result["baseline"] = score
                peak = score
            result["samples"] += 1
            result["score"] = score
            window.append(score)
            if len(window) == count:
                median = float(np.median(window))
                peak = max(peak, median)
                steady = max(window)-min(window) <= ratio*peak
                clear = min(window) >= peak*(1-ratio)
                if steady and clear and time.monotonic()-started >= minimum:
                    result["score"] = median
                    result["status"] = "stable" if result["enable_ack"] else "unsupported"
                    break
            time.sleep(.02)
        else:
            result["reason"] = "连续自动对焦画面在期限内未稳定，保持 AF 开启并终止本轮"
    except (AttributeError, TypeError, ValueError, RuntimeError, cv2.error) as error:
        result["reason"] = str(error)
    finally:
        result["peak"] = peak
        result["elapsed_s"] = time.monotonic()-started
    return result
