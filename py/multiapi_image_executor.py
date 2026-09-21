import base64
import io
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Tuple
from uuid import uuid4

# import tuple
import numpy as np
import requests
import torch
from PIL import Image

from .util.key_resolver import get_key, get_key_name

TASK_TYPE = "MULTIAPI_IMAGE_TASK"
CATEGORY = "MultiAPI Image Executor"


#############################################################################
## GenTask Start


class GeminiImageGenTask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "image1": ("IMAGE",),
                "model": (["gemini-2.5-flash-image"],),
                "aspect_ratio": (
                    [
                        "1:1",
                        "2:3",
                        "3:2",
                        "4:3",
                        "3:4",
                        "16:9",
                        "9:16",
                        "21:9",
                        "1:4",
                        "4:1",
                        "1:8",
                        "8:1",
                        "4:5",
                        "5:4",
                    ],
                ),
                "image_size": (["1K", "2K", "4K"],),
                "sequential_image_generation": (["auto", "enabled", "disabled"],),
                "max_images": ("INT", {"default": 1, "min": 1, "max": 10}),
                "response_format": (["url", "b64_json"],),
                "watermark": ("BOOLEAN", {"default": False}),
                "stream": ("BOOLEAN", {"default": False}),
                "base_url": ("STRING", {"default": "https://api.modelverse.cn"}),
                "use_local_images": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 18446744073709551615}),
                "enable_auto_retry": ("BOOLEAN", {"default": True}),
                "timeout": ("INT", {"default": 70, "min": 10, "max": 300}),
            },
            "optional": {
                "key": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "环境变量名称（留空使用默认名称）",
                    },
                ),
                **_optional_images(),
            },
        }

    RETURN_TYPES = (TASK_TYPE,)
    RETURN_NAMES = ("task",)
    FUNCTION = "submit"
    CATEGORY = CATEGORY

    def submit(self, prompt, image1, **kwargs):
        return _task("gemini", prompt, kwargs, {"image1": image1, **kwargs})


class SeedreamImageGenTask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "image1": ("IMAGE",),
                "model": (
                    [
                        "doubao-seedream-4-0-250828",
                        "doubao-seedream-4-5-251128",
                        "doubao-seedream-5-0-lite-260128",
                        "doubao-seedream-5-0-pro-260628",
                    ],
                ),
                "aspect_ratio": (
                    [
                        "1:1",
                        "4:3",
                        "3:4",
                        "16:9",
                        "9:16",
                        "3:2",
                        "2:3",
                        "21:9",
                    ],
                ),
                "resolution": (["1K", "1.5K", "2K", "3K", "4K"], {"default": "2K"}),
                "sequential_image_generation": (["auto", "enabled", "disabled"],),
                "max_images": ("INT", {"default": 1, "min": 1, "max": 10}),
                "response_format": (["url", "b64_json"],),
                "watermark": ("BOOLEAN", {"default": False}),
                "stream": ("BOOLEAN", {"default": False}),
                "base_url": ("STRING", {"default": "https://ark.cn-beijing.volces.com/api/v3"}),
                "use_local_images": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 18446744073709551615}),
                "enable_auto_retry": ("BOOLEAN", {"default": True}),
                "timeout": ("INT", {"default": 70, "min": 10, "max": 300}),
                "optimize_prompt_options": (["fast", "standard"],),
            },
            "optional": {
                "key": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "环境变量名称（留空使用 ARK_API_KEY）",
                    },
                ),
                "size": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "手动像素尺寸，例如 2048x2048（留空使用预设）",
                    },
                ),
                **_optional_images(),
            },
        }

    RETURN_TYPES = (TASK_TYPE,)
    RETURN_NAMES = ("task",)
    FUNCTION = "submit"
    CATEGORY = CATEGORY

    def submit(self, prompt, image1, **kwargs):
        return _task("seedream", prompt, kwargs, {"image1": image1, **kwargs})


class GPTImageGenTask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "image1": ("IMAGE",),
                "model": (["gpt-image-2"],),
                "aspect_ratio": (
                    ["auto", "1:1", "2:3", "3:2", "4:3", "3:4", "16:9", "9:16", "21:9", "2K", "4K"],
                ),
                "quality": (["low", "medium", "high"], {"default": "high"}),
                "max_images": ("INT", {"default": 1, "min": 1, "max": 10}),
                "output_format": (["png", "jpeg"],),
                "output_compression": ("INT", {"default": 100, "min": 0, "max": 100}),
                "base_url": ("STRING", {"default": "https://api.openai.com/v1"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 18446744073709551615}),
                "enable_auto_retry": ("BOOLEAN", {"default": True}),
                "timeout": ("INT", {"default": 70, "min": 10, "max": 300}),
            },
            "optional": {
                "key": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "环境变量名称（留空使用 OPENAI_API_KEY）",
                    },
                ),
                **_optional_images(),
            },
        }

    RETURN_TYPES = (TASK_TYPE,)
    RETURN_NAMES = ("task",)
    FUNCTION = "submit"
    CATEGORY = CATEGORY

    def submit(self, prompt, image1, **kwargs):
        return _task("gpt_image", prompt, kwargs, {"image1": image1, **kwargs})


class QwenImageGenTask:
    """Create a Qwen image-editing task for the MultiAPI executor."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "image1": ("IMAGE",),
                "model": (
                    [
                        "qwen-image-3.0-pro",
                        "qwen-image-3.0",
                        "qwen-image-2.0-pro",
                        "qwen-image-2.0",
                        "qwen-image-edit-max",
                        "qwen-image-edit-plus",
                        "qwen-image-edit",
                    ],
                    {"default": "qwen-image-3.0-pro"},
                ),
                "size": (
                    [
                        "auto",
                        "1024*1024",
                        "768*1152",
                        "1024*1536",
                        "1152*768",
                        "1536*1024",
                        "720*1280",
                        "1080*1920",
                        "1280*720",
                        "1920*1080",
                    ],
                    {"default": "1024*1024"},
                ),
                "max_images": ("INT", {"default": 1, "min": 1, "max": 6}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "prompt_extend": ("BOOLEAN", {"default": True}),
                "watermark": ("BOOLEAN", {"default": False}),
                "enable_thinking": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "base_url": (
                    "STRING",
                    {
                        "default": (
                            "https://dashscope.aliyuncs.com/api/v1/services/aigc/"
                            "multimodal-generation/generation"
                        ),
                    },
                ),
                "timeout": ("INT", {"default": 120, "min": 10, "max": 300}),
            },
            "optional": {
                "key": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "DASHSCOPE_API_KEY",
                    },
                ),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
            },
        }

    RETURN_TYPES = (TASK_TYPE,)
    RETURN_NAMES = ("task",)
    FUNCTION = "submit"
    CATEGORY = CATEGORY

    def submit(self, prompt, image1, **kwargs):
        return _task("qwen", prompt, kwargs, {"image1": image1, **kwargs})


## GenTask End
#############################################################################


@dataclass(frozen=True)
class ImageGenerationTask:
    provider: str
    prompt: str
    images: Tuple[torch.Tensor, ...]
    params: Dict[str, Any]


class DirectiveFailureError(RuntimeError):
    DIRECTIVES = {
        "[success]": "success",
        "[failure_timeout]": "failure_timeout",
        "[failure_safety]": "failure_safety",
        "[failure_network]": "failure_network",
        "[failure_other]": "failure_other",
    }

    MESSAGES = {
        "failure_timeout": "测试指令返回超时错误",
        "failure_safety": "测试指令返回安全审核错误",
        "failure_network": "测试指令返回网络错误",
        "failure_other": "测试指令返回其他错误",
    }

    def __init__(self, status):
        self.status = status
        super().__init__(self.MESSAGES[status])


def _images(kwargs):
    result = []
    for index in range(1, 7):
        image = kwargs.get(f"image{index}")
        if image is None:
            continue
        shape = getattr(image, "shape", ())
        if len(shape) < 3:
            continue
        height, width = int(shape[-3]), int(shape[-2])
        if height >= 14 and width >= 14:
            result.append(image)
    return tuple(result)


def _task(provider, prompt, params, kwargs):
    prompt = str(prompt or "").strip()
    images = _images(kwargs)
    if prompt and prompt.lower() not in DirectiveFailureError.DIRECTIVES and not images:
        raise ValueError("至少需要一张宽和高均不小于 14px 的参考图")
    clean_params = {
        key: value for key, value in params.items() if not re.fullmatch(r"image\d+", key)
    }
    return (ImageGenerationTask(provider, prompt, images, clean_params),)


def _optional_images():
    return {f"image{i}": ("IMAGE",) for i in range(2, 7)}


def _pil(tensor):
    sample = tensor[0] if tensor.ndim == 4 else tensor
    array = np.clip(sample.detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


def _tensor(image):
    return torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0).unsqueeze(0)


def _png_data_url(tensor):
    buffer = io.BytesIO()
    _pil(tensor).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _decode_image(data):
    if isinstance(data, str) and data.startswith("data:"):
        data = data.split(",", 1)[1]
    return _tensor(Image.open(io.BytesIO(base64.b64decode(data))))


def _download(url, timeout):
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return _tensor(Image.open(io.BytesIO(response.content)))


def _size(ratio):
    return {
        "auto": "auto",
        "1:1": "1024x1024",
        "2:3": "1024x1536",
        "3:2": "1536x1024",
        "4:3": "1536x1024",
        "3:4": "1024x1536",
        "16:9": "1536x1024",
        "9:16": "1024x1536",
        "21:9": "1536x1024",
        "2K": "2048x2048",
        "3K": "2133x3200",
        "3.5K": "2933x4400",
        "4K": "4096x4096",
    }.get(ratio, ratio)


_SEEDREAM_STANDARD_SIZES = {
    "1K": {
        "1:1": "1024x1024",
        "4:3": "1152x864",
        "3:4": "864x1152",
        "16:9": "1424x800",
        "9:16": "800x1424",
        "3:2": "1248x832",
        "2:3": "832x1248",
        "21:9": "1568x672",
    },
    "1.5K": {
        "1:1": "1536x1536",
        "4:3": "1792x1344",
        "3:4": "1344x1792",
        "16:9": "2048x1152",
        "9:16": "1152x2048",
        "3:2": "1872x1248",
        "2:3": "1248x1872",
        "21:9": "2352x1008",
    },
    "2K": {
        "1:1": "2048x2048",
        "4:3": "2304x1728",
        "3:4": "1728x2304",
        "16:9": "2848x1600",
        "9:16": "1600x2848",
        "3:2": "2496x1664",
        "2:3": "1664x2496",
        "21:9": "3136x1344",
    },
    "3K": {
        "1:1": "3072x3072",
        "4:3": "3456x2592",
        "3:4": "2592x3456",
        "16:9": "4096x2304",
        "9:16": "2304x4096",
        "3:2": "3744x2496",
        "2:3": "2496x3744",
        "21:9": "4704x2016",
    },
    "4K": {
        "1:1": "4096x4096",
        "4:3": "4704x3520",
        "3:4": "3520x4704",
        "16:9": "5504x3040",
        "9:16": "3040x5504",
        "3:2": "4992x3328",
        "2:3": "3328x4992",
        "21:9": "6240x2656",
    },
}

_SEEDREAM_PRO_2K_SIZES = {
    "1:1": "2048x2048",
    "4:3": "2368x1776",
    "3:4": "1776x2368",
    "16:9": "2816x1584",
    "9:16": "1584x2816",
    "3:2": "2496x1664",
    "2:3": "1664x2496",
    "21:9": "3136x1344",
}


def _seedream_size(model, aspect_ratio, resolution):
    if model == "doubao-seedream-5-0-pro-260628" and resolution == "2K":
        sizes = _SEEDREAM_PRO_2K_SIZES
    else:
        sizes = _SEEDREAM_STANDARD_SIZES.get(resolution)

    if not sizes or aspect_ratio not in sizes:
        raise ValueError(
            f"不支持的 Seedream 尺寸组合: model={model}, "
            f"aspect_ratio={aspect_ratio}, resolution={resolution}"
        )

    return sizes[aspect_ratio]


def _manual_pixel_size(value):
    value = str(value or "").strip()
    if not value:
        return None

    match = re.fullmatch(r"(\d+)\s*[xX×]\s*(\d+)", value)
    if not match:
        raise ValueError("手动 size 格式无效，请使用 宽x高，例如 2048x2048")

    width, height = (int(part) for part in match.groups())
    if width <= 0 or height <= 0:
        raise ValueError("手动 size 的宽和高必须大于 0")

    return f"{width}x{height}"


_USER_KEY_FILE_NAME = "keys.json"


def _user_key_file_path():
    try:
        import folder_paths

        user_directory = folder_paths.get_user_directory()
    except (ImportError, AttributeError):
        plugin_directory = os.path.dirname(os.path.abspath(__file__))
        custom_nodes_directory = os.path.dirname(plugin_directory)
        user_directory = os.path.join(os.path.dirname(custom_nodes_directory), "user")
    return os.path.join(user_directory, _USER_KEY_FILE_NAME)


def _load_user_keys():
    """Read the live user key file, creating an empty one on first use."""
    key_path = _user_key_file_path()
    os.makedirs(os.path.dirname(key_path), exist_ok=True)
    lock_path = key_path + ".lock"
    last_error = None

    # Retrying also tolerates editors that briefly truncate the file before saving.
    for attempt in range(5):
        with open(lock_path, "a+b") as lock_file:
            _lock_file(lock_file)
            try:
                if not os.path.exists(key_path):
                    file_descriptor = os.open(
                        key_path,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                        0o600,
                    )
                    try:
                        os.write(file_descriptor, b"{}\n")
                        os.fsync(file_descriptor)
                    finally:
                        os.close(file_descriptor)

                try:
                    with open(key_path, encoding="utf-8-sig") as key_file:
                        values = json.load(key_file)
                except (OSError, json.JSONDecodeError) as error:
                    last_error = error
                    values = None
            finally:
                _unlock_file(lock_file)

        if values is not None:
            if not isinstance(values, dict):
                raise ValueError(f"Key 文件必须是 JSON 对象: {key_path}")
            return values
        if attempt < 4:
            time.sleep(0.05)

    raise ValueError(f"无法解析 Key 文件 {key_path}: {last_error}") from last_error


def _environment_key(custom_name, default_names):
    custom_name = str(custom_name or "").strip()
    names = (custom_name,) if custom_name else tuple(default_names)
    return get_key(*names)

    user_keys = _load_user_keys()
    for name in names:
        value = user_keys.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()

    for name in names:
        value = os.getenv(name)
        if value:
            return value.strip()

    if custom_name:
        raise ValueError(f"Key 文件和环境变量中均未设置指定名称: {custom_name}")
    raise ValueError("Key 文件和环境变量中均未设置: " + "、".join(names))


_PROVIDER_KEY_NAMES = {
    "gemini": ("MODELVERSE_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "seedream": ("ARK_API_KEY",),
    "gpt_image": ("OPENAI_API_KEY",),
    "qwen": ("DASHSCOPE_API_KEY",),
}


def _task_key_name(task):
    """Return the configured key name, never its secret value."""
    custom_name = str(task.params.get("key") or "").strip()
    if custom_name:
        return custom_name

    names = _PROVIDER_KEY_NAMES.get(task.provider, ())
    return get_key_name(*names) or (names[0] if names else "")

    try:
        user_keys = _load_user_keys()
    except Exception:
        user_keys = {}
    for name in names:
        value = user_keys.get(name)
        if isinstance(value, str) and value.strip():
            return name
    for name in names:
        if os.getenv(name):
            return name
    return names[0] if names else ""


def _task_log_size(task):
    p = task.params
    if task.provider == "gemini":
        return f"{p.get('image_size', '')} ({p.get('aspect_ratio', '')})".strip()
    if task.provider == "gpt_image":
        return str(_size(p.get("aspect_ratio", "")))
    if task.provider == "seedream":
        manual_size = str(p.get("size") or "").strip()
        if manual_size:
            return manual_size
        try:
            return _seedream_size(p.get("model"), p.get("aspect_ratio"), p.get("resolution"))
        except Exception:
            return f"{p.get('resolution', '')} ({p.get('aspect_ratio', '')})".strip()
    if task.provider == "qwen":
        return str(p.get("size") or "auto")
    return str(p.get("size") or "")


def _api_log_directory():
    try:
        import folder_paths

        output_directory = folder_paths.get_output_directory()
    except (ImportError, AttributeError):
        plugin_directory = os.path.dirname(os.path.abspath(__file__))
        custom_nodes_directory = os.path.dirname(plugin_directory)
        output_directory = os.path.join(os.path.dirname(custom_nodes_directory), "output")
    return os.path.join(output_directory, "apilog")


def _execution_uuid():
    """Prefer ComfyUI's queue prompt ID so backend history can be queried."""
    try:
        from comfy_execution.utils import get_executing_context

        context = get_executing_context()
        prompt_id = getattr(context, "prompt_id", None)
        if prompt_id:
            return str(prompt_id)
    except (ImportError, AttributeError):
        pass
    return str(uuid4())


def _lock_file(lock_file):
    if os.name == "nt":
        import msvcrt

        lock_file.seek(0, os.SEEK_END)
        if lock_file.tell() == 0:
            lock_file.write(b"\0")
            lock_file.flush()
        lock_file.seek(0)
        while True:
            try:
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                time.sleep(0.05)
    else:
        import fcntl

        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)


def _unlock_file(lock_file):
    if os.name == "nt":
        import msvcrt

        lock_file.seek(0)
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _append_api_log(records, log_time):
    """Append complete JSONL records while holding a cross-process file lock."""
    log_directory = _api_log_directory()
    os.makedirs(log_directory, exist_ok=True)
    date_text = log_time.strftime("%Y-%m-%d")
    log_path = os.path.join(log_directory, f"{date_text}.jsonl")
    lock_path = os.path.join(log_directory, f".{date_text}.lock")
    payload = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records
    ).encode("utf-8")

    with open(lock_path, "a+b") as lock_file:
        _lock_file(lock_file)
        try:
            file_descriptor = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                position = 0
                while position < len(payload):
                    position += os.write(file_descriptor, payload[position:])
                os.fsync(file_descriptor)
            finally:
                os.close(file_descriptor)
        finally:
            _unlock_file(lock_file)


def _write_task_api_log(
    task,
    started_at,
    ended_at,
    attempt_count,
    images,
    error=None,
    prompt_uuid=None,
    task_index=0,
):
    success = error is None
    common = {
        "uuid": str(prompt_uuid or uuid4()),
        "provider": task.provider,
        "model": str(task.params.get("model") or ""),
        "key_name": _task_key_name(task),
        "start_time": started_at.isoformat(timespec="milliseconds"),
        "end_time": ended_at.isoformat(timespec="milliseconds"),
        "retried": attempt_count > 1,
        "attempt_count": attempt_count,
        "size": _task_log_size(task),
        "status": "success" if success else "failure",
        "image_num": int(task.params.get("max_images", 1)),
        "task_index": int(task_index),
    }

    if success:
        records = [dict(common) for _ in images]
    else:
        records = [
            dict(
                common,
                failure_status=_failure_status(error),
                error=str(error),
            )
        ]
    _append_api_log(records, ended_at)


def _test_directive_result(task):
    status = DirectiveFailureError.DIRECTIVES.get(task.prompt.strip().lower())
    if status is None:
        return None

    if status == "success":
        image = _tensor(Image.new("RGB", (512, 512), "green"))
        return [image], "测试指令: [success]\n执行状态: success（未调用图像生成 API）"

    raise DirectiveFailureError(status)


def _parse_json_values(text):
    """Parse one or more JSON values, including common SSE data lines."""
    text = str(text or "").lstrip("\ufeff").strip()
    if text.startswith(")]}'"):
        text = text.split("\n", 1)[1] if "\n" in text else ""

    sse_payloads = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("data:"):
            continue
        payload = stripped[5:].strip()
        if payload and payload != "[DONE]":
            sse_payloads.append(payload)
    if sse_payloads:
        text = "\n".join(sse_payloads)

    decoder = json.JSONDecoder()
    values = []
    position = 0
    while position < len(text):
        while position < len(text) and text[position].isspace():
            position += 1
        if position >= len(text):
            break
        value, position = decoder.raw_decode(text, position)
        values.append(value)
    return values


def _parse_response_json(response, api_name):
    try:
        return response.json()
    except ValueError as original_error:
        try:
            values = _parse_json_values(response.text)
        except (TypeError, ValueError, json.JSONDecodeError):
            values = []
        if len(values) == 1:
            return values[0]
        if len(values) > 1:
            return values

        content_type = response.headers.get("Content-Type", "unknown")
        preview = str(response.text or "")[:500].replace("\r", "\\r").replace("\n", "\\n")
        raise RuntimeError(
            f"{api_name} 返回无法解析的响应 (HTTP {response.status_code}, "
            f"Content-Type={content_type}): {original_error}; 响应开头: {preview or '<empty>'}"
        ) from original_error


def _merge_gemini_response(body):
    """Normalize standard and chunked Gemini responses to one response object."""
    if isinstance(body, dict):
        return body
    if not isinstance(body, list):
        raise RuntimeError(f"Gemini API 返回了不支持的 JSON 类型: {type(body).__name__}")

    merged = {"candidates": []}
    for chunk in body:
        if not isinstance(chunk, dict):
            continue
        if chunk.get("error"):
            return chunk
        candidates = chunk.get("candidates")
        if isinstance(candidates, list):
            merged["candidates"].extend(candidates)
        for metadata_key in ("usageMetadata", "usage_metadata", "modelVersion", "responseId"):
            if metadata_key in chunk:
                merged[metadata_key] = chunk[metadata_key]
    return merged


def _retry(task, operation, execution_uuid=None, task_index=0):
    prompt_uuid = execution_uuid or _execution_uuid()
    started_at = datetime.now().astimezone()
    attempt = 0
    attempts = 2 if task.params.get("enable_auto_retry", True) else 1
    try:
        directive_result = _test_directive_result(task)
        if directive_result is not None:
            images, log = directive_result
            ended_at = datetime.now().astimezone()
            _write_task_api_log(
                task,
                started_at,
                ended_at,
                attempt,
                images,
                prompt_uuid=prompt_uuid,
                task_index=task_index,
            )
            return images, log

        for attempt in range(1, attempts + 1):
            try:
                images, log = operation(task)
            except Exception:
                if attempt == attempts:
                    raise
                time.sleep(1.0)
                continue

            ended_at = datetime.now().astimezone()
            _write_task_api_log(
                task,
                started_at,
                ended_at,
                attempt,
                images,
                prompt_uuid=prompt_uuid,
                task_index=task_index,
            )
            return images, log + f"\n尝试次数: {attempt}"
    except Exception as error:
        try:
            _write_task_api_log(
                task,
                started_at,
                datetime.now().astimezone(),
                attempt,
                (),
                error=error,
                prompt_uuid=prompt_uuid,
                task_index=task_index,
            )
        except Exception as log_error:
            raise RuntimeError(f"{error}; API 日志写入失败: {log_error}") from error
        raise


def _build_gemini_url(base_url, model):
    """Build a Gemini generateContent URL from common base URL forms."""
    base = str(base_url or "").strip().rstrip("/")
    model = str(model or "").strip()
    if not base:
        raise ValueError("Gemini base_url 不能为空")
    if not model:
        raise ValueError("Gemini model 不能为空")

    lower_base = base.lower()
    if lower_base.endswith(":generatecontent"):
        return base
    if "/models/" in lower_base:
        return f"{base}:generateContent"
    if lower_base.endswith("/models"):
        return f"{base}/{model}:generateContent"
    if lower_base.endswith("/v1beta"):
        return f"{base}/models/{model}:generateContent"

    # Modelverse's OpenAI-compatible base is commonly saved as .../v1 in an
    # existing workflow. Gemini uses .../v1beta instead, so avoid /v1/v1beta.
    if lower_base.endswith("/v1"):
        base = base[:-3]
    return f"{base}/v1beta/models/{model}:generateContent"


def _gemini(task):
    p = task.params
    key = _environment_key(
        p.get("key"),
        ("MODELVERSE_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"),
    )

    url = _build_gemini_url(p["base_url"], p["model"])

    parts = [{"text": task.prompt}] + [
        {"inlineData": {"mimeType": "image/png", "data": _png_data_url(image).split(",", 1)[1]}}
        for image in task.images
    ]
    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"],
            "imageConfig": {"aspectRatio": p["aspect_ratio"], "imageSize": p["image_size"]},
        },
    }

    response = requests.post(
        url,
        headers={"x-goog-api-key": key, "Content-Type": "application/json"},
        json=payload,
        timeout=p["timeout"],
    )

    if response.status_code == 404:
        raise RuntimeError(
            f"Gemini API 地址不存在 (HTTP 404): {url}。"
            f"当前 base_url={p['base_url']!r}；请填写服务根地址（例如 "
            "https://api.modelverse.cn），也可填写 /v1beta、/v1beta/models "
            "或完整的 :generateContent 地址。"
        )

    body = _merge_gemini_response(_parse_response_json(response, f"Gemini API ({url})"))
    if not response.ok or body.get("error"):
        raise RuntimeError(f"Gemini API 错误 ({url}): {body.get('error') or response.text}")

    images, texts = [], []
    for candidate in body.get("candidates", []):
        for part in (candidate.get("content") or {}).get("parts", []):
            inline = part.get("inlineData") or part.get("inline_data")
            if inline and inline.get("data") and len(images) < p["max_images"]:
                images.append(_decode_image(inline["data"]))
            elif part.get("text"):
                texts.append(part["text"])

    if not images:
        raise RuntimeError("Gemini API 未返回图像")

    return (
        images,
        f"模型平台: Gemini\n模型: {p['model']}\n提示词: {task.prompt}\n生成数: {len(images)}"
        + ("\n模型文本: " + "\n".join(texts) if texts else ""),
    )


def _gpt(task):
    p = task.params
    key = _environment_key(p.get("key"), ("OPENAI_API_KEY",))
    files = []
    field = "image" if len(task.images) == 1 else "image[]"

    for index, image in enumerate(task.images, 1):
        buffer = io.BytesIO()
        _pil(image).save(buffer, format="PNG")
        files.append((field, (f"image_{index}.png", buffer.getvalue(), "image/png")))

    url = p["base_url"].rstrip("/")
    if not url.endswith("/images/edits"):
        url += "/images/edits"

    data = {
        "model": p["model"],
        "prompt": task.prompt,
        "size": _size(p["aspect_ratio"]),
        "n": str(p["max_images"]),
        "quality": p["quality"],
        "output_format": p["output_format"],
        "output_compression": str(p["output_compression"]),
    }
    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {key}"},
        files=files,
        data=data,
        timeout=p["timeout"],
    )

    body = response.json()
    if not response.ok or body.get("error"):
        raise RuntimeError(f"OpenAI API 错误: {body.get('error') or response.text}")

    images = []
    for item in body.get("data", []):
        if item.get("b64_json"):
            images.append(_decode_image(item["b64_json"]))
        elif item.get("url"):
            images.append(_download(item["url"], p["timeout"]))

    if not images:
        raise RuntimeError("OpenAI API 未返回图像")

    return (
        images,
        f"模型平台: GPT Image\n模型: {p['model']}\n提示词: {task.prompt}\n生成数: {len(images)}",
    )


def _seedream(task):
    p = task.params
    key = _environment_key(p.get("key"), ("ARK_API_KEY",))
    try:
        from volcenginesdkarkruntime import Ark
        from volcenginesdkarkruntime.types.images.images import (
            OptimizePromptOptions,
            SequentialImageGenerationOptions,
        )
    except ImportError as error:
        raise RuntimeError("Seedream 需要安装 volcengine-python-sdk[ark]") from error

    client = Ark(base_url=p["base_url"], api_key=key.strip(), timeout=p["timeout"], max_retries=0)
    model = p["model"]
    manual_size = _manual_pixel_size(p.get("size"))
    size = manual_size or _seedream_size(model, p["aspect_ratio"], p["resolution"])
    size_source = "手动输入" if manual_size else "分辨率与宽高比预设"
    sequential = p["sequential_image_generation"]
    options = SequentialImageGenerationOptions(max_images=p["max_images"])
    extra = {}

    if model == "doubao-seedream-5-0-pro-260628":
        sequential, options = None, None
        extra["optimize_prompt_options"] = OptimizePromptOptions(mode=p["optimize_prompt_options"])

    response = client.images.generate(
        model=model,
        prompt=task.prompt,
        image=[_png_data_url(image) for image in task.images],
        size=size,
        sequential_image_generation=sequential,
        sequential_image_generation_options=options,
        response_format=p["response_format"],
        watermark=p["watermark"],
        stream=p["stream"],
        **extra,
    )

    images = []
    for item in response.data:
        if getattr(item, "url", None):
            images.append(_download(item.url, p["timeout"]))
        elif getattr(item, "b64_json", None):
            images.append(_decode_image(item.b64_json))

    if not images:
        raise RuntimeError("Seedream API 未返回图像")

    return (
        images,
        f"模型平台: Seedream\n模型: {model}\n宽高比: {p['aspect_ratio']}\n"
        f"分辨率: {p['resolution']}\n生成尺寸: {size}\n尺寸来源: {size_source}\n"
        f"提示词: {task.prompt}\n"
        f"生成数: {len(images)}",
    )


def _qwen(task):
    """Call DashScope's Qwen image-editing endpoint with 1–3 local images."""
    p = task.params
    if not 1 <= len(task.images) <= 3:
        raise ValueError("Qwen image editing requires one to three input images.")

    key = _environment_key(p.get("key"), ("DASHSCOPE_API_KEY",))
    content = [{"image": _png_data_url(image)} for image in task.images]
    content.append({"text": task.prompt})
    model = str(p["model"])
    if model == "qwen-image-edit" and p["max_images"] != 1:
        raise ValueError("qwen-image-edit supports only one output image.")

    parameters = {
        "n": p["max_images"],
        "negative_prompt": p["negative_prompt"],
        "watermark": p["watermark"],
        "seed": p["seed"],
    }
    if model != "qwen-image-edit":
        parameters["prompt_extend"] = p["prompt_extend"]
    if model != "qwen-image-edit" and p["size"] != "auto":
        parameters["size"] = p["size"]
    if model.startswith("qwen-image-3.0"):
        parameters["enable_thinking"] = p["enable_thinking"]

    payload = {
        "model": p["model"],
        "input": {"messages": [{"role": "user", "content": content}]},
        "parameters": parameters,
    }
    url = str(p["base_url"] or "").strip()
    if not url:
        raise ValueError("Qwen base_url cannot be empty.")
    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload,
        timeout=p["timeout"],
    )
    body = _parse_response_json(response, f"Qwen API ({url})")
    if not isinstance(body, dict):
        raise RuntimeError(f"Qwen API returned an unsupported JSON type: {type(body).__name__}")
    if not response.ok or body.get("error") or body.get("code"):
        detail = body.get("error") or body.get("message") or body.get("code") or response.text
        raise RuntimeError(f"Qwen API error: {detail}")

    images = []
    for choice in (body.get("output") or {}).get("choices") or []:
        message = choice.get("message") or {}
        for item in message.get("content") or []:
            image_url = item.get("image") if isinstance(item, dict) else None
            if image_url:
                images.append(_download(image_url, p["timeout"]))

    if not images:
        raise RuntimeError("Qwen API returned no images.")

    return (
        images,
        f"Model platform: Qwen\nModel: {p['model']}\nInput images: {len(task.images)}\n"
        f"Size: {p['size']}\nPrompt: {task.prompt}\nGenerated: {len(images)}",
    )


PROVIDERS = {
    "gemini": _gemini,
    "seedream": _seedream,
    "gpt_image": _gpt,
    "qwen": _qwen,
}
FAILURE_SAFETY_CODES = (
    "safety",
    "moderation",
    "content_policy",
    "content filter",
    "policy violation",
)
FAILURE_NETWORK_CODES = (
    "connection",
    "network",
    "name resolution",
    "proxy",
    "ssl",
    "502",
    "503",
    "bad gateway",
    "service unavailable",
)
FAILURE_TIMEOUT_CODES = (
    "timeout",
    "timed out",
    "504",
)


def _is_timeout(err, text) -> bool:
    if not isinstance(err, (TimeoutError, requests.exceptions.Timeout)):
        return False
    return any(word in text for word in FAILURE_TIMEOUT_CODES)


def _failure_status(error):
    if isinstance(error, DirectiveFailureError):
        return error.status

    text = str(error).lower()
    if _is_timeout(error, text):
        return "failure_timeout"

    if any(word in text for word in FAILURE_SAFETY_CODES):
        return "failure_safety"

    if any(word in text for word in FAILURE_NETWORK_CODES):
        return "failure_network"

    return "failure_other"


class MultiAPIImageExecutor:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {f"task_{i}": (TASK_TYPE,) for i in range(1, 7)}
        return {
            "required": {
                "ignore_failure": ("INT", {"default": 0, "min": 0, "max": 1000}),
                "inputcount": ("INT", {"default": 6, "min": 1, "max": 1000}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("images", "log", "status")
    OUTPUT_IS_LIST = (True, False, False)
    FUNCTION = "execute"
    CATEGORY = CATEGORY

    def execute(self, ignore_failure=0, inputcount=6, task_1=None, **kwargs):
        connected_tasks = [task_1]
        connected_tasks.extend(kwargs.get(f"task_{i}") for i in range(2, int(inputcount) + 1))
        connected_tasks = [task for task in connected_tasks if task is not None]
        for task in connected_tasks:
            if not isinstance(task, ImageGenerationTask) or task.provider not in PROVIDERS:
                raise ValueError("收到无效的图像生成任务")

        tasks = [task for task in connected_tasks if task.prompt.strip()]
        ignored_empty_prompts = len(connected_tasks) - len(tasks)
        if not tasks:
            logs = [
                "MultiAPI Image Executor 任务汇总",
                "总任务数: 0",
                "成功任务数: 0",
                "失败任务数: 0",
                f"空提示词忽略数: {ignored_empty_prompts}",
                f"ignore_failure: {ignore_failure}",
                "没有可执行任务，未调用任何图像生成 API。",
            ]
            return [], "\n".join(logs), ""

        results = [None] * len(tasks)
        workers = min(len(tasks), 32)
        execution_uuid = _execution_uuid()
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="api-image") as pool:
            futures = {
                pool.submit(
                    _retry,
                    task,
                    PROVIDERS[task.provider],
                    execution_uuid,
                    index,
                ): index
                for index, task in enumerate(tasks)
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    images, log = future.result()
                    results[index] = {"ok": True, "images": images, "log": log}
                except Exception as error:
                    results[index] = {"ok": False, "error": error, "status": _failure_status(error)}

        failures = sum(not result["ok"] for result in results)
        placeholders = failures > int(ignore_failure)
        output_images, statuses = [], []
        logs = [
            "MultiAPI Image Executor 任务汇总",
            f"总任务数: {len(tasks)}",
            f"成功任务数: {len(tasks) - failures}",
            f"失败任务数: {failures}",
            f"空提示词忽略数: {ignored_empty_prompts}",
            f"ignore_failure: {ignore_failure}",
        ]

        for index, (task, result) in enumerate(zip(tasks, results), 1):
            if result["ok"]:
                output_images.extend(result["images"])
                statuses.extend(["success"] * len(result["images"]))
                logs.append(f"\n===== 任务 {index} =====\n{result['log']}")
            else:
                logs.append(
                    f"\n===== 任务 {index} 失败 =====\n模型平台: {task.provider}\n提示词: {task.prompt}\n状态: {result['status']}\n错误: {result['error']}"
                )
                if placeholders:
                    output_images.append(_tensor(Image.new("RGB", (512, 512), "red")))
                    statuses.append(result["status"])

        if failures and not placeholders:
            logs.append("\n失败任务数未超过 ignore_failure，失败任务已忽略且不输出占位图。")
        elif failures:
            logs.append("\n失败任务数超过 ignore_failure，已按失败任务补充红色占位图。")
        return output_images, "\n".join(logs), "|".join(statuses)


NODE_CLASS_MAPPINGS = {
    "MAIE_GeminiTask": GeminiImageGenTask,
    "MAIE_SeedreamTask": SeedreamImageGenTask,
    "MAIE_GPTImageTask": GPTImageGenTask,
    "MAIE_QwenTask": QwenImageGenTask,
    "MAIE_ExecuteTasks": MultiAPIImageExecutor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MAIE_QwenTask": "Qwen Image Task · MultiAPI Image Executor",
    "MAIE_GeminiTask": "Gemini Task · MultiAPI Image Executor",
    "MAIE_SeedreamTask": "Seedream Task · MultiAPI Image Executor",
    "MAIE_GPTImageTask": "GPT Image Task · MultiAPI Image Executor",
    "MAIE_ExecuteTasks": "Execute Tasks · MultiAPI Image Executor",
}
