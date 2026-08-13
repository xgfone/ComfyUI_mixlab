import base64
import io
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import requests
import torch
from PIL import Image

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
            "optional": _optional_images(),
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
                        "2:3",
                        "3:2",
                        "4:3",
                        "3:4",
                        "16:9",
                        "9:16",
                        "21:9",
                        "2K",
                        "3K",
                        "3.5K",
                        "4K",
                    ],
                ),
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
            "optional": _optional_images(),
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
            "optional": _optional_images(),
        }

    RETURN_TYPES = (TASK_TYPE,)
    RETURN_NAMES = ("task",)
    FUNCTION = "submit"
    CATEGORY = CATEGORY

    def submit(self, prompt, image1, **kwargs):
        return _task("gpt_image", prompt, kwargs, {"image1": image1, **kwargs})


## GenTask End
#############################################################################


@dataclass(frozen=True)
class ImageGenerationTask:
    provider: str
    prompt: str
    images: Tuple[torch.Tensor, ...]
    params: Dict[str, Any]


def _images(kwargs):
    result = []
    for index in range(1, 7):
        image = kwargs.get(f"image{index}")
        if image is not None:
            result.append(image)
    return tuple(result)


def _task(provider, prompt, params, kwargs):
    prompt = str(prompt or "").strip()
    if not prompt:
        raise ValueError("提示词不能为空")
    images = _images(kwargs)
    if not images:
        raise ValueError("至少需要一张参考图")
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


def _retry(task, operation):
    attempts = 2 if task.params.get("enable_auto_retry", True) else 1
    for attempt in range(1, attempts + 1):
        try:
            images, log = operation(task)
            return images, log + f"\n尝试次数: {attempt}"
        except Exception:
            if attempt == attempts:
                raise
            time.sleep(1.0)


def _gemini(task):
    p = task.params
    key = (
        os.getenv("MODELVERSE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )
    if not key:
        raise ValueError("未设置 MODELVERSE_API_KEY、GEMINI_API_KEY 或 GOOGLE_API_KEY")
    base = p["base_url"].rstrip("/")
    if base.endswith("/v1beta"):
        url = f"{base}/models/{p['model']}:generateContent"
    elif base.endswith("/models"):
        url = f"{base}/{p['model']}:generateContent"
    else:
        url = f"{base}/v1beta/models/{p['model']}:generateContent"
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
    body = response.json()
    if not response.ok or body.get("error"):
        raise RuntimeError(f"Gemini API 错误: {body.get('error') or response.text}")
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
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("未设置 OPENAI_API_KEY")
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
    key = os.getenv("ARK_API_KEY")
    if not key:
        raise ValueError("未设置 ARK_API_KEY")
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
        size=_size(p["aspect_ratio"]),
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
        f"模型平台: Seedream\n模型: {model}\n提示词: {task.prompt}\n生成数: {len(images)}",
    )


PROVIDERS = {"gemini": _gemini, "seedream": _seedream, "gpt_image": _gpt}


def _failure_status(error):
    text = str(error).lower()
    if isinstance(error, (TimeoutError, requests.exceptions.Timeout)) or any(
        word in text for word in ("timeout", "timed out", "504")
    ):
        return "failure_timeout"
    if any(
        word in text
        for word in ("safety", "moderation", "content_policy", "content filter", "policy violation")
    ):
        return "failure_safety"
    if any(
        word in text
        for word in (
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
    ):
        return "failure_network"
    return "failure_other"


class MultiAPIImageExecutor:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {f"task_{i}": (TASK_TYPE,) for i in range(2, 7)}
        return {
            "required": {
                "task_1": (TASK_TYPE,),
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

    def execute(self, task_1, ignore_failure=0, inputcount=6, **kwargs):
        tasks = [task_1]
        tasks.extend(kwargs.get(f"task_{i}") for i in range(2, int(inputcount) + 1))
        tasks = [task for task in tasks if task is not None]
        for task in tasks:
            if not isinstance(task, ImageGenerationTask) or task.provider not in PROVIDERS:
                raise ValueError("收到无效的图像生成任务")

        results = [None] * len(tasks)
        workers = min(len(tasks), 32)
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="api-image") as pool:
            futures = {
                pool.submit(_retry, task, PROVIDERS[task.provider]): index
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
    "MAIE_ExecuteTasks": MultiAPIImageExecutor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MAIE_GeminiTask": "Gemini Task · MultiAPI Image Executor",
    "MAIE_SeedreamTask": "Seedream Task · MultiAPI Image Executor",
    "MAIE_GPTImageTask": "GPT Image Task · MultiAPI Image Executor",
    "MAIE_ExecuteTasks": "Execute Tasks · MultiAPI Image Executor",
}
