# -*- coding: utf-8 -*-
"""
ComfyUI custom node: Doubao Chat (Single Turn, Ark SDK) — with Images & Deep Thinking

- Uses the official Ark Runtime SDK: `from volcenginesdkarkruntime import Ark`
- Single-turn chat with optional system prompt
- Supports up to 3 optional image inputs (as ComfyUI IMAGE tensors). Images are sent as base64 data URLs.
- Adds a boolean switch to enable "Deep Thinking" (thinking={"type": "enabled"}).

Setup
-----
1) Install SDK + Pillow in the same Python env that runs ComfyUI:
   pip install volcengine-python-sdk pillow

2) Put this file into:
   ComfyUI/custom_nodes/doubao_node_sdk_v2.py
   or inside a folder:
   ComfyUI/custom_nodes/DoubaoNode/doubao_node_sdk_v2.py

3) Restart ComfyUI. Look for node: "Doubao Chat (Single Turn, Ark SDK, Img+Thinking)".

4) Credentials:
   - Prefer setting environment variable: ARK_API_KEY
   - Or edit the generated config.json next to this file:
       {
         "ARK_API_KEY": "your_api_key",
         "base_url": "https://ark.cn-beijing.volces.com/api/v3",
         "default_model": "ep-xxxxxxxxxxxxxxxx",
         "default_temperature": 0.7,
         "default_top_p": 0.95,
         "default_max_tokens": 1024,
         "prefer_env_api_key": true
       }
   - "model" should usually be your Ark inference endpoint id (starts with "ep-").
     Some tenants may allow direct model names.

Notes
-----
- Image content is packed as OpenAI-style message content items:
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,<...>"}}.
- Deep Thinking toggle maps to: extra_body={"thinking": {"type": "enabled"}}
  If you need "auto"/"disabled" modes later, we can replace the boolean with an ENUM.
"""

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

try:
    # Official Ark Runtime SDK import (as per docs)
    from volcenginesdkarkruntime import Ark  # type: ignore
except Exception as _e:
    Ark = None  # will check later


# -------------------------
# Config helpers
# -------------------------


def _config_path() -> str:
    here = os.path.dirname(__file__)
    return os.path.join(os.path.abspath(here), "config.json")


def _default_config() -> Dict[str, Any]:
    return {
        "__comment": "Set ARK_API_KEY and default_model (typically an ep-xxxxxxxx endpoint id).",
        "ARK_API_KEY": "your_api_key",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "default_model": "ep-xxxxxxxxxxxxxxxx",
        "default_temperature": 0.7,
        "default_top_p": 0.95,
        "default_max_tokens": 1024,
        "prefer_env_api_key": True,
    }


def _load_config() -> Dict[str, Any]:
    cfg_path = _config_path()
    if not os.path.exists(cfg_path):
        cfg = _default_config()
        try:
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
            print("[DoubaoNodeSDKv2] config.json created. Please set ARK_API_KEY & default_model.")
        except Exception as e:
            print(f"[DoubaoNodeSDKv2] Failed to write config.json: {e}")
        return cfg
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[DoubaoNodeSDKv2] Failed to read config.json: {e}")
        return _default_config()


CONFIG = _load_config()


def _get_api_key(cfg: Dict[str, Any]) -> str:
    # Prefer environment variable if set, unless disabled by config
    if cfg.get("prefer_env_api_key", True):
        env_key = os.environ.get("ARK_API_KEY", "").strip()
        if env_key:
            return env_key
    key = (cfg.get("ARK_API_KEY") or "").strip()
    if not key or key == "your_api_key":
        raise ValueError("[DoubaoNodeSDKv2] Missing ARK_API_KEY. Set env ARK_API_KEY or config.json.")
    return key


def _get_base_url(cfg: Dict[str, Any]) -> str:
    base = (cfg.get("base_url") or "").strip().rstrip("/")
    if not base:
        base = "https://ark.cn-beijing.volces.com/api/v3"
    return base


def _get_client(max_retries: int = 2, timeout: int = 600):
    if Ark is None:
        raise RuntimeError(
            "[DoubaoNodeSDKv2] Ark SDK import failed. Install it in your ComfyUI environment:\n"
            "  pip install volcengine-python-sdk\n"
            "and ensure your Python can `from volcenginesdkarkruntime import Ark`."
        )
    api_key = _get_api_key(CONFIG)
    base_url = _get_base_url(CONFIG)

    # Initialize Ark client (OpenAI-compatible)
    return Ark(base_url=base_url, api_key=api_key, max_retries=max_retries, timeout=timeout)


# -------------------------
# Image helpers (ComfyUI tensor -> PNG base64 data URL)
# -------------------------


def _tensor_to_pil(image_tensor) -> Optional[Image.Image]:
    """
    Accepts ComfyUI IMAGE (torch.Tensor or np.ndarray) shaped [B,H,W,C] in 0..1 float.
    Returns a PIL.Image of the FIRST frame (B index 0).
    """
    try:
        # Lazy import torch if available
        import torch  # type: ignore

        if isinstance(image_tensor, torch.Tensor):
            arr = image_tensor[0].detach().cpu().numpy()
        else:
            arr = np.asarray(image_tensor)[0]
    except Exception:
        try:
            arr = np.asarray(image_tensor)[0]
        except Exception:
            return None

    # Clip & convert to uint8
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).round().astype(np.uint8)  # [H, W, C]

    # Handle channel count
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)  # make RGB
    mode = "RGBA" if arr.shape[-1] == 4 else "RGB"
    return Image.fromarray(arr, mode=mode)


def _pil_to_data_url_png(img: Image.Image) -> str:
    import base64
    import io

    bio = io.BytesIO()
    img.save(bio, format="PNG")
    b64 = base64.b64encode(bio.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


# -------------------------
# Core call (Ark SDK)
# -------------------------


def _ark_chat_single_turn(
    model: str,
    messages: list,
    temperature: float,
    top_p: float,
    max_tokens: int,
    enable_deep_thinking: bool,
    max_retries: int = 2,
    timeout: int = 600,
) -> str:
    client = _get_client(max_retries=max_retries, timeout=timeout)

    extra_body = None
    if enable_deep_thinking:
        # enabled / disabled / auto are common options; we expose a boolean "enabled" here
        extra_body = {"thinking": {"type": "enabled"}}

    try:
        if extra_body is None:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=float(temperature),
                top_p=float(top_p),
                max_tokens=int(max_tokens),
            )
        else:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=float(temperature),
                top_p=float(top_p),
                max_tokens=int(max_tokens),
                extra_body=extra_body,
            )
    except Exception as e:
        raise RuntimeError(
            "[DoubaoNodeSDKv2] Ark SDK request failed. "
            "Check network egress, base_url region, and that 'model' is a valid inference endpoint id (ep-...). "
            f"Original error: {e}"
        ) from e

    # extract assistant text
    try:
        choice0 = resp.choices[0]
        content = getattr(choice0.message, "content", None)
        if content is None:
            content = choice0.get("message", {}).get("content", "")
        return (content or "").strip()
    except Exception:
        try:
            return str(resp)
        except Exception:
            return ""


# -------------------------
# ComfyUI Node
# -------------------------


class DoubaoSingleTurnChatNodeSDKv2:
    """
    Single-turn Doubao chat via Ark SDK.
    - Optional system prompt
    - Up to 3 optional images (IMAGE tensors) as visual context
    - Boolean to enable Deep Thinking (thinking={"type":"enabled"})
    - Returns assistant text
    """

    @classmethod
    def INPUT_TYPES(cls):
        cfg = CONFIG
        max_tokens = int(cfg.get("default_max_tokens", 1024))
        temperature = float(cfg.get("default_temperature", 0.7))
        top_p = float(cfg.get("default_top_p", 0.95))
        return {
            "required": {
                "model": (
                    "STRING",
                    {
                        "default": str(cfg.get("default_model", "ep-xxxxxxxxxxxxxxxx")),
                        "multiline": False,
                        "tooltip": "Ark inference endpoint id (e.g., ep-xxx). Some tenants may allow model-name.",
                    },
                ),
                "user_prompt": ("STRING", {"default": "", "multiline": True, "placeholder": "Ask Doubao something..."}),
                "max_tokens": ("INT", {"default": max_tokens, "min": 1, "max": 8192, "step": 1}),
                "temperature": ("FLOAT", {"default": temperature, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": top_p, "min": 0.0, "max": 1.0, "step": 0.01}),
                "deep_thinking": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "timeout": ("INT", {"default": 600, "tooltip": "超时时间，单位秒"}),
                "max_retries": ("INT", {"default": 2, "tooltip": "最大重试次数"}),
                "system_prompt": ("STRING", {"default": "", "multiline": True, "placeholder": "system prompt"}),
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "chat"
    CATEGORY = "Doubao/Chat"

    def chat(
        self,
        model: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        deep_thinking: bool,
        system_prompt: str = "",
        max_retries: int = 2,
        timeout: int = 600,
        image_1=None,
        image_2=None,
        image_3=None,
    ):
        model = (model or "").strip()
        if not model:
            raise ValueError("[DoubaoNodeSDKv2] 'model' is required. Usually an ep-xxxx inference endpoint id.")

        if not isinstance(user_prompt, str) or not user_prompt.strip():
            return ("",)

        # Build content items
        content_items: List[Dict[str, Any]] = []
        # Put text first (common practice)
        if user_prompt and user_prompt.strip():
            content_items.append({"type": "text", "text": user_prompt.strip()})

        # Convert up to 3 images if provided
        for img_tensor in (image_1, image_2, image_3):
            if img_tensor is None:
                continue
            pil = _tensor_to_pil(img_tensor)
            if pil is None:
                continue
            url = _pil_to_data_url_png(pil)
            content_items.append({"type": "image_url", "image_url": {"url": url}})

        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        # If no images and no text, short-circuit
        if not content_items:
            return ("",)
        messages.append({"role": "user", "content": content_items})

        text = _ark_chat_single_turn(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            enable_deep_thinking=deep_thinking,
            max_retries=max_retries,
            timeout=timeout,
        )
        return (text,)
