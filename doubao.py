# -*- coding: utf-8 -*-
"""
ComfyUI custom node: Doubao Chat (Single Turn, Ark SDK)

- Uses the official Ark Runtime SDK: `from volcenginesdkarkruntime import Ark`
- Single-turn text chat (system + user -> assistant)
- OpenAI-compatible chat.completions interface under the hood

Setup
-----
1) Install SDK in the same Python env that runs ComfyUI:
   pip install volcengine-python-sdk

   (If your environment provides a different package name for Ark runtime,
    follow the official docs. Import path used here is:
    `from volcenginesdkarkruntime import Ark`.)

2) Put this file into:
   ComfyUI/custom_nodes/doubao_node_sdk.py
   or inside a folder:
   ComfyUI/custom_nodes/DoubaoNode/doubao_node_sdk.py

3) Restart ComfyUI. Look for node: "Doubao Chat (Single Turn, Ark SDK)".

4) Credentials:
   - Prefer setting environment variable: ARK_API_KEY
   - Or edit the generated config.json next to this file:
       {
         "ARK_API_KEY": "your_api_key",
         "base_url": "https://ark.cn-beijing.volces.com/api/v3",
         "default_model": "ep-xxxxxxxxxxxxxxxx",
         "request_timeout_sec": 30,
         "default_temperature": 0.7,
         "default_top_p": 0.95,
         "default_max_tokens": 1024
       }
   - "model" should usually be your Ark inference endpoint id (starts with "ep-"),
     unless your current plan/doc allows direct model-name calling.

Notes
-----
- This node focuses on *single-turn* text. You can extend it for streaming
  or multi-part content (images, tools) by following the Ark SDK docs.

"""

import json
import os
from typing import Any, Dict

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
    return os.path.join(os.path.abspath(here), "doubao_config.json")


def _default_config() -> Dict[str, Any]:
    return {
        "__comment": "Set ARK_API_KEY and default_model (typically an ep-xxxxxxxx endpoint id).",
        "ARK_API_KEY": "your_api_key",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "default_model": "doubao-seed-1-6-flash-250828",
        "request_timeout_sec": 30,
        "default_temperature": 0.7,
        "default_top_p": 0.95,
        "default_max_tokens": 1024,
    }


def _load_config() -> Dict[str, Any]:
    cfg_path = _config_path()
    if not os.path.exists(cfg_path):
        cfg = _default_config()
        try:
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
            print("[DoubaoNodeSDK] config.json created. Please set ARK_API_KEY & default_model.")
        except Exception as e:
            print(f"[DoubaoNodeSDK] Failed to write config.json: {e}")
        return cfg
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[DoubaoNodeSDK] Failed to read config.json: {e}")
        return _default_config()


CONFIG = _load_config()


def _get_api_key(cfg: Dict[str, Any]) -> str:
    env_key = os.environ.get("ARK_API_KEY", "").strip()
    if env_key:
        return env_key

    key = (cfg.get("ARK_API_KEY") or "").strip()
    if not key or key == "your_api_key":
        raise ValueError("[DoubaoNodeSDK] Missing ARK_API_KEY. Set env ARK_API_KEY or config.json.")
    return key


def _get_base_url(cfg: Dict[str, Any]) -> str:
    base = (cfg.get("base_url") or "").strip().rstrip("/")
    if not base:
        base = "https://ark.cn-beijing.volces.com/api/v3"
    return base


_CLIENT_SINGLETON = None


def _get_client():
    global _CLIENT_SINGLETON
    if _CLIENT_SINGLETON is not None:
        return _CLIENT_SINGLETON

    if Ark is None:
        raise RuntimeError(
            "[DoubaoNodeSDK] Ark SDK import failed. Install it in your ComfyUI environment:\n"
            "  pip install volcengine-python-sdk\n"
            "and ensure your Python can `from volcenginesdkarkruntime import Ark`.\n"
            "If your platform uses a different package name, follow the official docs."
        )
    api_key = _get_api_key(CONFIG)
    base_url = _get_base_url(CONFIG)

    # Initialize Ark client (OpenAI-compatible)
    _CLIENT_SINGLETON = Ark(base_url=base_url, api_key=api_key)
    return _CLIENT_SINGLETON


# -------------------------
# Core call (Ark SDK)
# -------------------------


def _ark_chat_single_turn(model: str, messages: list, temperature: float, top_p: float, max_tokens: int) -> str:
    client = _get_client()

    # The Ark SDK returns an OpenAI-style object.
    # We do not pass timeout here; use SDK defaults. Adjust if SDK later exposes request options.
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=float(temperature),
            top_p=float(top_p),
            max_tokens=int(max_tokens),
        )
    except Exception as e:
        # Bubble up clear hints to troubleshoot
        raise RuntimeError(
            "[DoubaoNodeSDK] Ark SDK request failed. "
            "Check network egress, base_url region, and that 'model' is a valid inference endpoint id (ep-...). "
            f"Original error: {e}"
        ) from e

    # extract assistant text
    try:
        # OpenAI-compatible: choices[0].message.content
        choice0 = resp.choices[0]
        # Some SDKs expose .message.content, some .delta in streams; we use message here.
        content = getattr(choice0.message, "content", None)
        if content is None:
            # fallback if resp is dict-like
            content = choice0.get("message", {}).get("content", "")
        return (content or "").strip()
    except Exception:
        # return raw for debugging
        try:
            return str(resp)
        except Exception:
            return ""


# -------------------------
# ComfyUI Node
# -------------------------


class DoubaoSingleTurnChatNodeSDK:
    """
    Single-turn Doubao chat via Ark SDK.
    - Optional system prompt
    - Required user prompt
    - Returns assistant text
    """

    @classmethod
    def INPUT_TYPES(cls):
        cfg = CONFIG
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
                "user_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Ask Doubao something...",
                    },
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": int(cfg.get("default_max_tokens", 1024)),
                        "min": 1,
                        "max": 8192,
                        "step": 1,
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": float(cfg.get("default_temperature", 0.7)),
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": float(cfg.get("default_top_p", 0.95)),
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
            },
            "optional": {
                "system_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Optional system prompt (role instruction)...",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "chat"
    CATEGORY = "Doubao/Chat"

    def chat(
        self, model: str, user_prompt: str, max_tokens: int, temperature: float, top_p: float, system_prompt: str = ""
    ):
        model = (model or "").strip()
        if not model:
            raise ValueError("[DoubaoNodeSDK] 'model' is required. Usually an ep-xxxx inference endpoint id.")

        if not isinstance(user_prompt, str) or not user_prompt.strip():
            return ("",)

        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": user_prompt.strip()})

        text = _ark_chat_single_turn(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        return (text,)
