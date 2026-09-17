"""fal.ai BiRefNet v2 background-removal node for ComfyUI."""

import base64
import io
import time

import numpy as np
import requests
import torch
from PIL import Image

from .util.key_resolver import get_key

FAL_MODEL_ID = "fal-ai/birefnet/v2"
# The durable queue endpoint is more reliable than the synchronous fal.run
# endpoint, especially for models that need to start a worker.
FAL_QUEUE_URL = f"https://queue.fal.run/{FAL_MODEL_ID}"
MODEL_OPTIONS = [
    "General Use (Light)",
    "General Use (Light 2K)",
    "General Use (Heavy)",
    "Matting",
    "Portrait",
    "General Use (Dynamic)",
]
RESOLUTION_OPTIONS = ["1024x1024", "2048x2048", "2304x2304"]


def _log(message):
    print(f"[BIMO fal.ai BiRefNet] {message}")


def _to_data_url(image):
    """Encode one ComfyUI IMAGE item as a PNG data URI accepted by fal."""
    pixels = np.clip(image.detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
    pil_image = Image.fromarray(pixels).convert("RGB")
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _image_tensor(data):
    return torch.from_numpy(np.asarray(data.convert("RGB"), dtype=np.float32) / 255.0).unsqueeze(0)


def _mask_tensor(data):
    return torch.from_numpy(np.asarray(data.convert("L"), dtype=np.float32) / 255.0).unsqueeze(0)


def _load_remote_image(url, timeout):
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).copy()


def _request_json(method, url, headers, timeout, **kwargs):
    """Send one fal queue request and surface its server response on failure."""
    try:
        response = requests.request(method, url, headers=headers, timeout=timeout, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as error:
        detail = getattr(error.response, "text", "") if getattr(error, "response", None) else ""
        raise RuntimeError(f"fal.ai {method} request failed: {detail[:1000] or error}") from error
    except ValueError as error:
        raise RuntimeError(f"fal.ai returned invalid JSON for {method} {url}.") from error


class FalBiRefNetRemoveBackground:
    """Remove image backgrounds with fal.ai's ``fal-ai/birefnet/v2`` endpoint."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "key": (
                    "STRING",
                    {
                        "default": "FAL_KEY",
                        "placeholder": "FAL_KEY",
                        "tooltip": "Name of the key in ComfyUI/conf/keys.json or an environment variable; the key value is never stored in the workflow.",
                    },
                ),
                "model": (MODEL_OPTIONS, {"default": "General Use (Light)"}),
                "operating_resolution": (RESOLUTION_OPTIONS, {"default": "1024x1024"}),
                "refine_foreground": ("BOOLEAN", {"default": True}),
                "mask_only": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Return only the segmentation mask. This skips foreground refinement and avoids downloading the background-removed image.",
                    },
                ),
                "output_format": (["png", "webp", "gif"], {"default": "png"}),
                "timeout": ("INT", {"default": 120, "min": 10, "max": 600, "step": 1}),
                "poll_interval": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 10.0, "step": 0.5}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "remove_background"
    CATEGORY = "BIMO/fal.ai"
    DESCRIPTION = "Remove backgrounds with fal-ai/birefnet/v2. The API key is read from ComfyUI/conf/keys.json, then the environment variable named in key."

    def remove_background(
        self,
        image,
        key,
        model,
        operating_resolution,
        refine_foreground,
        mask_only,
        output_format,
        timeout,
        poll_interval,
    ):
        key_name = str(key).strip() or "FAL_KEY"
        api_key = get_key(key_name)

        headers = {"Authorization": f"Key {api_key}", "Content-Type": "application/json"}
        output_images, output_masks = [], []
        batch_started = time.perf_counter()
        for index, item in enumerate(image, start=1):
            item_started = time.perf_counter()
            encode_started = time.perf_counter()
            image_data_url = _to_data_url(item)
            encode_seconds = time.perf_counter() - encode_started
            _log(
                f"Image {index}/{len(image)}: PNG/Base64 encoding finished in {encode_seconds:.2f}s."
            )

            payload = {
                "image_url": image_data_url,
                "model": model,
                "operating_resolution": operating_resolution,
                "output_mask": True,
                # fal skips refinement when mask_only is true; explicitly
                # disabling it also keeps the submitted request unambiguous.
                "refine_foreground": False if mask_only else refine_foreground,
                "mask_only": mask_only,
                "output_format": output_format,
            }
            submit_started = time.perf_counter()
            submitted = _request_json(
                "POST", FAL_QUEUE_URL, headers, min(timeout, 30), json=payload
            )
            submit_seconds = time.perf_counter() - submit_started
            request_id = submitted.get("request_id")
            status_url = submitted.get("status_url")
            response_url = submitted.get("response_url")
            if not request_id or not status_url or not response_url:
                raise RuntimeError(
                    f"fal.ai queue submission returned an incomplete response: {submitted}"
                )
            _log(
                f"Image {index}/{len(image)}: submitted in {submit_seconds:.2f}s (request {request_id})."
            )

            deadline = time.monotonic() + timeout
            queue_started = time.perf_counter()
            inference_started = None
            status_request_seconds = 0.0
            final_status = None
            while True:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"fal.ai request {request_id} did not finish within {timeout} seconds."
                    )
                status_started = time.perf_counter()
                status = _request_json(
                    "GET", status_url, headers, min(timeout, 30), params={"logs": "1"}
                )
                status_request_seconds += time.perf_counter() - status_started
                state = status.get("status")
                if state == "IN_PROGRESS" and inference_started is None:
                    inference_started = time.perf_counter()
                    _log(
                        f"Image {index}/{len(image)}: left queue after "
                        f"{inference_started - queue_started:.2f}s."
                    )
                if state == "COMPLETED":
                    if status.get("error"):
                        raise RuntimeError(f"fal.ai request {request_id} failed: {status['error']}")
                    final_status = status
                    break
                if state in {"FAILED", "CANCELLED"} or status.get("error"):
                    raise RuntimeError(
                        f"fal.ai request {request_id} failed: {status.get('error') or status}"
                    )
                if state not in {"IN_QUEUE", "IN_PROGRESS"}:
                    raise RuntimeError(
                        f"fal.ai request {request_id} returned an unknown status: {status}"
                    )
                time.sleep(poll_interval)

            completed_at = time.perf_counter()
            queue_seconds = (inference_started or completed_at) - queue_started
            inference_seconds = completed_at - (inference_started or queue_started)
            metrics = final_status.get("metrics") if isinstance(final_status, dict) else None
            metrics_note = f", service metrics={metrics}" if metrics else ""
            _log(
                f"Image {index}/{len(image)}: queue={queue_seconds:.2f}s, "
                f"inference wait={inference_seconds:.2f}s, status HTTP={status_request_seconds:.2f}s{metrics_note}."
            )

            result_started = time.perf_counter()
            result = _request_json("GET", response_url, headers, min(timeout, 30))
            result_seconds = time.perf_counter() - result_started

            image_url = (result.get("image") or {}).get("url")
            if not image_url:
                raise RuntimeError(f"fal.ai response has no output image URL: {result}")
            image_download_started = time.perf_counter()
            output = _load_remote_image(image_url, timeout)
            image_download_seconds = time.perf_counter() - image_download_started

            if mask_only:
                # In mask-only mode the API's required `image` output is the
                # segmentation mask, so only one result file is downloaded.
                output_images.append(_image_tensor(output))
                output_masks.append(_mask_tensor(output))
                mask_download_seconds = 0.0
            else:
                output_images.append(_image_tensor(output))
                mask_url = (result.get("mask_image") or {}).get("url")
                if mask_url:
                    mask_download_started = time.perf_counter()
                    output_masks.append(_mask_tensor(_load_remote_image(mask_url, timeout)))
                    mask_download_seconds = time.perf_counter() - mask_download_started
                elif "A" in output.getbands():
                    output_masks.append(_mask_tensor(output.getchannel("A")))
                    mask_download_seconds = 0.0
                else:
                    output_masks.append(
                        torch.ones((1, output.height, output.width), dtype=torch.float32)
                    )
                    mask_download_seconds = 0.0
            _log(
                f"Image {index}/{len(image)}: result metadata={result_seconds:.2f}s, "
                f"image download={image_download_seconds:.2f}s, mask download={mask_download_seconds:.2f}s, "
                f"total={time.perf_counter() - item_started:.2f}s."
            )

        _log(
            f"Batch of {len(image)} image(s) finished in {time.perf_counter() - batch_started:.2f}s."
        )
        return torch.cat(output_images, dim=0), torch.cat(output_masks, dim=0)


NODE_CLASS_MAPPINGS = {
    "FalBiRefNetRemoveBackground": FalBiRefNetRemoveBackground,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FalBiRefNetRemoveBackground": "BIMO fal.ai BiRefNet v2 Remove Background",
}
