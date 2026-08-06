from __future__ import annotations

import io

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from .util.aliyun_facebody_http import (
    AliyunFaceBodyError,
    AliyunFaceBodyHTTPClient,
    resolve_aliyun_credentials,
)

MAX_INPUT_BYTES = 4 * 1024 * 1024
MIN_IMAGE_DIMENSION = 11
MAX_IMAGE_DIMENSION = 1999


class AliyunFaceBeautyNodeError(RuntimeError):
    """Error surfaced by the ComfyUI node."""


class AliyunFaceBeautyNode:
    """Beautify ComfyUI images through Aliyun FaceBeauty without its SDK."""

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "beautify"
    CATEGORY = "Aliyun/FaceBody"
    DESCRIPTION = "使用阿里云人脸美颜API。"
    SEARCH_ALIASES = ("aliyun face beauty", "阿里云美颜", "人脸美颜")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "access_key_id": ("STRING", {"multiline": False, "default": ""}),
                "access_key_secret": ("STRING", {"multiline": False, "default": ""}),
                "sharp": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.1}),  # 锐化
                "smooth": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.1}),  # 磨皮
                "white": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.1}),  # 美白
            },
            "optional": {
                "image": ("IMAGE",),
                "image_url": ("STRING", {"multiline": False, "default": ""}),
            },
        }

    def beautify(
        self,
        access_key_id: str,
        access_key_secret: str,
        sharp: float,
        smooth: float,
        white: float,
        image: torch.Tensor | None = None,
        image_url: str | None = None,
    ) -> tuple[torch.Tensor]:
        try:
            credentials = resolve_aliyun_credentials(access_key_id, access_key_secret)
        except ValueError as exc:
            raise AliyunFaceBeautyNodeError(str(exc)) from exc

        source_url = (image_url or "").strip()
        if image is None and not source_url:
            raise AliyunFaceBeautyNodeError("missing image or image_url")

        output_tensors: list[torch.Tensor] = []
        try:
            with AliyunFaceBodyHTTPClient(credentials) as client:
                if image is not None:
                    self._validate_batch(image)
                    for index in range(image.shape[0]):
                        try:
                            image_bytes = self._tensor_to_png_bytes(image[index])
                            uploaded_url = client.upload_image(image_bytes)
                            result_url = client.face_beauty(
                                uploaded_url,
                                sharp=sharp,
                                smooth=smooth,
                                white=white,
                            )
                            output_tensors.append(
                                self._bytes_to_image_tensor(client.download_image(result_url))
                            )
                        except (
                            AliyunFaceBodyError,
                            UnidentifiedImageError,
                            ValueError,
                            OSError,
                        ) as exc:
                            raise AliyunFaceBeautyNodeError(
                                f"processing {index + 1} in batch failed: {exc}"
                            ) from exc
                else:
                    result_url = client.face_beauty(
                        source_url,
                        sharp=sharp,
                        smooth=smooth,
                        white=white,
                    )
                    output_tensors.append(
                        self._bytes_to_image_tensor(client.download_image(result_url))
                    )
        except AliyunFaceBeautyNodeError:
            raise
        except (AliyunFaceBodyError, ValueError, UnidentifiedImageError, OSError) as exc:
            raise AliyunFaceBeautyNodeError(f"Aliyun FaceBeauty failed: {exc}") from exc

        if not output_tensors:
            raise AliyunFaceBeautyNodeError("Aliyun FaceBeauty returned no images.")

        try:
            return (torch.cat(output_tensors, dim=0),)
        except RuntimeError as exc:
            raise AliyunFaceBeautyNodeError(
                "image results in batch have inconsistent dimensions, "
                "cannot be merged into a single IMAGE."
            ) from exc

    @staticmethod
    def _validate_batch(image: torch.Tensor) -> None:
        if not isinstance(image, torch.Tensor):
            raise AliyunFaceBeautyNodeError("image must be a ComfyUI IMAGE tensor。")
        if image.ndim != 4 or image.shape[0] < 1:
            raise AliyunFaceBeautyNodeError(
                f"image should be [B,H,W,C] and batch is not empty, but got shape={tuple(image.shape)}。"
            )
        if image.shape[-1] not in (1, 3, 4):
            raise AliyunFaceBeautyNodeError(
                f"image shape must be one of (1, 3, 4), but got {image.shape[-1]}。"
            )

    @staticmethod
    def _tensor_to_png_bytes(image_tensor: torch.Tensor) -> bytes:
        height, width, channels = image_tensor.shape
        if not (
            MIN_IMAGE_DIMENSION <= height <= MAX_IMAGE_DIMENSION
            and MIN_IMAGE_DIMENSION <= width <= MAX_IMAGE_DIMENSION
        ):
            raise ValueError(
                "Aliyun FaceBeauty requires image dimensions to be between 10 and 2000 pixels;"
                f" current is {width}x{height}。"
            )

        image_array = image_tensor.detach().to(device="cpu", dtype=torch.float32).numpy()
        image_array = np.nan_to_num(image_array, nan=0.0, posinf=1.0, neginf=0.0)
        image_array = np.clip(np.rint(image_array * 255.0), 0, 255).astype(np.uint8)

        if channels == 1:
            image_array = np.repeat(image_array, 3, axis=2)
        elif channels == 4:
            # FaceBeauty returns RGB. Composite transparent inputs on white so
            # their visible appearance is deterministic before processing.
            rgba = Image.fromarray(image_array, mode="RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            image_array = np.asarray(Image.alpha_composite(background, rgba).convert("RGB"))

        output = io.BytesIO()
        Image.fromarray(image_array, mode="RGB").save(output, format="PNG")
        image_bytes = output.getvalue()
        if len(image_bytes) > MAX_INPUT_BYTES:
            raise ValueError(
                "PNG image exceeds Aliyun FaceBeauty's 4 MB input limit: "
                f"{len(image_bytes) / (1024 * 1024):.2f} MB。"
            )
        return image_bytes

    @staticmethod
    def _bytes_to_image_tensor(image_bytes: bytes) -> torch.Tensor:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.load()
            image_array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(image_array.copy()).unsqueeze(0)
