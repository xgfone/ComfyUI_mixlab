import math
import os
import time

import numpy as np
import torch
from PIL import Image

try:
    import folder_paths
except Exception:  # Allows linting/testing outside ComfyUI.
    folder_paths = None


WEB_DIRECTORY = "./js"


def _pil_constants():
    transform_perspective = Image.Transform.PERSPECTIVE if hasattr(Image, "Transform") else Image.PERSPECTIVE
    resampling = getattr(Image, "Resampling", Image)
    return transform_perspective, resampling


def _perspective_coefficients(from_points, to_points):
    if len(from_points) != 4 or len(to_points) != 4:
        raise ValueError("from_points and to_points must both contain exactly 4 points.")

    matrix = []
    vector = []
    for (x, y), (u, v) in zip(from_points, to_points):
        matrix.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        matrix.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        vector.extend([u, v])

    matrix = np.asarray(matrix, dtype=np.float64)
    vector = np.asarray(vector, dtype=np.float64)

    try:
        coeffs = np.linalg.solve(matrix, vector)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Corner Pin failed: the 4 target points are degenerate. "
            "Please avoid overlapping points or a fully collapsed quadrilateral."
        ) from exc

    return coeffs.tolist()


def _tensor_to_pil_rgb(image_3d):
    arr = image_3d.detach().cpu().numpy()
    arr = np.clip(arr, 0.0, 1.0)

    if arr.ndim != 3:
        raise ValueError(f"Expected image item shape [H,W,C], got {arr.shape}")

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] >= 3:
        arr = arr[..., :3]
    else:
        raise ValueError(f"Unsupported channel count: {arr.shape[-1]}")

    arr = (arr * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _image_alpha_to_pil_mask(image_3d):
    if image_3d.ndim != 3 or image_3d.shape[-1] < 4:
        return None
    alpha = image_3d[..., 3].detach().cpu().numpy()
    alpha = np.clip(alpha, 0.0, 1.0)
    alpha = (alpha * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(alpha, mode="L")


def _normalize_mask_tensor(mask):
    if mask is None:
        return None
    if not isinstance(mask, torch.Tensor):
        raise ValueError("mask input must be a torch.Tensor")
    if mask.ndim == 2:
        return mask.unsqueeze(0)
    if mask.ndim == 3:
        return mask
    if mask.ndim == 4:
        return mask[..., 0]
    raise ValueError(f"Unsupported mask shape: {tuple(mask.shape)}")


def _mask_item_to_pil(mask_2d, target_width, target_height):
    _, pil_resampling = _pil_constants()
    arr = mask_2d.detach().cpu().numpy()
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0 + 0.5).astype(np.uint8)
    pil_mask = Image.fromarray(arr, mode="L")
    if pil_mask.size != (target_width, target_height):
        pil_mask = pil_mask.resize((target_width, target_height), resample=pil_resampling.BILINEAR)
    return pil_mask


def _pil_rgb_to_tensor(pil_image, device, dtype):
    arr = np.asarray(pil_image.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).to(device=device, dtype=dtype)


def _pil_mask_to_tensor(pil_mask, device, dtype):
    arr = np.asarray(pil_mask.convert("L")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).to(device=device, dtype=dtype)


def _safe_preview_image(image_tensor):
    """Save the first input image as a temp preview for the frontend editor.

    ComfyUI's frontend cannot directly read an upstream IMAGE tensor. Returning a
    ui.images item gives the JS editor an image URL after the node has run once.
    If this fails for any reason, the node still returns normal results.
    """
    if folder_paths is None:
        return []
    try:
        temp_dir = folder_paths.get_temp_directory()
        subfolder = "bimo_corner_pin"
        out_dir = os.path.join(temp_dir, subfolder)
        os.makedirs(out_dir, exist_ok=True)
        filename = f"corner_pin_source_{int(time.time() * 1000)}.png"
        pil_img = _tensor_to_pil_rgb(image_tensor[0])
        pil_img.save(os.path.join(out_dir, filename), compress_level=1)
        return [{"filename": filename, "subfolder": subfolder, "type": "temp"}]
    except Exception:
        return []


def _compute_canvas(width, height, dst_quad, expand_canvas):
    if not expand_canvas:
        return width, height, 0.0, 0.0, dst_quad

    xs = [p[0] for p in dst_quad]
    ys = [p[1] for p in dst_quad]
    min_x = math.floor(min(xs))
    min_y = math.floor(min(ys))
    max_x = math.ceil(max(xs))
    max_y = math.ceil(max(ys))

    # Include at least the original canvas origin in the bounding box so a tiny
    # inward transform does not unexpectedly shrink below 1 pixel.
    out_w = max(1, int(max_x - min_x + 1))
    out_h = max(1, int(max_y - min_y + 1))

    # Avoid creating absurdly huge accidental outputs if a widget is mistyped.
    max_side = 16384
    if out_w > max_side or out_h > max_side:
        raise ValueError(
            f"Expanded canvas is too large: {out_w}x{out_h}. "
            f"Please reduce corner coordinate range. Max side is {max_side}px."
        )

    shifted = [(x - min_x, y - min_y) for x, y in dst_quad]
    return out_w, out_h, float(-min_x), float(-min_y), shifted


class BIMO_CornerPinPerspective:
    """Photoshop-like corner-pin / perspective warp for ComfyUI.

    Coordinates are normalized against the original input image size. Values may
    be outside 0-1. If expand_canvas is enabled, the output canvas expands to fit
    the full transformed quadrilateral.
    """

    CATEGORY = "BIMO AI/image/transform"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "valid_mask")
    FUNCTION = "warp"

    @classmethod
    def INPUT_TYPES(cls):
        coord = {"default": 0.0, "min": -4.0, "max": 5.0, "step": 0.001, "display": "number"}
        color = {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001, "display": "number"}
        return {
            "required": {
                "image": ("IMAGE",),
                "edit_enabled": ("BOOLEAN", {"default": False}),
                "expand_canvas": ("BOOLEAN", {"default": True}),
                "tl_x": ("FLOAT", {**coord, "default": 0.0}),
                "tl_y": ("FLOAT", {**coord, "default": 0.0}),
                "tr_x": ("FLOAT", {**coord, "default": 1.0}),
                "tr_y": ("FLOAT", {**coord, "default": 0.0}),
                "br_x": ("FLOAT", {**coord, "default": 1.0}),
                "br_y": ("FLOAT", {**coord, "default": 1.0}),
                "bl_x": ("FLOAT", {**coord, "default": 0.0}),
                "bl_y": ("FLOAT", {**coord, "default": 1.0}),
                "background_r": ("FLOAT", {**color, "default": 0.0}),
                "background_g": ("FLOAT", {**color, "default": 0.0}),
                "background_b": ("FLOAT", {**color, "default": 0.0}),
                "resampling": (["bicubic", "bilinear", "nearest"], {"default": "bicubic"}),
            },
            "optional": {"mask": ("MASK",)},
        }

    def warp(
        self,
        image,
        edit_enabled,
        expand_canvas,
        tl_x,
        tl_y,
        tr_x,
        tr_y,
        br_x,
        br_y,
        bl_x,
        bl_y,
        background_r,
        background_g,
        background_b,
        resampling,
        mask=None,
    ):
        if image.ndim != 4:
            raise ValueError(f"Expected ComfyUI IMAGE shape [B,H,W,C], got {tuple(image.shape)}")

        device = image.device
        dtype = image.dtype
        batch, height, width, channels = image.shape
        if height < 2 or width < 2:
            raise ValueError("Image width and height must both be at least 2 pixels.")

        mask = _normalize_mask_tensor(mask)
        if mask is not None and mask.shape[0] not in (1, batch):
            raise ValueError(
                f"Mask batch size must be 1 or match image batch size. Got image batch={batch}, mask batch={mask.shape[0]}."
            )

        transform_perspective, pil_resampling = _pil_constants()
        resample_filter = {
            "bicubic": pil_resampling.BICUBIC,
            "bilinear": pil_resampling.BILINEAR,
            "nearest": pil_resampling.NEAREST,
        }.get(resampling, pil_resampling.BICUBIC)

        # Destination points are normalized to the original source dimensions.
        dst_quad_raw = [
            (float(tl_x) * (width - 1), float(tl_y) * (height - 1)),
            (float(tr_x) * (width - 1), float(tr_y) * (height - 1)),
            (float(br_x) * (width - 1), float(br_y) * (height - 1)),
            (float(bl_x) * (width - 1), float(bl_y) * (height - 1)),
        ]

        out_w, out_h, shift_x, shift_y, dst_quad = _compute_canvas(width, height, dst_quad_raw, bool(expand_canvas))

        src_rect = [
            (0.0, 0.0),
            (float(width - 1), 0.0),
            (float(width - 1), float(height - 1)),
            (0.0, float(height - 1)),
        ]
        coeffs = _perspective_coefficients(dst_quad, src_rect)

        fill_color = (
            int(round(float(background_r) * 255)),
            int(round(float(background_g) * 255)),
            int(round(float(background_b) * 255)),
        )

        output_images = []
        output_masks = []
        for i in range(batch):
            pil_img = _tensor_to_pil_rgb(image[i])
            warped = pil_img.transform(
                (out_w, out_h),
                transform_perspective,
                coeffs,
                resample=resample_filter,
                fillcolor=fill_color,
            )

            if mask is not None:
                mask_index = 0 if mask.shape[0] == 1 else i
                source_mask = _mask_item_to_pil(mask[mask_index], width, height)
            else:
                source_mask = _image_alpha_to_pil_mask(image[i])
                if source_mask is None:
                    source_mask = Image.new("L", (width, height), 255)

            warped_mask = source_mask.transform(
                (out_w, out_h),
                transform_perspective,
                coeffs,
                resample=pil_resampling.BILINEAR,
                fillcolor=0,
            )

            output_images.append(_pil_rgb_to_tensor(warped, device, dtype))
            output_masks.append(_pil_mask_to_tensor(warped_mask, device, dtype))

        out_image = torch.stack(output_images, dim=0)
        out_mask = torch.stack(output_masks, dim=0)

        preview_images = _safe_preview_image(image)
        if preview_images:
            return {"ui": {"images": preview_images}, "result": (out_image, out_mask)}
        return (out_image, out_mask)
