"""Standalone YOLOv8 detection node for ComfyUI.

Place YOLO ``.pt`` models in ``ComfyUI/models/yolo``.  Requires the
``ultralytics`` Python package, which supplies the YOLOv8 runtime.
"""

import glob
import os

import numpy as np
import torch
from PIL import Image

import folder_paths


MODEL_DIRECTORY = os.path.join(folder_paths.models_dir, "yolo")


def _log(message, level="info"):
    print(f"[BIMO YOLOv8 Detect] {level.upper()}: {message}")


def _pil_to_image_tensor(image):
    """Convert a PIL RGB image to ComfyUI IMAGE format: [1, H, W, C]."""
    return torch.from_numpy(np.asarray(image).astype(np.float32) / 255.0).unsqueeze(0)


def _pil_to_mask_tensor(image):
    """Convert a PIL grayscale image to ComfyUI MASK format: [1, H, W]."""
    pixels = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(pixels).unsqueeze(0)


def _tensor_to_pil(image):
    pixels = image.detach().cpu().float().squeeze(0).numpy()
    pixels = np.clip(pixels * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(pixels).convert("RGB")


def _parse_classes(value):
    """Parse class IDs such as ``0,2,5-7``; return None for all classes."""
    if not value or not value.strip():
        return None
    try:
        classes = []
        for part in value.split(","):
            part = part.strip()
            if "-" in part:
                start, end = (int(number.strip()) for number in part.split("-", 1))
                if start > end:
                    raise ValueError("range start is greater than end")
                classes.extend(range(start, end + 1))
            else:
                classes.append(int(part))
        return classes
    except ValueError:
        _log(f"Invalid classes value '{value}'; using all classes.", "warning")
        return None


class BimoYoloV8Detect:
    """Run a YOLOv8 detection or segmentation model and return ComfyUI masks."""

    @classmethod
    def INPUT_TYPES(cls):
        model_files = sorted(
            os.path.basename(path) for path in glob.glob(os.path.join(MODEL_DIRECTORY, "*.pt"))
        )
        if not model_files:
            model_files = ["(no .pt models in models/yolo)"]
        return {
            "required": {
                "image": ("IMAGE",),
                "yolo_model": (model_files,),
                "mask_merge": (["all", "1", "2", "3", "4", "5", "6", "7", "8", "9"],),
            },
            "optional": {
                "conf": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "iou": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01}),
                "classes": ("STRING", {"default": "", "multiline": False}),
                "device": ("STRING", {"default": "auto"}),
                "max_det": ("INT", {"default": 300, "min": 1, "max": 1000, "step": 1}),
                "retina_masks": ("BOOLEAN", {"default": True}),
                "agnostic_nms": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE", "MASK")
    RETURN_NAMES = ("mask", "yolo_plot_image", "yolo_masks")
    FUNCTION = "detect"
    CATEGORY = "BIMO/Detection"

    def detect(self, image, yolo_model, mask_merge, conf=0.25, iou=0.45, classes="",
               device="auto", max_det=300, retina_masks=True, agnostic_nms=False):
        model_file = os.path.join(MODEL_DIRECTORY, yolo_model)
        if not os.path.isfile(model_file):
            raise FileNotFoundError(
                f"YOLO model not found: {model_file}. Put a .pt model in ComfyUI/models/yolo."
            )
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise ImportError("BimoYoloV8Detect requires ultralytics. Install it in ComfyUI's Python environment.") from error

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model = YOLO(model_file)
        class_ids = _parse_classes(classes)
        merged_masks, plotted_images, detection_masks = [], [], []

        for item in image:
            pil_image = _tensor_to_pil(item.unsqueeze(0))
            result = model(
                pil_image, conf=conf, iou=iou, classes=class_ids, device=device,
                max_det=max_det, retina_masks=retina_masks, agnostic_nms=agnostic_nms,
            )[0]
            plotted = np.asarray(result.plot())[:, :, ::-1].copy()  # Ultralytics plot is BGR.
            plotted_images.append(_pil_to_image_tensor(Image.fromarray(plotted)))

            per_image_masks = []
            if result.masks is not None and len(result.masks.data):
                for mask in result.masks.data:
                    mask_array = (mask.detach().cpu().numpy() * 255).astype(np.uint8)
                    per_image_masks.append(_pil_to_mask_tensor(Image.fromarray(mask_array)))
            elif result.boxes is not None and len(result.boxes.xyxy):
                width, height = pil_image.size
                for box in result.boxes.xyxy:
                    x1, y1, x2, y2 = box.detach().cpu().numpy().astype(int)
                    x1, x2 = sorted((max(0, x1), min(width, x2)))
                    y1, y2 = sorted((max(0, y1), min(height, y2)))
                    mask = Image.new("L", (width, height), 0)
                    if x2 > x1 and y2 > y1:
                        mask.paste(255, (x1, y1, x2, y2))
                    per_image_masks.append(_pil_to_mask_tensor(mask))

            if not per_image_masks:
                per_image_masks = [torch.zeros((1, pil_image.height, pil_image.width), dtype=torch.float32)]

            detection_masks.extend(per_image_masks)
            selected = per_image_masks if mask_merge == "all" else per_image_masks[:int(mask_merge)]
            merged_masks.append(torch.clamp(torch.stack(selected).sum(dim=0), 0.0, 1.0))

        _log(f"Processed {len(merged_masks)} image(s) using {os.path.basename(model_file)}.")
        return (torch.cat(merged_masks), torch.cat(plotted_images), torch.cat(detection_masks))


NODE_CLASS_MAPPINGS = {
    "BimoYoloV8Detect": BimoYoloV8Detect,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BimoYoloV8Detect": "BIMO YOLOv8 Detect",
}
