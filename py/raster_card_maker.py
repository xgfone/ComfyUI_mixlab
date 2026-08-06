# import cv2
import numpy as np
import torch
from PIL import Image


class RasterCardMaker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),  #
                "image2": ("IMAGE",),  #
                "width": ("INT", {"default": 8}),  #
                "height": ("INT", {"default": 8}),  #
                "LPI": ("INT", {"default": 50}),  #
                "DPI": ("INT", {"default": 300}),  #
                "direction": (["horizontal", "vertical"],),
                "position": (
                    [
                        "top-left",
                        "top-center",
                        "top-right",
                        "middle-left",
                        "center",
                        "middle-right",
                        "bottom-left",
                        "bottom-center",
                        "bottom-right",
                    ],
                    {"default": "center"},
                ),
                "offset_x": ("INT", {"default": 0}),
                "offset_y": ("INT", {"default": 0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("rastrt_image",)
    FUNCTION = "raster_blend_by_lpi_on_a4"
    CATEGORY = "Conchshell Image Analysis"

    def raster_blend_by_lpi_on_a4(
        self, image1, image2, width, height, LPI, DPI, direction, position, offset_x, offset_y
    ):
        def cm_to_pixel(cm, DPI=300):
            return int(cm / 2.54 * DPI)

        # def tensor2pil(t_image: torch.Tensor) -> Image.Image:
        #     return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

        def pil2tensor(image: Image.Image) -> torch.Tensor:
            return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

        # def pil_from_tensor_resized(t_img, target_size):
        #     np_img = (t_img[0].cpu().numpy() * 255).astype(np.uint8)
        #     resized = cv2.resize(np_img, target_size, interpolation=cv2.INTER_LINEAR)
        #     return Image.fromarray(resized)

        def calculate_offset(position, canvas_width, canvas_height, image_width, image_height):
            positions = {
                "top-left": (0, 0),
                "top-center": ((canvas_width - image_width) // 2, 0),
                "top-right": (canvas_width - image_width, 0),
                "middle-left": (0, (canvas_height - image_height) // 2),
                "center": ((canvas_width - image_width) // 2, (canvas_height - image_height) // 2),
                "middle-right": (canvas_width - image_width, (canvas_height - image_height) // 2),
                "bottom-left": (0, canvas_height - image_height),
                "bottom-center": ((canvas_width - image_width) // 2, canvas_height - image_height),
                "bottom-right": (canvas_width - image_width, canvas_height - image_height),
            }
            return positions.get(position, (0, 0))

        # 计算像素尺寸
        raster_width_px = cm_to_pixel(width, DPI)
        raster_height_px = cm_to_pixel(height, DPI)
        a4_width_px = cm_to_pixel(10.16, DPI)
        a4_height_px = cm_to_pixel(15.24, DPI)

        # 条纹宽度（像素）
        stripe_px = DPI // LPI // 2
        print(stripe_px)
        # img1 = Image.open(image1).convert("RGB").resize((raster_width_px, raster_height_px))
        # img2 = Image.open(image2).convert("RGB").resize((raster_width_px, raster_height_px))
        # img1 = tensor2pil(image1)
        # img2 = tensor2pil(image2)
        image1 = image1.cpu().numpy()
        image2 = image2.cpu().numpy()

        # Convert to PIL
        img1 = Image.fromarray((image1[0] * 255).astype(np.uint8))
        img1 = img1.resize((raster_width_px, raster_height_px))
        img2 = Image.fromarray((image2[0] * 255).astype(np.uint8))
        img2 = img2.resize((raster_width_px, raster_height_px))
        # img1 = pil_from_tensor_resized(image1,(raster_height_px, raster_width_px))
        # img2 = pil_from_tensor_resized(image2,(raster_height_px, raster_width_px))
        #  # Resize tensor first（快）
        # image1_resized = F.interpolate(image1, size=(raster_height_px, raster_width_px), mode='bilinear', align_corners=False)
        # image2_resized = F.interpolate(image2, size=(raster_height_px, raster_width_px), mode='bilinear', align_corners=False)

        # # Convert to PIL（快）
        # img1 = Image.fromarray((image1_resized[0] * 255).byte().cpu().numpy())
        # img2 = Image.fromarray((image2_resized[0] * 255).byte().cpu().numpy())

        arr1 = np.array(img1)
        arr2 = np.array(img2)
        output = np.zeros_like(arr1)

        if direction == "vertical":
            # 竖纹：按列交错
            for x in range(0, raster_width_px, stripe_px * 2):
                output[:, x : x + stripe_px] = arr1[:, x : x + stripe_px]
                output[:, x + stripe_px : x + 2 * stripe_px] = arr2[:, x + stripe_px : x + 2 * stripe_px]
        elif direction == "horizontal":
            # 横纹：按行交错
            for y in range(0, raster_height_px, stripe_px * 2):
                output[y : y + stripe_px, :] = arr1[y : y + stripe_px, :]
                output[y + stripe_px : y + 2 * stripe_px, :] = arr2[y + stripe_px : y + 2 * stripe_px, :]
        else:
            raise ValueError("direction must be [vertical] or [horizontal]")

        # 创建光栅图像
        raster_img = Image.fromarray(output)

        # 创建A4画布并粘贴到中间
        a4_img = Image.new("RGB", (a4_width_px, a4_height_px), color="white")
        # offset_x = (a4_width_px - raster_width_px) // 2
        # offset_y = (a4_height_px - raster_height_px) // 2
        base_offset_x, base_offset_y = calculate_offset(
            position, a4_width_px, a4_height_px, raster_width_px, raster_height_px
        )
        final_offset_x = base_offset_x + offset_x
        final_offset_y = base_offset_y + offset_y

        a4_img.paste(raster_img, (final_offset_x, final_offset_y))

        # a4_img.paste(raster_img, (offset_x, offset_y))
        result = pil2tensor(a4_img)

        return (result,)
