# -*- coding: utf-8 -*-
import io

import numpy as np
import requests
import torch
from PIL import Image

# 尝试导入阿里云 SDK
try:
    from alibabacloud_facebody20191230.client import Client
    from alibabacloud_facebody20191230.models import FaceBeautyAdvanceRequest, FaceBeautyRequest
    from alibabacloud_tea_openapi.models import Config
    from alibabacloud_tea_util.models import RuntimeOptions

    ALIYUN_SDK_AVAILABLE = True
except ImportError as e:
    ALIYUN_SDK_AVAILABLE = False
    print(f"Warning: Aliyun SDK load failed. Error details: {e}")


class AliyunFaceBeautyNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "access_key_id": ("STRING", {"multiline": False, "default": ""}),
                "access_key_secret": ("STRING", {"multiline": False, "default": ""}),
                "sharp": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.1}),  # 锐化
                "smooth": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.1}),  # 磨皮
                "white": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.1}),  # 美白
            },
            "optional": {
                "image": ("IMAGE",),  # ComfyUI 图片输入
                "image_url": ("STRING", {"multiline": False, "default": ""}),  # URL 输入
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "beautify"
    CATEGORY = "Aliyun/FaceBody"

    def create_client(self, access_key_id, access_key_secret, region_id="cn-shanghai"):
        config = Config(access_key_id=access_key_id, access_key_secret=access_key_secret)
        config.endpoint = f"facebody.{region_id}.aliyuncs.com"
        return Client(config)

    def tensor_to_bytes(self, image_tensor):
        # ComfyUI tensor shape: [H, W, C] (after squeezing batch) -> numpy
        i = 255.0 * image_tensor.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

        # Convert to bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)
        return img_byte_arr

    def url_to_image_tensor(self, image_url):
        response = requests.get(image_url)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content))

        # Convert PIL to Tensor [1, H, W, C]
        img = img.convert("RGB")
        image_np = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(image_np).unsqueeze(0)

    def beautify(self, access_key_id, access_key_secret, sharp, smooth, white, image=None, image_url=None):
        if not ALIYUN_SDK_AVAILABLE:
            raise ImportError("Aliyun SDK is not installed.")

        if not access_key_id or not access_key_secret:
            raise ValueError("Access Key ID and Secret are required.")

        client = self.create_client(access_key_id, access_key_secret)
        runtime = RuntimeOptions()

        output_tensors = []

        # 优先处理 ComfyUI 传入的 Tensor 图片
        if image is not None:
            # image 是 [B, H, W, C]
            for i in range(image.shape[0]):
                current_img = image[i]
                img_stream = self.tensor_to_bytes(current_img)

                # 使用 AdvanceRequest 上传本地流
                request = FaceBeautyAdvanceRequest()
                request.image_urlobject = img_stream
                request.sharp = sharp
                request.smooth = smooth
                request.white = white

                try:
                    # 调用阿里云接口
                    response = client.face_beauty_advance(request, runtime)
                    result_url = response.body.data.image_url
                    print(f"Aliyun Processed URL: {result_url}")

                    # 下载结果并转换为 Tensor
                    output_tensors.append(self.url_to_image_tensor(result_url))

                except Exception as e:
                    print(f"Error processing image index {i}: {e}")
                    # 如果失败，为了防止中断，可以选择返回原图或报错，这里选择报错
                    raise e

        # 如果没有 Tensor 输入，但有 URL
        elif image_url and image_url.strip() != "":
            request = FaceBeautyRequest()
            request.image_url = image_url
            request.sharp = sharp
            request.smooth = smooth
            request.white = white

            try:
                response = client.face_beauty(request, runtime)
                result_url = response.body.data.image_url
                print(f"Aliyun Processed URL: {result_url}")
                output_tensors.append(self.url_to_image_tensor(result_url))
            except Exception as e:
                print(f"Error processing URL {image_url}: {e}")
                raise e

        else:
            raise ValueError("No input provided. Please connect an IMAGE or provide an image_url.")

        if not output_tensors:
            raise ValueError("Processing failed, no images returned.")

        # 合并 Batch
        final_tensor = torch.cat(output_tensors, dim=0)
        return (final_tensor,)
