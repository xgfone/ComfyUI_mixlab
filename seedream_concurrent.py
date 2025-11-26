import asyncio
import io
import os
import time

import aiohttp
import numpy as np
import torch
from PIL import Image
from volcenginesdkarkruntime import Ark
from volcenginesdkarkruntime.types.images.images import SequentialImageGenerationOptions


class SeedreamImageGenerateConcurrent:
    """
    A ComfyUI node for generating images using Volcengine Seedream API with Concurrency Support
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {"multiline": True, "default": "", "placeholder": "Enter your image generation prompt here..."},
                ),
                "image1": ("IMAGE",),
                "model": (["doubao-seedream-4-0-250828"], {"default": "doubao-seedream-4-0-250828"}),
                "aspect_ratio": (
                    ["1:1", "2:3", "3:2", "4:3", "3:4", "16:9", "9:16", "21:9", "2K", "3K", "3.5K", "4K"],
                    {"default": "1:1"},
                ),
                "sequential_image_generation": (["auto", "enabled", "disabled"], {"default": "auto"}),
                # 新增并发控制参数
                "batch_size": (
                    "INT",
                    {"default": 1, "min": 1, "max": 5, "step": 1, "tooltip": "并发请求数量（同时发起多少个任务）"},
                ),
                "max_images": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 10,
                        "step": 1,
                        "label": "images_per_req",  # UI显示名称
                        "tooltip": "单次请求生成的图片数量（组图模式）",
                    },
                ),
                "response_format": (["url", "b64_json"], {"default": "url"}),
                "watermark": ("BOOLEAN", {"default": False}),
                "stream": ("BOOLEAN", {"default": False}),
                "base_url": ("STRING", {"default": "https://ark.cn-beijing.volces.com/api/v3"}),
                "use_local_images": ("BOOLEAN", {"default": True, "tooltip": "使用本地图像（Base64格式，官方支持）"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 18446744073709551615, "step": 1}),
                "enable_auto_retry": ("BOOLEAN", {"default": True, "tooltip": "启用自动重试机制"}),
            },
            "optional": {"image2": ("IMAGE",), "image3": ("IMAGE",), "image4": ("IMAGE",), "image5": ("IMAGE",)},
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "text")
    OUTPUT_IS_LIST = (False, False)  # 改回False，因为我们会把所有批次结果合并成一个大Batch
    FUNCTION = "generate_images"
    CATEGORY = "image/generation"

    def __init__(self):
        self.client = None
        self.max_retries = 3
        self.retry_delay = 1.0

    def tensor_to_pil(self, tensor):
        i = 255.0 * tensor.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        return img

    def pil_to_tensor(self, pil_image):
        img = np.array(pil_image).astype(np.float32) / 255.0
        return torch.from_numpy(img)[None,]

    def validate_input_data(self, image1, retry_count=0):
        # ... (保持原有的验证逻辑不变) ...
        max_retries = 3
        if image1 is None:
            if retry_count < max_retries:
                print(
                    f"输入验证失败 (尝试 {retry_count + 1}/{max_retries + 1}): image1 为 None，等待 {self.retry_delay} 秒后重试..."
                )
                time.sleep(self.retry_delay)
                return False, "image1_none"
            else:
                raise ValueError("image1 参数是必需的")
        if not isinstance(image1, torch.Tensor):
            if retry_count < max_retries:
                time.sleep(self.retry_delay)
                return False, "image1_type"
            else:
                raise ValueError("image1 必须是torch.Tensor类型")
        if len(image1.shape) < 3:
            if retry_count < max_retries:
                time.sleep(self.retry_delay)
                return False, "image1_shape"
            else:
                raise ValueError("image1 tensor形状无效")
        return True, "success"

    def convert_image_to_supported_format(self, pil_image, use_local_images=False):
        # ... (保持原有的转换逻辑不变) ...
        try:
            if use_local_images:
                try:
                    import base64

                    if pil_image.mode != "RGB":
                        pil_image = pil_image.convert("RGB")
                    buffered = io.BytesIO()
                    pil_image.save(buffered, format="PNG")
                    img_bytes = buffered.getvalue()
                    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
                    data_url = f"data:image/png;base64,{img_base64}"
                    return data_url
                except Exception:
                    return self._get_example_image_url()
            return self._get_example_image_url()
        except Exception:
            return self._get_example_image_url()

    def _get_example_image_url(self):
        example_urls = [
            "https://ark-project.tos-cn-beijing.volces.com/doc_image/seedream4_imagesToimages_1.png",
            "https://ark-project.tos-cn-beijing.volces.com/doc_image/seedream4_imagesToimages_2.png",
        ]
        import random

        return random.choice(example_urls)

    def aspect_ratio_to_size(self, aspect_ratio):
        ratio_map = {
            "1:1": "2048x2048",
            "4:3": "2304x1728",
            "3:4": "1728x2304",
            "16:9": "2560x1440",
            "9:16": "1440x2560",
            "3:2": "2496x1664",
            "2:3": "1664x2496",
            "21:9": "3024x1296",
            "2K": "2K",
            "3K": "2133x3200",
            "3.5K": "2933x4400",
            "4K": "4K",
        }
        return ratio_map.get(aspect_ratio, "2048x2048")

    # 新增异步下载函数
    async def _download_image_async(self, session, url):
        try:
            async with session.get(url) as response:
                response.raise_for_status()
                data = await response.read()
                image = Image.open(io.BytesIO(data))
                if image.mode != "RGB":
                    image = image.convert("RGB")
                return self.pil_to_tensor(image)
        except Exception as e:
            print(f"下载失败: {e}")
            placeholder = Image.new("RGB", (512, 512), color="black")
            return self.pil_to_tensor(placeholder)

    def initialize_client(self, base_url):
        api_key = os.environ.get("ARK_API_KEY")
        if not api_key:
            raise ValueError("API Key is required. Please set ARK_API_KEY environment variable.")
        self.client = Ark(base_url=base_url, api_key=api_key.strip())

    # 改为异步入口函数
    async def generate_images(
        self,
        prompt,
        image1,
        model,
        aspect_ratio,
        sequential_image_generation,
        batch_size,
        max_images,
        response_format,
        watermark,
        stream,
        base_url,
        use_local_images,
        seed,
        enable_auto_retry,
        image2=None,
        image3=None,
        image4=None,
        image5=None,
    ):
        # 验证逻辑 (保持同步)
        max_attempts = self.max_retries + 1 if enable_auto_retry else 1
        validation_passed = False
        for retry_count in range(max_attempts):
            try:
                is_valid, _ = self.validate_input_data(image1, retry_count)
                if is_valid:
                    validation_passed = True
                    break
            except Exception:
                if retry_count == max_attempts - 1:
                    raise
                time.sleep(self.retry_delay)

        if not validation_passed:
            raise ValueError("输入验证失败")

        # 初始化客户端
        self.initialize_client(base_url)

        # 准备输入图像 (预处理，避免在异步循环中重复处理)
        input_images = [img for img in [image1, image2, image3, image4, image5] if img is not None]
        image_urls = []
        for img_tensor in input_images:
            pil_img = self.tensor_to_pil(img_tensor.squeeze(0))
            url = self.convert_image_to_supported_format(pil_img, use_local_images)
            image_urls.append(url)

        if not image_urls:
            image_urls = ["https://ark-project.tos-cn-beijing.volces.com/doc_image/seedream4_imagesToimages_1.png"]

        size = self.aspect_ratio_to_size(aspect_ratio)
        generation_options = SequentialImageGenerationOptions(max_images=max_images)

        # 定义单个任务的异步函数
        async def process_single_batch(task_index):
            current_seed = seed + task_index
            # 映射 Seed 防止溢出
            normalized_seed = current_seed if current_seed <= 2147483647 else current_seed % 2147483647

            task_log = []
            task_tensors = []

            try:
                # 使用 asyncio.to_thread 在线程池中运行同步 SDK 调用，防止阻塞
                # 注意：Ark SDK 目前不支持 seed 参数，但我们逻辑上使用它来区分任务
                print(f"🚀 启动任务 {task_index + 1}/{batch_size} (Seed: {normalized_seed})")

                loop = asyncio.get_running_loop()
                images_response = await loop.run_in_executor(
                    None,
                    lambda: self.client.images.generate(
                        model=model,
                        prompt=prompt,
                        image=image_urls,
                        size=size,
                        sequential_image_generation=sequential_image_generation,
                        sequential_image_generation_options=generation_options,
                        response_format=response_format,
                        watermark=watermark,
                        stream=stream,
                    ),
                )

                # 处理结果
                task_log.append(f"✅ 任务 {task_index + 1} 完成，生成 {len(images_response.data)} 张图")

                # 异步下载图片
                async with aiohttp.ClientSession() as session:
                    if response_format == "url":
                        download_tasks = [
                            self._download_image_async(session, item.url) for item in images_response.data
                        ]
                        task_tensors = await asyncio.gather(*download_tasks)
                    else:
                        # 处理 b64_json
                        import base64

                        for item in images_response.data:
                            image_bytes = base64.b64decode(item.b64_json)
                            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                            task_tensors.append(self.pil_to_tensor(image))

                return task_tensors, "\n".join(task_log)

            except Exception as e:
                error_msg = f"❌ 任务 {task_index + 1} 失败: {str(e)}"
                print(error_msg)
                # 返回红色占位图
                error_img = self.pil_to_tensor(Image.new("RGB", (512, 512), color="red"))
                return [error_img], error_msg

        # 并发执行所有任务
        tasks = [process_single_batch(i) for i in range(batch_size)]
        results = await asyncio.gather(*tasks)

        # 汇总结果
        all_tensors = []
        all_logs = [f"📊 并发报告: 总任务数 {batch_size}\n"]

        for tensors, log in results:
            all_tensors.extend(tensors)
            all_logs.append(log)

        # 最终合并 Tensor
        if not all_tensors:
            final_tensor = self.pil_to_tensor(Image.new("RGB", (512, 512), color="black"))
        else:
            final_tensor = torch.cat(all_tensors, dim=0)

        return (final_tensor, "\n".join(all_logs))
