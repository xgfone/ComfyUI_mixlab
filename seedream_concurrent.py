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
    A ComfyUI node for generating images using Volcengine Seedream API
    Features: Concurrency, Timeout, Smart Filtering, and Auto-Retry
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
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 5, "step": 1, "tooltip": "并发请求数量"}),
                "max_images": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 10,
                        "step": 1,
                        "label": "images_per_req",
                        "tooltip": "单次请求生成的图片数量",
                    },
                ),
                "response_format": (["url", "b64_json"], {"default": "url"}),
                "watermark": ("BOOLEAN", {"default": False}),
                "stream": ("BOOLEAN", {"default": False}),
                "base_url": ("STRING", {"default": "https://ark.cn-beijing.volces.com/api/v3"}),
                "use_local_images": ("BOOLEAN", {"default": True, "tooltip": "使用本地图像（Base64格式）"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 18446744073709551615, "step": 1}),
                "enable_auto_retry": ("BOOLEAN", {"default": True, "tooltip": "启用输入验证的自动重试"}),
                "max_retries": ("INT", {"default": 0, "min": 0, "max": 5, "step": 1, "tooltip": "最大重试的次数。"}),
                "timeout": ("INT", {"default": 70, "min": 10, "max": 300, "step": 1, "tooltip": "最大等待时间(秒)。"}),
            },
            "optional": {"image2": ("IMAGE",), "image3": ("IMAGE",), "image4": ("IMAGE",), "image5": ("IMAGE",)},
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "text")
    OUTPUT_IS_LIST = (False, False)
    FUNCTION = "generate_images"
    CATEGORY = "image/generation"

    def __init__(self):
        self.client = None
        self.input_validation_retries = 3
        self.retry_delay = 1.0

    def tensor_to_pil(self, tensor):
        i = 255.0 * tensor.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        return img

    def pil_to_tensor(self, pil_image):
        img = np.array(pil_image).astype(np.float32) / 255.0
        return torch.from_numpy(img)[None,]

    def validate_input_data(self, image1, retry_count=0):
        max_retries = 3
        if image1 is None:
            if retry_count < max_retries:
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
                    return f"data:image/png;base64,{img_base64}"
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
            print(f"下载图片失败: {e}")
            return None  # 下载失败返回 None，不返回黑色占位符

    def initialize_client(self, base_url):
        api_key = os.environ.get("ARK_API_KEY")
        if not api_key:
            raise ValueError("API Key is required. Please set ARK_API_KEY environment variable.")
        self.client = Ark(base_url=base_url, api_key=api_key.strip())

    async def generate_images(
        self,
        prompt,
        image1,
        model,
        aspect_ratio,
        sequential_image_generation,
        batch_size,
        max_images,
        timeout,
        max_retries,
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
        # --- 1. 输入验证 ---
        max_attempts = self.input_validation_retries + 1 if enable_auto_retry else 1
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

        self.initialize_client(base_url)

        # --- 2. 准备输入图像 ---
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

        # --- 3. 定义单个任务逻辑 ---
        async def process_single_batch(task_index, current_try_seed):
            # 确保 Seed 不溢出
            normalized_seed = current_try_seed if current_try_seed <= 2147483647 else current_try_seed % 2147483647

            task_log = []
            task_tensors = []

            try:
                print(f"🚀 启动任务 {task_index + 1}/{batch_size} (Seed: {normalized_seed})")
                loop = asyncio.get_running_loop()

                # API 调用
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
                        # 注意：Seedream API 目前可能不支持直接传 seed，但我们在逻辑上区分了任务
                    ),
                )

                task_log.append(f"✅ 任务 {task_index + 1} 成功，API返回 {len(images_response.data)} 张图")

                # 下载图片
                async with aiohttp.ClientSession() as session:
                    if response_format == "url":
                        download_tasks = [
                            self._download_image_async(session, item.url) for item in images_response.data
                        ]
                        downloaded_results = await asyncio.gather(*download_tasks)
                        # 过滤下载失败的 None
                        task_tensors = [t for t in downloaded_results if t is not None]
                    else:
                        import base64

                        for item in images_response.data:
                            try:
                                image_bytes = base64.b64decode(item.b64_json)
                                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                                task_tensors.append(self.pil_to_tensor(image))
                            except Exception as e:
                                print(f"Base64解码失败: {e}")

                return task_tensors, "\n".join(task_log)

            except asyncio.CancelledError:
                raise  # 必须抛出
            except Exception as e:
                error_msg = f"❌ 任务 {task_index + 1} 失败: {str(e)}"
                print(error_msg)
                # 失败时返回 None，不返回错误图片
                return None, error_msg

        # --- 4. 带有重试机制的主循环 ---

        all_logs = []
        final_valid_tensors = []

        # 总尝试次数 = 1 (首次) + 重试次数
        total_attempts = 1 + max_retries

        for attempt in range(total_attempts):
            is_retry = attempt > 0
            if is_retry:
                retry_msg = f"\n🔄 第 {attempt} 次重试 (共 {max_retries} 次)..."
                print(retry_msg)
                all_logs.append(retry_msg)
                # 稍微改变一下 seed，防止因特定 seed 导致的失败
                current_batch_seed = seed + (attempt * 100)
            else:
                current_batch_seed = seed

            # 创建任务列表
            tasks = [asyncio.create_task(process_single_batch(i, current_batch_seed + i)) for i in range(batch_size)]

            print(f"⏳ [第{attempt + 1}轮] 开始并发执行，超时设定: {timeout}秒...")

            # 等待结果
            done, pending = await asyncio.wait(tasks, timeout=timeout)

            # 取消超时任务
            if pending:
                timeout_msg = f"⚠️ [第{attempt + 1}轮] {len(pending)} 个任务超时被取消。"
                print(timeout_msg)
                all_logs.append(timeout_msg)
                for task in pending:
                    task.cancel()

            # 收集本轮结果
            batch_tensors = []
            for task in done:
                try:
                    result = task.result()
                    if result is not None:
                        tensors, log = result
                        if tensors:  # 确保 tensors 列表不为空
                            batch_tensors.extend(tensors)
                        all_logs.append(log)
                    else:
                        # 任务内部捕获了异常并返回 None
                        pass
                except Exception as e:
                    all_logs.append(f"❌ 任务异常: {str(e)}")

            # 检查本轮是否成功
            if len(batch_tensors) > 0:
                final_valid_tensors = batch_tensors
                success_msg = f"✅ [第{attempt + 1}轮] 成功获取 {len(final_valid_tensors)} 张图片。"
                print(success_msg)
                all_logs.append(success_msg)
                break  # 成功则跳出重试循环
            else:
                fail_msg = f"❌ [第{attempt + 1}轮] 未获取任何有效图片。"
                print(fail_msg)
                all_logs.append(fail_msg)
                if attempt < total_attempts - 1:
                    await asyncio.sleep(2)  # 重试前等待2秒

        # --- 5. 最终结果处理 ---

        if not final_valid_tensors:
            err_final = "⚠️ 所有尝试（包括重试）均已失败，未生成有效图片。返回黑色占位图。"
            print(err_final)
            all_logs.append(err_final)
            final_tensor = self.pil_to_tensor(Image.new("RGB", (512, 512), color="black"))
        else:
            # 只要有图，就只返回成功的图
            final_tensor = torch.cat(final_valid_tensors, dim=0)

        return (final_tensor, "\n".join(all_logs))
