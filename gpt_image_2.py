import io
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests
import torch
from PIL import Image


class GPTImage2Generator:
    """
    A ComfyUI node for editing/composing images using OPENAI gpt-image-2 API.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "Enter your image generation prompt here...",
                    },
                ),
                "image1": ("IMAGE",),
                "model": (["gpt-image-2"], {"default": "gpt-image-2"}),
                "aspect_ratio": (
                    ["auto", "1:1", "2:3", "3:2", "4:3", "3:4", "16:9", "9:16", "21:9", "2K", "4K"],
                    {"default": "1:1"},
                ),
                "quality": (["low", "medium", "high"], {"default": "high"}),
                "max_images": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "output_format": (["png", "jpeg"], {"default": "png"}),
                "output_compression": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),
                "base_url": ("STRING", {"default": "https://api.openai.com/v1"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 18446744073709551615, "step": 1}),
                "enable_auto_retry": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "启用自动重试机制，处理云端工作流的异步执行问题"},
                ),
                "timeout": (
                    "INT",
                    {
                        "default": 70,
                        "min": 10,
                        "max": 300,
                        "step": 1,
                        "tooltip": "最大等待时间(秒)。",
                    },
                ),
            },
            "optional": {
                # ========= 任务1额外参考图 =========
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                # ========= 任务2 =========
                "prompt2": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "Task 2 prompt（留空则不执行）",
                    },
                ),
                "task2_image1": ("IMAGE",),
                "task2_image2": ("IMAGE",),
                "task2_image3": ("IMAGE",),
                # ========= 任务3 =========
                "prompt3": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "Task 3 prompt（留空则不执行）",
                    },
                ),
                "task3_image1": ("IMAGE",),
                "task3_image2": ("IMAGE",),
                "task3_image3": ("IMAGE",),
                # ========= 任务4 =========
                "prompt4": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "Task 4 prompt（留空则不执行）",
                    },
                ),
                "task4_image1": ("IMAGE",),
                "task4_image2": ("IMAGE",),
                "task4_image3": ("IMAGE",),
                # ========= 任务5 =========
                "prompt5": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "Task 5 prompt（留空则不执行）",
                    },
                ),
                "task5_image1": ("IMAGE",),
                "task5_image2": ("IMAGE",),
                "task5_image3": ("IMAGE",),
                # ========= 任务6 =========
                "prompt6": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "Task 6 prompt（留空则不执行）",
                    },
                ),
                "task6_image1": ("IMAGE",),
                "task6_image2": ("IMAGE",),
                "task6_image3": ("IMAGE",),
                # ========= 失败容忍 =========
                "ignore_failure": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 6,
                        "step": 1,
                        "tooltip": "最多允许失败的任务数。失败数 <= ignore_failure 时忽略失败任务；失败数 > ignore_failure 时输出正常生成图片，并为每个失败任务补 1 张占位图。",
                    },
                ),
                "task2_image4": ("IMAGE",),
                "task2_image5": ("IMAGE",),
                "task2_image_add": ("IMAGE",),
                "task3_image4": ("IMAGE",),
                "task3_image5": ("IMAGE",),
                "task3_image_add": ("IMAGE",),
                "task4_image4": ("IMAGE",),
                "task4_image5": ("IMAGE",),
                "task4_image_add": ("IMAGE",),
                "task5_image4": ("IMAGE",),
                "task5_image5": ("IMAGE",),
                "task5_image_add": ("IMAGE",),
                "task6_image4": ("IMAGE",),
                "task6_image5": ("IMAGE",),
                "task6_image_add": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "text")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "generate_images"
    CATEGORY = "image/generation"

    def __init__(self):
        self.max_retries = 1
        self.retry_delay = 1.0  # 秒

    def tensor_to_pil(self, tensor):
        """Convert ComfyUI tensor to PIL Image"""
        # Convert tensor to numpy array
        i = 255.0 * tensor.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        return img

    def pil_to_tensor(self, pil_image):
        """Convert PIL Image to ComfyUI tensor"""
        img = np.array(pil_image).astype(np.float32) / 255.0
        return torch.from_numpy(img)[None,]

    def make_placeholder_tensor(self, color="red"):
        """Create a single placeholder tensor used only when failures exceed ignore_failure."""
        placeholder = Image.new("RGB", (512, 512), color=color)
        return self.pil_to_tensor(placeholder)

    def is_timeout_error(self, error):
        """Best-effort timeout detection for requests / SDK / builtin timeout exceptions."""
        if isinstance(error, (TimeoutError, requests.exceptions.Timeout)):
            return True

        error_text = str(error).lower()
        timeout_keywords = [
            "timeout",
            "timed out",
            "read timed out",
            "connect timeout",
            "request timed out",
            "deadline exceeded",
            "504",
            "gateway timeout",
        ]
        return any(keyword in error_text for keyword in timeout_keywords)

    def validate_input_data(self, image1, retry_count=0):
        """
        验证输入数据的完整性，支持重试机制处理云端工作流的异步特性
        """
        max_retries = 1

        # 基本验证
        if image1 is None:
            if retry_count < max_retries:
                print(
                    f"输入验证失败 (尝试 {retry_count + 1}/{max_retries + 1}): image1 为 None，等待 {self.retry_delay} 秒后重试..."
                )
                time.sleep(self.retry_delay)
                return False, "image1_none"
            else:
                raise ValueError("image1 参数是必需的，请确保上游节点已正确连接并执行完成")

        # 检查tensor类型
        if not isinstance(image1, torch.Tensor):
            if retry_count < max_retries:
                print(
                    f"输入验证失败 (尝试 {retry_count + 1}/{max_retries + 1}): image1 类型错误 {type(image1)}，等待 {self.retry_delay} 秒后重试..."
                )
                time.sleep(self.retry_delay)
                return False, "image1_type"
            else:
                raise ValueError(f"image1 必须是torch.Tensor类型，当前类型: {type(image1)}")

        # 检查tensor形状
        if len(image1.shape) < 3:
            if retry_count < max_retries:
                print(
                    f"输入验证失败 (尝试 {retry_count + 1}/{max_retries + 1}): image1 形状无效 {image1.shape}，等待 {self.retry_delay} 秒后重试..."
                )
                time.sleep(self.retry_delay)
                return False, "image1_shape"
            else:
                raise ValueError(f"image1 tensor形状无效: {image1.shape}，期望至少3维")

        # 检查tensor数据质量 - 避免全零或无效数据
        if torch.all(image1 == 0) or torch.isnan(image1).any():
            if retry_count < max_retries:
                print(
                    f"输入验证失败 (尝试 {retry_count + 1}/{max_retries + 1}): image1 数据质量问题（全零或包含NaN），等待 {self.retry_delay} 秒后重试..."
                )
                time.sleep(self.retry_delay)
                return False, "image1_quality"
            else:
                print("警告: image1 包含异常数据，但将继续执行...")

        print(f"✅ 输入验证通过: image1 形状 {image1.shape}, 数据类型 {image1.dtype}")
        return True, "success"

    def pil_to_png_bytes(self, pil_image):
        """Convert a PIL image to the multipart PNG bytes required by /images/edits."""
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        return buffered.getvalue()

    def aspect_ratio_to_size(self, aspect_ratio):
        """Convert aspect ratio to size parameter"""
        ratio_map = {
            "auto": "auto",
            "1:1": "2048x2048",
            "4:3": "2304x1728",
            "3:4": "1728x2304",
            "16:9": "2560x1440",
            "9:16": "1440x2560",
            "3:2": "2496x1664",
            "2:3": "1664x2496",
            "21:9": "3024x1296",
            "2K": "2048x2048",
            "4K": "3840x2160",
        }

        return ratio_map.get(aspect_ratio, aspect_ratio)

    def download_image_from_url(self, url):
        """Download image from URL and convert to tensor"""
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self.pil_to_tensor(image)

    def get_api_key(self):
        """Read the OPENAI API key from the environment."""
        api_key = os.environ.get("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("API Key is required. Please set OPENAI_API_KEY environment variable.")

        return api_key.strip()

    def generate_images(
        self,
        prompt,
        image1,
        model,
        aspect_ratio,
        quality,
        max_images,
        output_format,
        output_compression,
        base_url,
        seed,
        enable_auto_retry,
        timeout,
        image2=None,
        image3=None,
        image4=None,
        image5=None,
        image6=None,
        prompt2="",
        task2_image1=None,
        task2_image2=None,
        task2_image3=None,
        prompt3="",
        task3_image1=None,
        task3_image2=None,
        task3_image3=None,
        prompt4="",
        task4_image1=None,
        task4_image2=None,
        task4_image3=None,
        prompt5="",
        task5_image1=None,
        task5_image2=None,
        task5_image3=None,
        prompt6="",
        task6_image1=None,
        task6_image2=None,
        task6_image3=None,
        ignore_failure=0,
        task2_image4=None,
        task2_image5=None,
        task2_image_add=None,
        task3_image4=None,
        task3_image5=None,
        task3_image_add=None,
        task4_image4=None,
        task4_image5=None,
        task4_image_add=None,
        task5_image4=None,
        task5_image5=None,
        task5_image_add=None,
        task6_image4=None,
        task6_image5=None,
        task6_image_add=None,
    ):
        """
        支持最多 6 个任务的并行执行：
        - 任务1：prompt + image1(+image2~image6)
        - 任务2~6：promptX + taskX_image1~3
        使用 ThreadPoolExecutor 并行调用 gpt-image-2 接口。
        """

        # ========== 1. 组装任务列表 ==========
        tasks = []

        # 任务1：必跑（因为 image1 / prompt 是必填）
        images = [image1, image4, image2, image3, image5, image6]
        task1_images = [img for img in images if img is not None and img.shape[1] >= 14]
        tasks.append(
            {
                "index": 1,
                "prompt": prompt,
                "images": task1_images,
            }
        )

        # 任务2~6：如果 promptX 非空且至少有一张图，就加入任务
        def add_task_if_valid(idx, p, img1, img2, img3, img4, img5, img6):
            if p is None:
                p = ""
            p = p.strip()
            imgs = [
                img
                for img in [img1, img2, img3, img4, img5, img6]
                if img is not None and img.shape[1] >= 14
            ]
            if p != "" and len(imgs) > 0:
                tasks.append(
                    {
                        "index": idx,
                        "prompt": p,
                        "images": imgs,
                    }
                )

        add_task_if_valid(
            2,
            prompt2,
            task2_image1,
            task2_image_add,
            task2_image2,
            task2_image3,
            task2_image4,
            task2_image5,
        )
        add_task_if_valid(
            3,
            prompt3,
            task3_image1,
            task3_image_add,
            task3_image2,
            task3_image3,
            task3_image4,
            task3_image5,
        )
        add_task_if_valid(
            4,
            prompt4,
            task4_image1,
            task4_image_add,
            task4_image2,
            task4_image3,
            task4_image4,
            task4_image5,
        )
        add_task_if_valid(
            5,
            prompt5,
            task5_image1,
            task5_image_add,
            task5_image2,
            task5_image3,
            task5_image4,
            task5_image5,
        )
        add_task_if_valid(
            6,
            prompt6,
            task6_image1,
            task6_image_add,
            task6_image2,
            task6_image3,
            task6_image4,
            task6_image5,
        )

        if not tasks:
            raise ValueError("没有可执行的任务，请至少提供任务1的 prompt 和 image1。")

        # ========== 2. 失败容忍参数标准化 ==========
        try:
            ignore_failure = int(ignore_failure)
        except Exception:
            ignore_failure = 0
        ignore_failure = max(0, min(ignore_failure, len(tasks)))

        # ========== 3. 定义单任务执行函数（在线程里跑） ==========
        def run_single_task(task, attempt_no=1):
            t_idx = task["index"]
            t_prompt = task["prompt"]
            t_images = task["images"]

            # 最多只取前 6 张，兼容 _execute_generation 的 image1~6 接口
            t_image1 = t_images[0] if len(t_images) > 0 else None
            t_image2 = t_images[1] if len(t_images) > 1 else None
            t_image3 = t_images[2] if len(t_images) > 2 else None
            t_image4 = t_images[3] if len(t_images) > 3 else None
            t_image5 = t_images[4] if len(t_images) > 4 else None
            t_image6 = t_images[5] if len(t_images) > 5 else None

            try:
                # 输入校验只做一次；失败交给并发层统计，不在这里返回占位图。
                is_valid, _ = self.validate_input_data(t_image1, retry_count=0)
                if not is_valid:
                    self.validate_input_data(t_image1, retry_count=self.max_retries)

                print(f"🚀 开始执行图像生成 - 任务 {t_idx} (attempt={attempt_no})")

                output_tensors, text_output = self._execute_generation(
                    t_prompt,
                    t_image1,
                    model,
                    aspect_ratio,
                    quality,
                    max_images,
                    output_format,
                    output_compression,
                    base_url,
                    seed,
                    enable_auto_retry,
                    timeout,
                    t_image2,
                    t_image3,
                    t_image4,
                    t_image5,
                    t_image6,
                )

                return {
                    "index": t_idx,
                    "ok": True,
                    "images": output_tensors,
                    "text": text_output,
                    "attempt": attempt_no,
                }

            except Exception as e:
                is_timeout = self.is_timeout_error(e)
                print(
                    f"任务 {t_idx} 执行失败 (attempt={attempt_no}, timeout={is_timeout}): {str(e)}"
                )
                return {
                    "index": t_idx,
                    "ok": False,
                    "images": [],
                    "text": str(e),
                    "error": e,
                    "is_timeout": is_timeout,
                    "attempt": attempt_no,
                    "task": task,
                }

        def run_tasks_parallel(task_list, attempt_no=1):
            results = []
            max_workers = min(len(task_list), 6)
            if max_workers <= 0:
                return results

            print(
                f"🔧 并行执行 gpt-image-2 任务数: {len(task_list)} (max_workers={max_workers}, attempt={attempt_no})"
            )

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(run_single_task, task, attempt_no): task["index"]
                    for task in task_list
                }
                for future in as_completed(futures):
                    results.append(future.result())
            return results

        # ========== 4. 首轮并行执行：任何失败都不在单任务内返回占位图 ==========
        all_results = run_tasks_parallel(tasks, attempt_no=1)

        # ========== 5. 仅当“超时失败任务数 > ignore_failure”时，才重试这些超时任务 ==========
        # 注意：普通失败不会自动重试；这是为了避免超大图、参数错误等确定性失败被反复请求。
        if enable_auto_retry and self.max_retries > 0:
            for retry_round in range(self.max_retries):
                timeout_failures = [
                    r for r in all_results if (not r.get("ok")) and r.get("is_timeout")
                ]
                if len(timeout_failures) <= ignore_failure:
                    break

                print(
                    f"⏱️ 超时失败任务数 {len(timeout_failures)} > ignore_failure {ignore_failure}，"
                    f"开始第 {retry_round + 1}/{self.max_retries} 轮超时任务重试..."
                )
                time.sleep(self.retry_delay)

                retry_tasks = [r["task"] for r in timeout_failures]
                retry_results = run_tasks_parallel(retry_tasks, attempt_no=retry_round + 2)

                retry_by_index = {r["index"]: r for r in retry_results}
                new_results = []
                for r in all_results:
                    replacement = retry_by_index.get(r["index"])
                    new_results.append(replacement if replacement is not None else r)
                all_results = new_results

        success_results = [r for r in all_results if r.get("ok")]
        failed_results = [r for r in all_results if not r.get("ok")]
        timeout_failed_results = [r for r in failed_results if r.get("is_timeout")]

        # ========== 6. 按最终失败数决定输出策略 ==========
        # - 失败数 <= ignore_failure：容忍失败，仅输出成功图片，不补占位图。
        # - 失败数 > ignore_failure：保留所有成功图片，并按失败任务数补同等数量的占位图。
        success_results.sort(key=lambda r: r["index"])
        failed_results.sort(key=lambda r: r["index"])

        should_add_placeholders = len(failed_results) > ignore_failure

        if not success_results and not should_add_placeholders:
            # 所有任务都失败，但失败数没有超过 ignore_failure 的极端情况，一般只会出现在 ignore_failure >= 任务数。
            return (
                [],
                f"⚠️ 所有任务均失败，但失败数未超过 ignore_failure={ignore_failure}，因此不返回占位图。",
            )

        all_output_tensors = []
        all_result_texts = [
            "📊 gpt-image-2 批量任务汇总:",
            f"总任务数: {len(tasks)}",
            f"成功任务数: {len(success_results)}",
            f"失败任务数: {len(failed_results)}",
            f"超时失败数: {len(timeout_failed_results)}",
            f"ignore_failure: {ignore_failure}",
            "",
        ]

        if failed_results:
            if should_add_placeholders:
                all_result_texts.append(
                    f"❌ 失败任务数 {len(failed_results)} > ignore_failure {ignore_failure}，"
                    f"已保留成功图片，并补充 {len(failed_results)} 张失败占位图。"
                )
            else:
                all_result_texts.append("⚠️ 以下失败任务已被 ignore_failure 容忍，未输出占位图:")

            for r in failed_results:
                all_result_texts.append(
                    f"- 任务 {r['index']}: timeout={r.get('is_timeout')}, attempt={r.get('attempt')}, error={r.get('text')}"
                )
            all_result_texts.append("")

        # 按任务 index 输出：成功任务输出真实图片；当失败数超过 ignore_failure 时，失败任务输出 1 张占位图。
        # 这样可以尽量保持输出顺序与任务顺序一致。
        result_by_index = {r["index"]: r for r in all_results}
        for task in sorted(tasks, key=lambda x: x["index"]):
            r = result_by_index.get(task["index"])
            if not r:
                continue

            t_idx = r["index"]
            if r.get("ok"):
                all_output_tensors.extend(r["images"])
                all_result_texts.append(f"===== 任务 {t_idx} =====\n{r['text']}")
            elif should_add_placeholders:
                all_output_tensors.append(self.make_placeholder_tensor("red"))
                all_result_texts.append(f"===== 任务 {t_idx} 失败占位图 =====\n{r.get('text')}")

        text_output = "\n\n".join(all_result_texts)
        return (all_output_tensors, text_output)

    def _execute_generation(
        self,
        prompt,
        image1,
        model,
        aspect_ratio,
        quality,
        max_images,
        output_format,
        output_compression,
        base_url,
        seed,
        enable_auto_retry,
        timeout,
        image2=None,
        image3=None,
        image4=None,
        image5=None,
        image6=None,
    ):
        """
        实际执行图像生成的核心逻辑
        """
        try:
            # 标准化seed参数 - 将大的seed值映射到有效范围内
            normalized_seed = seed
            if seed > 2147483647:
                # 使用模运算将大seed值映射到有效范围
                normalized_seed = seed % 2147483647
                print(f"原始seed值 {seed} 被标准化为 {normalized_seed}")

            api_key = self.get_api_key()

            # Collect input images
            input_images = [image1]
            if image2 is not None:
                input_images.append(image2)
            if image3 is not None:
                input_images.append(image3)
            if image4 is not None:
                input_images.append(image4)
            if image5 is not None:
                input_images.append(image5)
            if image6 is not None:
                input_images.append(image6)

            # gpt-image-2 编辑接口要求 multipart/form-data。
            multipart_field = "image" if len(input_images) == 1 else "image[]"
            files = []
            for i, img_tensor in enumerate(input_images):
                pil_img = self.tensor_to_pil(img_tensor.squeeze(0))
                files.append(
                    (
                        multipart_field,
                        (f"image_{i + 1}.png", self.pil_to_png_bytes(pil_img), "image/png"),
                    )
                )

            # Convert aspect ratio to size
            size = self.aspect_ratio_to_size(aspect_ratio)

            endpoint = base_url.rstrip("/")
            if not endpoint.endswith("/images/edits"):
                endpoint += "/images/edits"

            response = requests.post(
                endpoint,
                headers={"Authorization": f"Bearer {api_key}"},
                files=files,
                data={
                    "model": model,
                    "prompt": prompt,
                    "size": size,
                    "n": str(max_images),
                    "quality": quality,
                    "output_format": output_format,
                    "output_compression": str(output_compression),
                },
                timeout=timeout,
            )

            try:
                response_data = response.json()
            except ValueError as json_error:
                response.raise_for_status()
                raise RuntimeError("API 返回了无法解析的非 JSON 响应") from json_error

            if not response.ok or response_data.get("error"):
                api_error = response_data.get("error") or {}
                message = (
                    api_error.get("message") or response.text or f"HTTP {response.status_code}"
                )
                code = api_error.get("code")
                request_id = api_error.get("param")
                details = ", ".join(
                    item
                    for item in [
                        f"code={code}" if code else "",
                        f"request_id={request_id}" if request_id else "",
                    ]
                    if item
                )
                raise RuntimeError(
                    f"OpenAI API 错误: {message}" + (f" ({details})" if details else "")
                )

            image_items = response_data.get("data") or []

            # Process generated images and collect information
            output_tensors = []
            result_info = []

            # Collect basic generation info
            result_info.append("🎨 生成信息:")
            result_info.append(f"📝 提示词: {prompt}")
            result_info.append(f"🔧 模型: {model}")
            result_info.append(f"📐 输出尺寸: {size}")
            result_info.append(f"🖼️ 生成数量: {len(image_items)}")
            result_info.append(
                f"📊 输入图像: {len([img for img in [image1, image2, image3, image4, image5, image6] if img is not None])}"
            )
            result_info.append(
                f"🎲 种子值: {normalized_seed}"
                + (f" (原始: {seed})" if seed != normalized_seed else "")
            )
            result_info.append(
                f"⚡ 执行状态: 成功 (自动重试: {'启用' if enable_auto_retry else '禁用'})"
            )
            result_info.append("")

            import base64

            for i, image_data in enumerate(image_items):
                result_info.append(f"📷 图像 {i + 1}:")
                image_b64 = image_data.get("b64_json")
                image_url = image_data.get("url")
                if image_b64:
                    raw_b64 = (
                        image_b64.split(",", 1)[-1] if image_b64.startswith("data:") else image_b64
                    )
                    image_bytes = base64.b64decode(raw_b64)
                    image = Image.open(io.BytesIO(image_bytes))
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    tensor = self.pil_to_tensor(image)
                    output_tensors.append(tensor)
                    result_info.append("   ✅ 已解析 Base64 图像")
                elif image_url:
                    output_tensors.append(self.download_image_from_url(image_url))
                    result_info.append(f"   🔗 URL: {image_url}")
                else:
                    raise RuntimeError(f"第 {i + 1} 个结果不含 b64_json 或 url")

                result_info.append("")

            # Add generation parameters info
            result_info.append("⚙️ 生成参数:")
            result_info.append(f"   🎯 输出格式: {output_format}")
            result_info.append(f"   ⭐ 图片质量: {quality}")
            result_info.append(f"   🗜️ 压缩强度: {output_compression}")
            result_info.append(f"   🌐 API地址: {endpoint}")

            if not output_tensors:
                # 不在单任务内返回占位图；统一交给 generate_images 根据 ignore_failure 决定。
                raise RuntimeError("任务未生成任何图像: API data 为空")

            # Join all info into a single text output
            text_output = "\n".join(result_info)

            return (output_tensors, text_output)

        except Exception as e:
            error_msg = str(e)

            # 确保normalized_seed在错误处理时也可用
            normalized_seed = seed
            if seed > 2147483647:
                normalized_seed = seed % 2147483647

            # Create detailed error text output with specific troubleshooting
            error_text_parts = ["❌ 图像生成失败", "", f"🔍 错误信息: {error_msg}", ""]

            # 根据错误类型提供具体的解决建议
            if "image1 参数是必需的" in error_msg:
                error_text_parts.extend(
                    [
                        "🚨 输入图像问题:",
                        "   • image1 输入未连接或上游节点未执行完成",
                        "   • 请确保LoadImage或其他图像生成节点已正确连接",
                        "   • 建议等待上游节点完全执行后再运行此节点",
                        "   • 如果使用API调用，请确保所有依赖节点按正确顺序执行",
                        "",
                    ]
                )
            elif "torch.Tensor" in error_msg:
                error_text_parts.extend(
                    [
                        "🚨 数据类型问题:",
                        "   • 输入的image1不是有效的图像tensor",
                        "   • 请检查上游节点是否正确输出图像数据",
                        "   • 确保连接的是图像输出端口，而不是其他类型的输出",
                        "",
                    ]
                )
            elif "Invalid image file" in error_msg:
                error_text_parts.extend(
                    [
                        "🚨 图像文件问题:",
                        "   • 上游LoadImage节点的图像文件无效或不存在",
                        "   • 常见原因:",
                        "     - 文件路径格式错误（如：client:syai-prod/...）",
                        "     - 临时文件还未生成完成",
                        "     - 文件权限或网络问题",
                        "     - 工作流执行顺序问题",
                        "   • 解决方案:",
                        "     1. 检查LoadImage节点的输入路径是否正确",
                        "     2. 确保使用本地文件路径而非URL格式",
                        "     3. 等待上游节点完全执行后再运行",
                        "     4. 检查文件是否存在且可读",
                        "",
                    ]
                )
            elif "API Key" in error_msg:
                error_text_parts.extend(
                    [
                        "🚨 API配置问题:",
                        "   • OPENAI_API_KEY 环境变量未设置或无效",
                        "   • 请设置环境变量: export OPENAI_API_KEY='your_api_key'",
                        "   • 确保API Key有效且有足够的配额",
                        "",
                    ]
                )
            elif "bigger than max" in error_msg and "seed" in error_msg:
                error_text_parts.extend(
                    [
                        "🚨 Seed值溢出问题:",
                        f"   • 原始seed值 {seed} 超过了系统支持的最大值",
                        f"   • 已自动标准化为: {normalized_seed}",
                        "   • 这不会影响图像生成质量，只是用于工作流跟踪",
                        "   • 建议使用较小的seed值以避免此警告",
                        "",
                    ]
                )

            error_text_parts.extend(
                [
                    f"📝 提示词: {prompt}",
                    f"🔧 模型: {model}",
                    f"📐 宽高比: {aspect_ratio}",
                    f"🖼️ 最大图像数: {max_images}",
                    f"🌐 API地址: {base_url}",
                    f"⭐ 图片质量: {quality}",
                    f"🎯 输出格式: {output_format}",
                    f"🗜️ 压缩强度: {output_compression}",
                    f"🎲 种子值: {normalized_seed}"
                    + (f" (原始: {seed})" if seed != normalized_seed else ""),
                    "",
                    "💡 故障排除步骤:",
                    "   1. 检查所有节点连接是否正确",
                    "   2. 确保上游节点已完全执行",
                    "   3. 验证API Key和网络连接",
                    "   4. 查看ComfyUI控制台获取详细日志",
                ]
            )

            error_text = "\n".join(error_text_parts)

            # 打印详细错误信息到控制台以便调试
            print("GPTImage2Generate 错误详情:")
            print(f"  错误类型: {type(e).__name__}")
            print(f"  错误信息: {error_msg}")
            print(f"  image1 类型: {type(image1) if 'image1' in locals() else 'undefined'}")
            if "image1" in locals() and image1 is not None:
                print(f"  image1 形状: {getattr(image1, 'shape', 'N/A')}")

            # 不在单任务内返回错误占位图；统一交给 generate_images 根据 ignore_failure 决定。
            raise RuntimeError(error_text) from e
