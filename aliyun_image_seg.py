import io
import os

import numpy as np
import requests
import torch
from alibabacloud_imageseg20191230.client import Client as ImagesegClient
from alibabacloud_imageseg20191230.models import SegmentCommonImageAdvanceRequest
from alibabacloud_tea_openapi.models import Config
from alibabacloud_tea_util.models import RuntimeOptions
from PIL import Image


def _tensor_to_pil(img_tensor: torch.Tensor) -> Image.Image:
    """
    ComfyUI IMAGE: torch float32, [H, W, 3], range 0..1
    """
    img = img_tensor.detach().cpu().numpy()
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(img, mode="RGB")


def _pil_to_image_tensor(pil: Image.Image) -> torch.Tensor:
    """
    Return ComfyUI IMAGE: torch float32, [H, W, 3], range 0..1
    """
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    arr = np.array(pil).astype(np.float32) / 255.0
    return torch.from_numpy(arr)


def _pil_mask_to_comfy_mask(pil: Image.Image) -> torch.Tensor:
    """
    Return ComfyUI MASK: torch float32, [H, W], range 0..1
    """
    if pil.mode != "L":
        pil = pil.convert("L")
    arr = np.array(pil).astype(np.float32) / 255.0
    return torch.from_numpy(arr)


def _rgba_alpha_to_mask(pil_rgba: Image.Image) -> torch.Tensor:
    """
    If result is RGBA, use alpha as mask.
    """
    if pil_rgba.mode != "RGBA":
        pil_rgba = pil_rgba.convert("RGBA")
    alpha = pil_rgba.split()[-1]  # L
    return _pil_mask_to_comfy_mask(alpha)


class AliyunCommonSegmentation:
    """
    阿里云 通用分割（SegmentCommonImage）- ComfyUI 节点
    支持 ReturnForm: crop / mask / whiteBK（以及空=四通道 PNG）
    文档说明：ReturnForm 为 crop/mask/whiteBK 时分别返回裁剪透明图/二值mask/白底图。:contentReference[oaicite:2]{index=2}
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "return_form": (["crop", "mask", "whiteBK"], {"default": "crop"}),
            },
            "optional": {
                # 建议优先用环境变量（更安全）；也支持这里手填方便测试
                "access_key_id": ("STRING", {"default": ""}),
                "access_key_secret": ("STRING", {"default": ""}),
                "endpoint": ("STRING", {"default": "imageseg.cn-shanghai.aliyuncs.com"}),
                "region_id": ("STRING", {"default": "cn-shanghai"}),
                "timeout_seconds": ("INT", {"default": 60, "min": 5, "max": 300}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "debug")
    FUNCTION = "run"
    CATEGORY = "Aliyun/Imageseg"

    def _make_client(self, ak: str, sk: str, endpoint: str, region_id: str) -> ImagesegClient:
        if not ak:
            ak = os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_ID", "")
        if not sk:
            sk = os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_SECRET", "")
        if not ak or not sk:
            raise ValueError(
                "缺少 AccessKey：请设置环境变量 ALIBABA_CLOUD_ACCESS_KEY_ID / ALIBABA_CLOUD_ACCESS_KEY_SECRET，"
                "或在节点里填写 access_key_id / access_key_secret。"
            )

        config = Config(access_key_id=ak, access_key_secret=sk, endpoint=endpoint, region_id=region_id)
        return ImagesegClient(config)

    def _call_api_and_download(self, client: ImagesegClient, pil_img: Image.Image, return_form: str, timeout: int):
        # 传本地/内存文件：使用 AdvanceRequest 的 image_urlobject（BytesIO）
        # 阿里云示例里就是 image_urlobject = io.BytesIO(img) + segment_common_image_advance(...):contentReference[oaicite:3]{index=3}
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)

        req = SegmentCommonImageAdvanceRequest()
        req.image_urlobject = buf
        req.return_form = return_form

        runtime = RuntimeOptions()
        resp = client.segment_common_image_advance(req, runtime)

        # 返回的是临时可访问 ImageURL（有效期约 30 分钟）:contentReference[oaicite:4]{index=4}
        result_url = resp.body.data.image_url if hasattr(resp.body.data, "image_url") else resp.body.data.imageURL
        r = requests.get(result_url, timeout=timeout)
        r.raise_for_status()
        out_pil = Image.open(io.BytesIO(r.content))
        out_pil.load()
        return out_pil, str(result_url)

    def run(
        self,
        image,
        return_form,
        access_key_id="",
        access_key_secret="",
        endpoint="imageseg.cn-shanghai.aliyuncs.com",
        region_id="cn-shanghai",
        timeout_seconds=60,
    ):

        if not access_key_id or not access_key_secret:
            access_key_id = os.environ.get("ALIYUN_APPKEY", "")
            access_key_secret = os.environ.get("ALIYUN_SECRET", "")

        if not access_key_id or not access_key_secret:
            raise ValueError("API Key is required. Please set ARK_API_KEY environment variable.")
        client = self._make_client(access_key_id, access_key_secret, endpoint, region_id)

        # 支持 batch：逐张调用（外部API无法并行批处理时只能这样）
        out_images = []
        out_masks = []
        debug_lines = []

        if len(image.shape) != 4:
            raise ValueError("输入 IMAGE 维度不正确，期望 [B,H,W,C]")

        b = image.shape[0]
        for i in range(b):
            pil_in = _tensor_to_pil(image[i])
            out_pil, url = self._call_api_and_download(client, pil_in, return_form, int(timeout_seconds))

            # 统一输出：IMAGE + MASK（mask 没有就给全0占位，方便下游连线）
            if return_form == "mask":
                # 输出通常是单通道 mask 图:contentReference[oaicite:5]{index=5}
                mask = _pil_mask_to_comfy_mask(out_pil)
                # 同时给一个可预览的 3 通道“mask图”
                preview = out_pil.convert("L").convert("RGB")
                out_img_tensor = _pil_to_image_tensor(preview)
                out_images.append(out_img_tensor)
                out_masks.append(mask)
            else:
                # crop 通常是四通道 PNG（透明背景），whiteBK 是白底 RGB:contentReference[oaicite:6]{index=6}
                if out_pil.mode == "RGBA":
                    mask = _rgba_alpha_to_mask(out_pil)
                    out_rgb = out_pil.convert("RGB")
                else:
                    # whiteBK / 其它无 alpha：mask 给全0占位
                    out_rgb = out_pil.convert("RGB")
                    w, h = out_rgb.size
                    mask = torch.zeros((h, w), dtype=torch.float32)

                out_images.append(_pil_to_image_tensor(out_rgb))
                out_masks.append(mask)

            debug_lines.append(f"[{i}] return_form={return_form} url={url}")

        out_images = torch.stack(out_images, dim=0)  # [B,H,W,3]
        # MASK 在 ComfyUI 通常是 [B,H,W]
        out_masks = torch.stack(out_masks, dim=0)

        debug = "\n".join(debug_lines)
        return (out_images, out_masks, debug)
