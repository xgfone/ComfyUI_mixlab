import math

import torch


class AutoGamma:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "MagickWand/Enhance"
    DESCRIPTION = "Automatically adjusts the gamma level of the image so the mean brightness becomes 0.5."

    def execute(self, image):
        # image 格式为 [batch, height, width, channels]
        # 1. 计算每张图的平均亮度 (mean)
        # dim=[1, 2, 3] 表示在 高、宽、通道 维度上求平均，保留 batch 维度
        means = torch.mean(image, dim=[1, 2, 3], keepdim=True)

        # 2. 限制 mean 的范围，防止 log(0) 或除以 0 错误
        # 极亮或极暗的图片会导致计算出的 gamma 无穷大或为 0
        means = torch.clamp(means, min=1e-6, max=1.0 - 1e-6)

        # 3. 计算 Gamma 值
        # 公式: Gamma = log(mean) / log(0.5)
        # 我们实际应用时使用的是指数 (1/Gamma)，所以可以直接计算倒数
        # exponent = 1 / Gamma = log(0.5) / log(mean)
        log_05 = math.log(0.5)
        exponents = log_05 / torch.log(means)

        # 4. 应用 Gamma 校正
        # PyTorch 的 pow 支持广播机制，[B, H, W, C] ** [B, 1, 1, 1]
        result = torch.pow(image, exponents)

        return (result,)
