import json
import os

import numpy as np
import torch


# --- 核心算法部分 (保持不变) ---
def lerp(a, b, w):
    return a + w * (b - a)


def saturate(x):
    return np.clip(x, 0.0, 1.0)


def np_rgb2hsv(c):
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    max_c = np.maximum(r, np.maximum(g, b))
    min_c = np.minimum(r, np.minimum(g, b))
    diff = max_c - min_c
    h = np.zeros_like(max_c)
    mask_diff = diff != 0
    mask_r = (max_c == r) & mask_diff
    mask_g = (max_c == g) & (~mask_r) & mask_diff
    mask_b = (max_c == b) & (~mask_r) & (~mask_g) & mask_diff
    h[mask_r] = (g[mask_r] - b[mask_r]) / diff[mask_r]
    h[mask_g] = (b[mask_g] - r[mask_g]) / diff[mask_g] + 2.0
    h[mask_b] = (r[mask_b] - g[mask_b]) / diff[mask_b] + 4.0
    h = (h / 6.0) % 1.0
    s = np.zeros_like(max_c)
    s[mask_diff] = diff[mask_diff] / max_c[mask_diff]
    v = max_c
    return np.stack([h, s, v], axis=-1)


def np_hsv2rgb(c):
    h, s, v = c[..., 0], c[..., 1], c[..., 2]
    i = (h * 6.0).astype(int)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    r, g, b = np.zeros_like(h), np.zeros_like(h), np.zeros_like(h)
    mask_0 = i == 0
    r[mask_0], g[mask_0], b[mask_0] = v[mask_0], t[mask_0], p[mask_0]
    mask_1 = i == 1
    r[mask_1], g[mask_1], b[mask_1] = q[mask_1], v[mask_1], p[mask_1]
    mask_2 = i == 2
    r[mask_2], g[mask_2], b[mask_2] = p[mask_2], v[mask_2], t[mask_2]
    mask_3 = i == 3
    r[mask_3], g[mask_3], b[mask_3] = p[mask_3], q[mask_3], v[mask_3]
    mask_4 = i == 4
    r[mask_4], g[mask_4], b[mask_4] = t[mask_4], p[mask_4], v[mask_4]
    mask_5 = i == 5
    r[mask_5], g[mask_5], b[mask_5] = v[mask_5], p[mask_5], q[mask_5]
    return np.stack([r, g, b], axis=-1)


class ChromaKeyNode:
    def __init__(self):
        self.base_dir = os.path.dirname(__file__)
        self.json_dir = os.path.join(self.base_dir, "presets")
        if not os.path.exists(self.json_dir):
            os.makedirs(self.json_dir)

    @classmethod
    def INPUT_TYPES(s):
        json_files = ["None"]
        preset_dir = os.path.join(os.path.dirname(__file__), "presets")
        if os.path.exists(preset_dir):
            json_files += [f for f in os.listdir(preset_dir) if f.endswith(".json")]

        return {
            "required": {
                "image": ("IMAGE",),
                "preset_json": (json_files,),
                "key_color_r": ("INT", {"default": 0, "min": 0, "max": 255}),
                "key_color_g": ("INT", {"default": 255, "min": 0, "max": 255}),
                "key_color_b": ("INT", {"default": 0, "min": 0, "max": 255}),
                "Prekey_despill": (
                    "FLOAT",
                    {"default": -1000.0, "min": -2000.0, "max": 1000.0, "step": 1.0},
                ),
                "Prekey_saturate": (
                    "FLOAT",
                    {"default": -1000.0, "min": -2000.0, "max": 1000.0, "step": 1.0},
                ),
                "Matte_white": (
                    "FLOAT",
                    {"default": 0.0, "min": -2000.0, "max": 1000.0, "step": 1.0},
                ),
                "Matte_black": (
                    "FLOAT",
                    {"default": 0.0, "min": -2000.0, "max": 1000.0, "step": 1.0},
                ),
                "Matte_highlights": (
                    "FLOAT",
                    {"default": -1000.0, "min": -2000.0, "max": 1000.0, "step": 1.0},
                ),
                "Matte_shadows": (
                    "FLOAT",
                    {"default": -1000.0, "min": -2000.0, "max": 1000.0, "step": 1.0},
                ),
                "Shadows": (
                    "FLOAT",
                    {"default": -1000.0, "min": -2000.0, "max": 1000.0, "step": 1.0},
                ),
                "Spill_reduction": (
                    "FLOAT",
                    {"default": 0.0, "min": -2000.0, "max": 1000.0, "step": 1.0},
                ),
                "Spill_balance": (
                    "FLOAT",
                    {"default": 0.0, "min": -2000.0, "max": 1000.0, "step": 1.0},
                ),
                "Spill_unpremultiply": (
                    "FLOAT",
                    {"default": 0.0, "min": -2000.0, "max": 1000.0, "step": 1.0},
                ),
                "Premultiply": (
                    "FLOAT",
                    {"default": 0.0, "min": -2000.0, "max": 1000.0, "step": 1.0},
                ),
                "Use_alternate_key_method": ("BOOLEAN", {"default": True}),
            }
        }

    # 修改这里：增加 PROCESSED_IMAGE
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("Result (Transparent)", "Mask", "Result (Opaque)")
    FUNCTION = "process"
    CATEGORY = "OBS/ChromaKey"

    def process(
        self,
        image,
        preset_json,
        key_color_r,
        key_color_g,
        key_color_b,
        Prekey_despill,
        Prekey_saturate,
        Matte_white,
        Matte_black,
        Matte_highlights,
        Matte_shadows,
        Shadows,
        Spill_reduction,
        Spill_balance,
        Spill_unpremultiply,
        Premultiply,
        Use_alternate_key_method,
    ):
        params = {
            "Prekey_despill": Prekey_despill,
            "Prekey_saturate": Prekey_saturate,
            "Matte_white": Matte_white,
            "Matte_black": Matte_black,
            "Matte_highlights": Matte_highlights,
            "Matte_shadows": Matte_shadows,
            "Shadows": Shadows,
            "Spill_reduction": Spill_reduction,
            "Spill_balance": Spill_balance,
            "Spill_unpremultiply": Spill_unpremultiply,
            "Premultiply": Premultiply,
            "Use_alternate_key_method": Use_alternate_key_method,
            # 这里的 Show_xxx 参数不再需要从节点输入获取，因为我们总是计算所有结果
            "Show_Alpha": False,
            "Show_PrekeyFG": False,
            "Show_ProcessedFG": False,
        }

        current_key_color = (key_color_r, key_color_g, key_color_b)

        if preset_json != "None":
            json_path = os.path.join(self.json_dir, preset_json)
            if os.path.exists(json_path):
                try:
                    with open(json_path, "r") as f:
                        data = json.load(f)
                        if "key_color" in data:
                            current_key_color = tuple(data["key_color"])
                        if "params" in data:
                            for k, v in data["params"].items():
                                params[k] = v
                    print(f"Loaded preset: {preset_json}")
                except Exception as e:
                    print(f"Error loading preset {preset_json}: {e}")

        results_transparent = []
        masks = []
        results_opaque = []  # 新增列表

        for i in range(image.shape[0]):
            img_np = image[i].cpu().numpy()
            img_rgba = np.dstack((img_np, np.ones((img_np.shape[0], img_np.shape[1])))) * 255.0

            # 核心处理：这里不再返回单一结果，而是返回一个包含所需数据的字典
            processed_data = self.process_core(img_rgba, params, current_key_color)

            # 1. 透明图像
            res_tensor = torch.from_numpy(processed_data["final_transparent"].astype(np.float32) / 255.0)
            results_transparent.append(res_tensor[..., :3])  # RGB
            masks.append(res_tensor[..., 3])  # Alpha

            # 2. 不透明处理后图像
            opaque_tensor = torch.from_numpy(processed_data["final_opaque"].astype(np.float32) / 255.0)
            results_opaque.append(opaque_tensor[..., :3])  # 只要 RGB (Alpha 肯定是 1)

        return (
            torch.stack(results_transparent),
            torch.stack(masks),
            torch.stack(results_opaque),
        )

    def process_core(self, img_array, params, key_color_rgb):
        # ... (参数提取逻辑同前) ...
        raw_color = img_array.astype(np.float32) / 255.0
        color = raw_color.copy()
        Key_color = np.array(
            [
                key_color_rgb[0] / 255.0,
                key_color_rgb[1] / 255.0,
                key_color_rgb[2] / 255.0,
                1.0,
            ]
        )

        def get_val(name, factor):
            return (1000 + params[name]) * factor

        # ... (所有参数获取代码，同 V5) ...
        Prekey_despill2 = get_val("Prekey_despill", 0.0002)
        Prekey_saturate2 = get_val("Prekey_saturate", 0.002)
        Matte_white2 = get_val("Matte_white", 0.001)
        Matte_black2 = get_val("Matte_black", 0.005)
        Matte_highlights2 = get_val("Matte_highlights", 0.0005)
        Matte_shadows2 = get_val("Matte_shadows", 0.002)
        Spill_reduction2 = get_val("Spill_reduction", 0.0005)
        Spill_balance2 = get_val("Spill_balance", 0.0005)
        Spill_unpremultiply2 = get_val("Spill_unpremultiply", 0.001)
        Premultiply2 = get_val("Premultiply", 0.001)
        Use_alternate_key_method = params["Use_alternate_key_method"]

        # 1. 色相偏移
        color_hsv = np_rgb2hsv(color[..., :3])
        key_hsv = np_rgb2hsv(Key_color[:3].reshape(1, 1, 3))
        hueOffset = 0.333333 - key_hsv[..., 0]
        color_hsv[..., 0] = (color_hsv[..., 0] + hueOffset) % 1.0
        color_hsv[..., 1] *= 1.0 + Prekey_saturate2
        color[..., :3] = np_hsv2rgb(color_hsv)

        key_hsv_shifted = key_hsv.copy()
        key_hsv_shifted[..., 0] = (key_hsv_shifted[..., 0] + hueOffset) % 1.0
        key_hsv_shifted[..., 1] *= 1.0 + Prekey_saturate2
        Key_color2_rgb = np_hsv2rgb(key_hsv_shifted)

        color[..., 0] += Prekey_despill2
        color[..., 2] += Prekey_despill2
        # PrekeyFG_shifted = color.copy() # 这里不需要预览了

        # 2. 抠像
        alpha = np.ones_like(color[..., 0])
        if Use_alternate_key_method:
            Y = (0.299 * color[..., 0]) + (0.587 * color[..., 1]) + (0.114 * color[..., 2])
            U, V = (
                ((color[..., 2] - Y) * 0.565) + 0.5,
                ((color[..., 0] - Y) * 0.713) + 0.5,
            )
            Y2 = (0.299 * Key_color2_rgb[..., 0]) + (0.587 * Key_color2_rgb[..., 1]) + (0.114 * Key_color2_rgb[..., 2])
            U2, V2 = (
                ((Key_color2_rgb[..., 2] - Y2) * 0.565) + 0.5,
                ((Key_color2_rgb[..., 0] - Y2) * 0.713) + 0.5,
            )
            U2, V2 = np.where(U2 == 0, 1e-4, U2), np.where(V2 == 0, 1e-4, V2)
            alpha = 1.0 - 2.0 * np.maximum(np.abs((U / U2) - 1.0), np.abs((V / V2) - 1.0))
        else:
            alpha = color[..., 1] - np.maximum(color[..., 0] + Prekey_despill2, color[..., 2] + Prekey_despill2)

        # 3. 亮度遮罩
        if Matte_highlights2 > 0.0:
            alpha -= Matte_highlights2 * saturate(
                (color[..., 0] - Key_color[0]) + (color[..., 1] - Key_color[1]) + (color[..., 2] - Key_color[2])
            )
        if Matte_shadows2 > 0.0:
            alpha -= Matte_shadows2 * saturate(
                ((1.0 - color[..., 0]) - (1.0 - Key_color[0]))
                + ((1.0 - color[..., 1]) - (1.0 - Key_color[1]))
                + ((1.0 - color[..., 2]) - (1.0 - Key_color[2]))
            )

        # 4. Alpha 调整
        alpha = saturate((1.0 - alpha * Matte_black2) * Matte_white2)

        # 5. 预乘 & 溢色去除 (复用之前逻辑)
        mask_proc = (alpha != 0.0) & (alpha != 1.0)
        alpha_exp = alpha[..., np.newaxis]
        if Premultiply2 < 1.0:
            div_alpha = np.where(
                alpha_exp > 0,
                raw_color[..., :3] / (alpha_exp + 1e-6),
                raw_color[..., :3],
            )
            color[..., :3] = np.where(
                mask_proc[..., np.newaxis],
                lerp(div_alpha, raw_color[..., :3], Premultiply2),
                color[..., :3],
            )
        else:
            color[..., :3] = np.where(
                mask_proc[..., np.newaxis],
                lerp(
                    raw_color[..., :3],
                    raw_color[..., :3] * alpha_exp,
                    Premultiply2 - 1.0,
                ),
                color[..., :3],
            )
        color[..., :3] = np.where((alpha == 1.0)[..., np.newaxis], raw_color[..., :3], color[..., :3])

        Spill_rb = lerp(color[..., 0], color[..., 2], Spill_balance2)
        Spill_unpre = np.zeros_like(color[..., 1])
        mask_gt = color[..., 1] > Spill_rb
        Spill_unpre[mask_gt] = color[..., 1][mask_gt] - Spill_rb[mask_gt]
        factor = Spill_unpremultiply2 * Spill_unpre
        kc_safe = Key_color + 1e-6
        for i in range(3):
            color[..., i] = lerp(color[..., i], color[..., i] / kc_safe[i], factor)

        rb_max, rb_min = (
            np.maximum(color[..., 0], color[..., 2]),
            np.minimum(color[..., 0], color[..., 2]),
        )
        rb_bal = lerp(color[..., 0], color[..., 2], Spill_balance2)
        thresh1 = rb_max
        thresh2 = lerp(rb_max, rb_bal, (Spill_reduction2 - 0.25) * 4.0)
        thresh3 = lerp(rb_bal, rb_min, (Spill_reduction2 - 0.5) * 2.0)
        if Spill_reduction2 < 0.25:
            color[..., 1] = np.where(
                color[..., 1] > thresh1,
                lerp(color[..., 1], thresh1, Spill_reduction2 * 4.0),
                color[..., 1],
            )
        elif Spill_reduction2 < 0.5:
            color[..., 1] = np.where(color[..., 1] > thresh2, thresh2, color[..., 1])
        else:
            color[..., 1] = np.where(color[..., 1] > thresh3, thresh3, color[..., 1])

        # 8. 色相还原
        color_hsv_back = np_rgb2hsv(color[..., :3])
        color_hsv_back[..., 0] = (color_hsv_back[..., 0] - hueOffset) % 1.0
        color[..., :3] = np_hsv2rgb(color_hsv_back)

        # --- 生成两种输出 ---

        # A. 带透明通道的结果
        res_transparent = color.copy()
        res_transparent[..., 3] = alpha
        mask_zero = alpha == 0.0
        for i in range(3):
            res_transparent[..., i][mask_zero] = 0

        # B. 不透明结果 (强制 Alpha 为 1)
        res_opaque = color.copy()
        res_opaque[..., 3] = 1.0

        return {
            "final_transparent": np.clip(res_transparent * 255.0, 0, 255).astype(np.uint8),
            "final_opaque": np.clip(res_opaque * 255.0, 0, 255).astype(np.uint8),
        }


NODE_CLASS_MAPPINGS = {"ChromaKey": ChromaKeyNode}

NODE_DISPLAY_NAME_MAPPINGS = {"ChromaKey": "Chroma Key"}
