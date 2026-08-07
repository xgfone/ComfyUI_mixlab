# -*- coding: utf-8 -*-

from .py.aliyun_face_beauty import AliyunFaceBeautyNode
from .py.auto_gamma import AutoGamma
from .py.bimoai_segment_node import BimoAISegmentImage
from .py.bimoai_text_split import BimoAITextSplitIndex
from .py.chroma_key import ChromaKeyNode
from .py.color_ratio_node import ColorRatioCalculator
from .py.corner_pin import WEB_DIRECTORY, BIMO_CornerPinPerspective
from .py.doubao import DoubaoSingleTurnChatNodeSDKv2
from .py.garment_category import GarmentCategoryMapper, GarmentCategoryMapperBatch
from .py.gemini_image_node_executor import GeminiImageGenerateExecutor
from .py.gpt_image_2 import GPTImage2Generator
from .py.load_image_from_url import LoadImageAndMaskFromUrl
from .py.mask_sort import MaskSorter
from .py.prompt_logo_cleaner import PromptLogoCleaner
from .py.seedream_concurrent import SeedreamImageGenerateConcurrent
from .py.seedream_node_executor import SeedreamImageGenerateExecutor
from .py.split_string import SplitString
from .py.switch_case_node import SwitchCaseNodePro
from .py.was_text_shuffle import WASTextShuffle
from .py.zho_text_image import Text_Image_Multiline_Zho_autofit, Text_Image_Zho_autofit

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]


NODE_CLASS_MAPPINGS = {
    "SplitString": SplitString,
    "AutoGamma": AutoGamma,
    "ChromaKey": ChromaKeyNode,
    "MaskSorter": MaskSorter,
    "SwitchCaseNodePro": SwitchCaseNodePro,
    "AliyunFaceBeauty": AliyunFaceBeautyNode,
    "PromptLogoCleaner": PromptLogoCleaner,
    "GPTImage2Generator": GPTImage2Generator,
    "ColorRatioCalculator": ColorRatioCalculator,
    "LoadImageAndMaskFromUrl": LoadImageAndMaskFromUrl,
    "GarmentCategoryMapper": GarmentCategoryMapper,
    "GarmentCategoryMapperBatch": GarmentCategoryMapperBatch,
    "DoubaoSingleTurnChatNodeSDKv2": DoubaoSingleTurnChatNodeSDKv2,
    "SeedreamImageGenerateConcurrent": SeedreamImageGenerateConcurrent,
    "SeedreamImageGenerateExecutor": SeedreamImageGenerateExecutor,
    "Text_Image_Zho_autofit": Text_Image_Zho_autofit,
    "Text_Image_Multiline_Zho_autofit": Text_Image_Multiline_Zho_autofit,
    "BIMO_CornerPinPerspective": BIMO_CornerPinPerspective,
    "GeminiImageGenerate": GeminiImageGenerateExecutor,
    "BimoAITextSplitIndex": BimoAITextSplitIndex,
    "BimoAISegmentImage": BimoAISegmentImage,
    "WASTextShuffle": WASTextShuffle,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "SplitString": "Split String",
    "AutoGamma": "Auto Gamma",
    "GPTImage2Generator": "OpenAI GPT Image 2",
    "ColorRatioCalculator": "Color Ratio Calculator",
    "ChromaKey": "Chroma Key",
    "MaskSorter": "🧩 Mask Sorter (多蒙版排序)",
    "SwitchCaseNodePro": "Switch Case Node Pro",
    "AliyunFaceBeauty": "Aliyun Face Beauty (Retouch)",
    "PromptLogoCleaner": "Prompt Logo Cleaner (Remove Logo Words)",
    "LoadImageAndMaskFromUrl": "Load Image And Mask From Url",
    "GarmentCategoryMapper": "Garment Category Mapper (1/2/3)",
    "GarmentCategoryMapperBatch": "Garment Category Mapper (Batch)",
    "DoubaoSingleTurnChatNodeSDKv2": "Doubao Chat (Single Turn, Ark SDK)",
    "SeedreamImageGenerateConcurrent": "Seedream Image Generate (Concurrent)",
    "SeedreamImageGenerateExecutor": "Seedream Image Generate Executor",
    "Text_Image_Zho_autofit": "Text Image Zho AutoFit",
    "Text_Image_Multiline_Zho_autofit": "Text Image Multiline Zho AutoFit",
    "BIMO_CornerPinPerspective": "Corner Pin / Perspective Warp",
    "GeminiImageGenerate": "Gemini Image Generator",
    "BimoAITextSplitIndex": "BimoAI文本分隔元素读取",
    "BimoAISegmentImage": "BimoAI Image Segment",
    "WASTextShuffle": "WAS Text Shuffle",
}
