from .aliyun_face_beauty import AliyunFaceBeautyNode
from .aliyun_image_seg import AliyunCommonSegmentation
from .auto_gamma import AutoGamma
from .chroma_key import ChromaKeyNode
from .color_ratio_node import ColorRatioCalculator
from .corner_pin import WEB_DIRECTORY, BIMO_CornerPinPerspective
from .doubao import DoubaoSingleTurnChatNodeSDKv2
from .face2mask import FaceToMaskCopy
from .garment_category import GarmentCategoryMapper, GarmentCategoryMapperBatch
from .gpt_image_2 import GPTImage2Generator
from .load_image_from_url import LoadImageAndMaskFromUrl
from .mask_sort import MaskSorter
from .prompt_logo_cleaner import PromptLogoCleaner
from .raster_card_maker import RasterCardMaker
from .seedream_concurrent import SeedreamImageGenerateConcurrent
from .seedream_node_executor import SeedreamImageGenerateExecutor
from .split_string import SplitString
from .switch_case_node import SwitchCaseNodePro
from .zho_text_image import Text_Image_Multiline_Zho_autofit, Text_Image_Zho_autofit

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]


NODE_CLASS_MAPPINGS = {
    "SplitString": SplitString,
    "AutoGamma": AutoGamma,
    "ChromaKey": ChromaKeyNode,
    "MaskSorter": MaskSorter,
    "FaceToMaskCopy": FaceToMaskCopy,
    "SwitchCaseNodePro": SwitchCaseNodePro,
    "RasterCardMaker": RasterCardMaker,
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
    "AliyunCommonSegmentation": AliyunCommonSegmentation,
    "Text_Image_Zho_autofit": Text_Image_Zho_autofit,
    "Text_Image_Multiline_Zho_autofit": Text_Image_Multiline_Zho_autofit,
    "BIMO_CornerPinPerspective": BIMO_CornerPinPerspective,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "SplitString": "Split String",
    "AutoGamma": "Auto Gamma",
    "GPTImage2Generator": "OpenAI GPT Image 2",
    "ColorRatioCalculator": "Color Ratio Calculator",
    "ChromaKey": "Chroma Key",
    "MaskSorter": "🧩 Mask Sorter (多蒙版排序)",
    "FaceToMaskCopy": "Face To Mask(Copy)",
    "SwitchCaseNodePro": "Switch Case Node Pro",
    "RasterCardMaker": "Raster Card Maker",
    "AliyunFaceBeauty": "Aliyun Face Beauty (Retouch)",
    "PromptLogoCleaner": "Prompt Logo Cleaner (Remove Logo Words)",
    "LoadImageAndMaskFromUrl": "Load Image And Mask From Url",
    "GarmentCategoryMapper": "Garment Category Mapper (1/2/3)",
    "GarmentCategoryMapperBatch": "Garment Category Mapper (Batch)",
    "DoubaoSingleTurnChatNodeSDKv2": "Doubao Chat (Single Turn, Ark SDK)",
    "SeedreamImageGenerateConcurrent": "Seedream Image Generate (Concurrent)",
    "SeedreamImageGenerateExecutor": "Seedream Image Generate Executor",
    "AliyunCommonSegmentation": "Aliyun Common Segmentation (crop/mask/whiteBK)",
    "Text_Image_Zho_autofit": "Text Image Zho AutoFit",
    "Text_Image_Multiline_Zho_autofit": "Text Image Multiline Zho AutoFit",
    "BIMO_CornerPinPerspective": "Corner Pin / Perspective Warp",
}
