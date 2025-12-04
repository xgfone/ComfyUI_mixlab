from .aliyun_face_beauty import AliyunFaceBeautyNode
from .doubao import DoubaoSingleTurnChatNodeSDKv2
from .face2mask import FaceToMaskCopy
from .garment_category import GarmentCategoryMapper, GarmentCategoryMapperBatch
from .load_image_from_url import LoadImageAndMaskFromUrl
from .mask_sort import MaskSorter
from .prompt_logo_cleaner import PromptLogoCleaner
from .raster_card_maker import RasterCardMaker
from .seedream_concurrent import SeedreamImageGenerateConcurrent

NODE_CLASS_MAPPINGS = {
    "MaskSorter": MaskSorter,
    "FaceToMaskCopy": FaceToMaskCopy,
    "RasterCardMaker": RasterCardMaker,
    "AliyunFaceBeauty": AliyunFaceBeautyNode,
    "PromptLogoCleaner": PromptLogoCleaner,
    "LoadImageAndMaskFromUrl": LoadImageAndMaskFromUrl,
    "GarmentCategoryMapper": GarmentCategoryMapper,
    "GarmentCategoryMapperBatch": GarmentCategoryMapperBatch,
    "DoubaoSingleTurnChatNodeSDKv2": DoubaoSingleTurnChatNodeSDKv2,
    "SeedreamImageGenerateConcurrent": SeedreamImageGenerateConcurrent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskSorter": "🧩 Mask Sorter (多蒙版排序)",
    "FaceToMaskCopy": "Face To Mask(Copy)",
    "RasterCardMaker": "Raster Card Maker",
    "AliyunFaceBeauty": "Aliyun Face Beauty (Retouch)",
    "PromptLogoCleaner": "Prompt Logo Cleaner (Remove Logo Words)",
    "LoadImageAndMaskFromUrl": "Load Image And Mask From Url",
    "GarmentCategoryMapper": "Garment Category Mapper (1/2/3)",
    "GarmentCategoryMapperBatch": "Garment Category Mapper (Batch)",
    "DoubaoSingleTurnChatNodeSDKv2": "Doubao Chat (Single Turn, Ark SDK)",
    "SeedreamImageGenerateConcurrent": "Seedream Image Generate (Concurrent)",
}
