from .face2mask import FaceToMaskCopy
from .garment_category import GarmentCategoryMapper, GarmentCategoryMapperBatch
from .load_image_from_url import LoadImageAndMaskFromUrl
from .prompt_logo_cleaner import PromptLogoCleaner
from .raster_card_maker import RasterCardMaker

NODE_CLASS_MAPPINGS = {
    "FaceToMaskCopy": FaceToMaskCopy,
    "RasterCardMaker": RasterCardMaker,
    "PromptLogoCleaner": PromptLogoCleaner,
    "LoadImageAndMaskFromUrl": LoadImageAndMaskFromUrl,
    "GarmentCategoryMapper": GarmentCategoryMapper,
    "GarmentCategoryMapperBatch": GarmentCategoryMapperBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceToMaskCopy": "Face To Mask(Copy)",
    "RasterCardMaker": "Raster Card Maker",
    "PromptLogoCleaner": "Prompt Logo Cleaner (Remove Logo Words)",
    "LoadImageAndMaskFromUrl": "Load Image And Mask From Url",
    "GarmentCategoryMapper": "Garment Category Mapper (1/2/3)",
    "GarmentCategoryMapperBatch": "Garment Category Mapper (Batch)",
}
