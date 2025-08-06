from .face2mask import FaceToMaskCopy
from .load_image_from_url import LoadImageAndMaskFromUrl
from .prompt_logo_cleaner import PromptLogoCleaner
from .raster_card_maker import RasterCardMaker

NODE_CLASS_MAPPINGS = {
    "FaceToMaskCopy": FaceToMaskCopy,
    "RasterCardMaker": RasterCardMaker,
    "PromptLogoCleaner": PromptLogoCleaner,
    "LoadImageAndMaskFromUrl": LoadImageAndMaskFromUrl,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceToMaskCopy": "Face To Mask(Copy)",
    "RasterCardMaker": "Raster Card Maker",
    "PromptLogoCleaner": "Prompt Logo Cleaner (Remove Logo Words)",
    "LoadImageAndMaskFromUrl": "Load Image And Mask From Url",
}
