# -*- coding: utf-8 -*-

from .py.aliyun_face_beauty import AliyunFaceBeautyNode
from .py.auto_gamma import AutoGamma
from .py.bimoai_segment_node import BimoAISegmentImage
from .py.chroma_key import ChromaKeyNode
from .py.corner_pin import WEB_DIRECTORY, BIMO_CornerPinPerspective
from .py.doubao import DoubaoSingleTurnChatNodeSDKv2
from .py.garment_category import GarmentCategoryMapper, GarmentCategoryMapperBatch
from .py.gemini_image_node_executor import GeminiImageGenerateExecutor
from .py.gpt_image_2 import GPTImage2Generator
from .py.seedream_concurrent import SeedreamImageGenerateConcurrent
from .py.seedream_node_executor import SeedreamImageGenerateExecutor

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]


NODE_CLASS_MAPPINGS = {
    "AliyunFaceBeauty": AliyunFaceBeautyNode,
    "AutoGamma": AutoGamma,
    "BIMO_CornerPinPerspective": BIMO_CornerPinPerspective,
    "BimoAISegmentImage": BimoAISegmentImage,
    "ChromaKey": ChromaKeyNode,
    "DoubaoSingleTurnChatNodeSDKv2": DoubaoSingleTurnChatNodeSDKv2,
    "GarmentCategoryMapper": GarmentCategoryMapper,
    "GarmentCategoryMapperBatch": GarmentCategoryMapperBatch,
    "GeminiImageGenerate": GeminiImageGenerateExecutor,
    "GPTImage2Generator": GPTImage2Generator,
    "SeedreamImageGenerateConcurrent": SeedreamImageGenerateConcurrent,
    "SeedreamImageGenerateExecutor": SeedreamImageGenerateExecutor,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "AliyunFaceBeauty": "Aliyun Face Beauty (Retouch)",
    "AutoGamma": "Auto Gamma",
    "BIMO_CornerPinPerspective": "Corner Pin / Perspective Warp",
    "BimoAISegmentImage": "BimoAI Image Segment",
    "ChromaKey": "Chroma Key",
    "DoubaoSingleTurnChatNodeSDKv2": "Doubao Chat (Single Turn, Ark SDK)",
    "GarmentCategoryMapper": "Garment Category Mapper (1/2/3)",
    "GarmentCategoryMapperBatch": "Garment Category Mapper (Batch)",
    "GeminiImageGenerate": "Gemini Image Generator",
    "GPTImage2Generator": "OpenAI GPT Image 2",
    "SeedreamImageGenerateConcurrent": "Seedream Image Generate (Concurrent)",
    "SeedreamImageGenerateExecutor": "Seedream Image Generate Executor",
}
