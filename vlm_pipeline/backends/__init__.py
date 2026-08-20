from vlm_pipeline.backends.florence2 import Florence2Backend
from vlm_pipeline.backends.got_ocr2 import GotOcr2Backend
from vlm_pipeline.backends.qwen_vl import QwenVLBackend

BACKENDS = {
    "gotocr2": GotOcr2Backend,
    "florence2": Florence2Backend,
    "qwen": QwenVLBackend,
}
