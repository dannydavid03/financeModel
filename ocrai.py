from typing import Optional, Any
import io
import base64
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.utils import Input, Output
from langchain_ollama import ChatOllama
from PIL import Image

from prompt import create_ocr_prompt

class OcrChain(Runnable[Input, Output]):
    def __init__(self, model: str, base_url: str, temperature: float):
        # Standard text generation (no JSON mode, no Pydantic)
        self._llm = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
        )
        self._ocr_prompt = create_ocr_prompt()

    def invoke(self, image_filename: str, config: Optional[RunnableConfig] = None, **kwargs: Any) -> str:
        image_data = self._read_image(image_filename)
        input_data = {"image_data": image_data}
        
        # Returns the raw Markdown string
        return self._create_chain().invoke(input_data, config, **kwargs).content

    def _create_chain(self) -> Runnable:
        return self._ocr_prompt | self._llm

    def _read_image(self, image_filename: str) -> str:
        file = Image.open(image_filename)
        if file.mode in ('RGBA', 'P'): 
            file = file.convert('RGB')
            
        # CRITICAL: Upscale image to help model see small "Note" numbers
        w, h = file.size
        if w < 1000:
            new_size = (w * 2, h * 2)
            file = file.resize(new_size, Image.Resampling.LANCZOS)
            
        buf = io.BytesIO()
        file.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")