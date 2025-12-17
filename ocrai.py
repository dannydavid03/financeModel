from typing import Optional, Any
import io
import base64
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.utils import Input, Output
from langchain_ollama import ChatOllama
from PIL import Image

from prompt import create_ocr_prompt, create_equity_prompt, create_cash_flow_prompt

class OcrChain(Runnable[Input, Output]):
    def __init__(self, model: str, base_url: str, temperature: float, mode: str = "fixed"):
        self._llm = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
        )
        if mode == "equity":
            self._ocr_prompt = create_equity_prompt()
        elif mode == "cash":
            self._ocr_prompt = create_cash_flow_prompt()
        else:
            self._ocr_prompt = create_ocr_prompt()

    def invoke(self, image_filename: str, config: Optional[RunnableConfig] = None, **kwargs: Any) -> str:
        image_data = self._read_image(image_filename)
        input_data = {"image_data": image_data}
        return self._create_chain().invoke(input_data, config, **kwargs).content

    def _create_chain(self) -> Runnable:
        return self._ocr_prompt | self._llm

    def _read_image(self, image_filename: str) -> str:
        try:
            file = Image.open(image_filename)
            if file.mode in ('RGBA', 'P'): 
                file = file.convert('RGB')
                
            w, h = file.size
            
            # Target 2000px width for better text resolution
            TARGET_WIDTH = 2000
            if w < TARGET_WIDTH:
                scale_factor = TARGET_WIDTH / w
                new_size = (int(w * scale_factor), int(h * scale_factor))
                file = file.resize(new_size, Image.Resampling.LANCZOS)
                
            buf = io.BytesIO()
            file.save(buf, format="PNG") 
            return base64.b64encode(buf.getvalue()).decode("utf-8")
            
        except Exception as e:
            raise RuntimeError(f"Failed to process image: {e}")