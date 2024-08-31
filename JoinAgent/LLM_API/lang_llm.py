import base64
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
import os
import requests
from pydantic import Field, BaseModel

class TextLLMConfig(BaseModel):
    model: str = "deepseek-coder"
    api_key: str = Field(default_factory=lambda: os.getenv('MULTI_LLM_API'))

class TextLLM(LLM):
    config: TextLLMConfig = Field(default_factory=TextLLMConfig)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        super().__init__()
        self.config = TextLLMConfig(**kwargs)
        if not self.config.api_key:
            raise ValueError("API key not found. Please set the MULTI_LLM_API environment variable.")

    @property
    def _llm_type(self) -> str:
        return "text_llm"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": self.config.model}

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        return self.ask(prompt)

    def ask(self, prompt: str) -> str:
        url = "https://api.openai-next.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."}, 
                {"role": "user", "content": prompt}
            ]
        }
        
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error occurred during the API request: {e}")

class VisionLLMConfig(BaseModel):
    model: str = "gpt-4o-mini"
    api_key: str = Field(default_factory=lambda: os.getenv('MULTI_LLM_API'))

class VisionLLM(LLM):
    config: VisionLLMConfig = Field(default_factory=VisionLLMConfig)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        super().__init__()
        self.config = VisionLLMConfig(**kwargs)
        if not self.config.api_key:
            raise ValueError("API key not found. Please set the MULTI_LLM_API environment variable.")

    @property
    def _llm_type(self) -> str:
        return "vision_llm"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": self.config.model}

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        raise NotImplementedError("VisionLLM requires an image. Use the 'look' method instead.")

    def look(self, image_path: str, prompt: str = "What's in this image?") -> str:
        url = "https://api.openai-next.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}"
        }

        base64_image = self._encode_image(image_path)

        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
        }

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error occurred during the API request: {e}")

    @staticmethod
    def _encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
