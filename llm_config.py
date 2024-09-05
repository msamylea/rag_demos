import os
from dotenv import load_dotenv
from pathlib import Path
import asyncio
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List, Union
import google.generativeai as genai
from huggingface_hub import InferenceClient
from openai import AsyncOpenAI, OpenAI
from PIL import Image

path = Path(__file__).parent / ".env"   
load_dotenv(dotenv_path=path)

class LLMConfig:
    def __init__(self, provider: str, model: str, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
        self.provider = provider.lower()
        self.model = model
        self.base_url = base_url or self._get_base_url()
        self.params = kwargs
        self.api_key = self._get_api_key(api_key)

    def _get_api_key(self, provided_key: Optional[str]) -> str:
        if provided_key:
            return provided_key
        if self.provider in ["ollama", "llamacpp", "litellm"]:
            return "not_required"
        env_var_map = {
            "openai": "OPENAI_API_KEY",
            "huggingface": "HF_TOKEN",
            "huggingface-openai": "HF_TOKEN",
            "huggingface-text": "HF_TOKEN",
            "gemini": "GENAI_API_KEY",
            "sdxl": "HF_TOKEN",
              }
        env_var = env_var_map.get(self.provider)
        api_key = os.environ.get(env_var) if env_var else None
        if not api_key:
            raise ValueError(f"API key for {self.provider} is not set. Please set the appropriate environment variable or provide it directly.")
        return api_key

    def _get_base_url(self) -> Optional[str]:
        if self.provider == "ollama":
            return "http://localhost:11434/v1"
        return None

class BaseLLM(ABC):
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = self._create_client()

    @abstractmethod
    def _create_client(self):
        pass

    @abstractmethod
    def get_response(self, prompt: str) -> Any:
        pass

    @abstractmethod
    async def get_aresponse(self, prompt: str) -> Any:
        pass

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": self.config.provider,
            "model": self.config.model,
            "parameters": self.config.params
        }
class OpenAILLM(BaseLLM):
    def _create_client(self):
        self.sync_client = OpenAI(base_url=self.config.base_url, api_key=self.config.api_key)
        self.async_client = AsyncOpenAI(base_url=self.config.base_url, api_key=self.config.api_key)

    def get_response(self, prompt: str) -> str:
        response = self.sync_client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            **self.config.params
        )
        return response.choices[0].message.content

    async def get_aresponse(self, prompt: str):
        stream = await self.async_client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            **self.config.params
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

class GeminiLLM(BaseLLM):
    def _create_client(self):
        genai.configure(api_key=self.config.api_key)
        return genai.GenerativeModel(model_name=self.config.model)

    def _prepare_content(self, prompt: Union[str, List[Union[str, Image.Image]]]) -> Union[str, List[Union[str, Image.Image]]]:
        if isinstance(prompt, str):
            return prompt
        elif isinstance(prompt, list):
            return [item if isinstance(item, (str, Image.Image)) else str(item) for item in prompt]
        else:
            return str(prompt)

    def get_response(self, prompt: Union[str, List[Union[str, Image.Image]]]) -> str:
        generation_config = genai.GenerationConfig(**{k: v for k, v in self.config.params.items() if k in ['temperature', 'max_output_tokens', 'top_p', 'top_k']})
        content = self._prepare_content(prompt)
        response = self.client.generate_content(content, generation_config=generation_config)
        response.resolve()
        return response.text

    async def get_aresponse(self, prompt: Union[str, List[Union[str, Image.Image]]]):
        generation_config = genai.GenerationConfig(**{k: v for k, v in self.config.params.items() if k in ['temperature', 'max_output_tokens', 'top_p', 'top_k']})
        content = self._prepare_content(prompt)
        response = self.client.generate_content(content, generation_config=generation_config, stream=True)
        for chunk in response:
            yield chunk.text
            await asyncio.sleep(0.01)

class SDXLLLM(BaseLLM):
    
    def _create_client(self):
        return InferenceClient(model=self.config.model, token=self.config.api_key)

    def _generate_filename(self, prompt: str) -> str:
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Use the first few words of the prompt, removing any non-alphanumeric characters
        prompt_part = "".join(c for c in prompt[:30] if c.isalnum() or c.isspace()).rstrip()
        prompt_part = prompt_part.replace(" ", "_")
        return f"{timestamp}_{prompt_part}.jpg"

    def get_response(self, prompt: str, save_dir: str = "./generated_images") -> str:
        try:
            # Ensure the save directory exists
            os.makedirs(save_dir, exist_ok=True)

            image = self.client.text_to_image(prompt, **self.config.params)
            if isinstance(image, Image.Image):
                filename = self._generate_filename(prompt)
                image_path = os.path.join(save_dir, filename)
                image.save(image_path)
                return f"Image saved as {image_path}"
            else:
                return "Failed to generate image"
        except Exception as e:
            return f"Error generating image: {str(e)}"

    async def get_aresponse(self, prompt: str, save_dir: str = "./generated_images"):
        # SDXL doesn't support async streaming, so we'll return the full response
        yield self.get_response(prompt, save_dir)

class HFOpenAIAPILLM(BaseLLM):
    def _create_client(self):
        base_url = f"https://api-inference.huggingface.co/models/{self.config.model}/v1/"
        self.sync_client = OpenAI(base_url=base_url, api_key=self.config.api_key)
        self.async_client = AsyncOpenAI(base_url=base_url, api_key=self.config.api_key)

    
    def get_response(self, prompt: str, **kwargs) -> str:
        response = self.sync_client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            **self.config.params
        )
        return response.choices[0].message.content

    async def get_aresponse(self, prompt: str):
        stream = await self.async_client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            **self.config.params
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

class Ollama(BaseLLM):
    def _create_client(self):
        self.sync_client = OpenAI(base_url=self.config.base_url, api_key="ollama")
        self.async_client = AsyncOpenAI(base_url=self.config.base_url, api_key="ollama")

    def get_response(self, prompt: str) -> str:
        response = self.sync_client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            **self.config.params
        )
        return response.choices[0].message.content

    async def get_aresponse(self, prompt: str):
        stream = await self.async_client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            **self.config.params
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

class HFTextLLM(BaseLLM):
    def _create_client(self):
        return InferenceClient(model=self.config.model, token=self.config.api_key)

    def get_response(self, prompt: str) -> str:
        parameters = {k: v for k, v in self.config.params.items() if k in ['temperature', 'max_new_tokens', 'top_p', 'top_k', 'stream', 'grammar']}
        response = self.client.text_generation(prompt, **parameters)
        if 'tools' in self.config.params:
            return response.choices[0].message.tool_calls[0].function
        else:
            return response

    async def get_aresponse(self, prompt: str):
        parameters = {k: v for k, v in self.config.params.items() if k in ['temperature', 'max_new_tokens', 'top_p', 'top_k', 'tools', 'tool_choice', 'tool_prompt']}
        parameters['stream'] = True
        async for response in self.client.text_generation(prompt, **parameters, stream=True):
            yield response


class LLMFactory:
    @staticmethod
    def create_llm(config: LLMConfig) -> BaseLLM:
        llm_classes = {
            "openai": OpenAILLM,
            "gemini": GeminiLLM,
            "sdxl": SDXLLLM,
            "huggingface-openai": HFOpenAIAPILLM,
            "huggingface-text": HFTextLLM,
            "ollama": Ollama,
        }
        if config.provider not in llm_classes:
            raise ValueError(f"Unsupported provider: {config.provider}")
        return llm_classes[config.provider](config)

def get_llm(provider: str, model: str, **kwargs) -> BaseLLM:
    config = LLMConfig(provider, model, **kwargs)
    return LLMFactory.create_llm(config)
# Utility functions

def batch_process(llm: BaseLLM, prompts: List[str]) -> List[str]:
    """Process a batch of prompts and return their responses."""
    return [llm.get_response(prompt) for prompt in prompts]

async def batch_process_async(llm: BaseLLM, prompts: List[str]) -> List[str]:
    """Process a batch of prompts asynchronously and return their responses."""
    async def process_prompt(prompt):
        result = ""
        async for chunk in llm.get_aresponse(prompt):
            result += chunk
        return result
    
    return await asyncio.gather(*[process_prompt(prompt) for prompt in prompts])

def compare_responses(llms: List[BaseLLM], prompt: str) -> Dict[str, str]:
    """Compare responses from multiple LLMs for the same prompt."""
    return {llm.get_model_info()['model']: llm.get_response(prompt) for llm in llms}

async def stream_to_file(llm: BaseLLM, prompt: str, filename: str):
    """Stream the LLM response to a file."""
    with open(filename, 'w') as f:
        async for chunk in llm.get_aresponse(prompt):
            f.write(chunk)
            f.flush()

