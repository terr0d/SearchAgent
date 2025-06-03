import aiohttp
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class OllamaModel:
    name: str
    size: int
    parameter_size: Optional[str] = None
    family: Optional[str] = None
    quantization_level: Optional[str] = None
    
    @property
    def size_gb(self) -> float:
        return self.size / (1024 ** 3)
    
    @property
    def is_qwen_family(self) -> bool:
        if self.family and 'qwen3' in self.family.lower():
            return True
        return False


class OllamaAPI:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
    
    async def get_models(self) -> List[OllamaModel]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        models = []
                        
                        for model_data in data.get("models", []):
                            details = model_data.get("details", {})
                            model = OllamaModel(
                                name=model_data["name"],
                                size=model_data.get("size", 0),
                                parameter_size=details.get("parameter_size"),
                                family=details.get("family"),
                                quantization_level=details.get("quantization_level")
                            )
                            models.append(model)
                        
                        return sorted(models, key=lambda m: m.name)
                    else:
                        raise Exception(f"Ошибка при получении списка моделей: HTTP {response.status}")
        except aiohttp.ClientError as e:
            raise Exception(f"Не удалось подключиться к Ollama API: {e}")