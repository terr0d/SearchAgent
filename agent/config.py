from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum


class ModelProvider(str, Enum):
    OLLAMA = "ollama"


@dataclass
class ModelConfig:
    name: str
    provider: ModelProvider
    temperature: float = 0.3
    max_tokens: Optional[int] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)
    family: Optional[str] = None
    
    @property
    def is_qwen_family(self) -> bool:
        if self.family and 'qwen3' in self.family.lower():
            return True
        return False


@dataclass
class AgentConfig:
    default_model: ModelConfig = field(default_factory=lambda: ModelConfig(
        name="qwen3:8b",
        provider=ModelProvider.OLLAMA,
        temperature=0.3
    ))
    
    max_iterations: int = 10
    max_search_results: int = 5
    max_content_length: int = 10000
    request_timeout: int = 8 
    
    ollama_base_url: str = "http://localhost:11434"
    
    available_models: Dict[str, ModelConfig] = field(default_factory=dict)
    
    def get_model(self, model_name: Optional[str] = None) -> ModelConfig:
        if model_name and model_name in self.available_models:
            return self.available_models[model_name]
        return self.default_model
    
    def set_model(self, model_name: str, temperature: float = 0.3, family: Optional[str] = None):
        self.default_model = ModelConfig(
            name=model_name,
            provider=ModelProvider.OLLAMA,
            temperature=temperature,
            family=family
        )


config = AgentConfig()