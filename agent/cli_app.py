import asyncio
from typing import Optional, List
from agent import SearchAgent
from config import AgentConfig
from ollama_utils import OllamaAPI, OllamaModel


class SearchAgentCLI:
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.agent = None
        self.ollama_api = OllamaAPI()
        self.available_models: List[OllamaModel] = []
    
    async def initialize(self):
        print("Получение списка доступных моделей...")
        try:
            self.available_models = await self.ollama_api.get_models()
            
            if not self.available_models:
                print("Не найдено доступных моделей в Ollama")
                return False
            
            print("\nДоступные модели:")
            for idx, model in enumerate(self.available_models, 1):
                mode_str = "(пошаговый режим)" if not model.is_qwen_family else "(ReAct режим)"
                print(f"{idx}. {model.name} {mode_str}")
            
            default_idx = None
            default_name = self.config.default_model.name
            for idx, model in enumerate(self.available_models, 1):
                if model.name == default_name:
                    default_idx = idx
                    break
            
            prompt = f"\nВыберите модель (1-{len(self.available_models)})"
            if default_idx:
                prompt += f" [по умолчанию: {default_idx}]"
            prompt += ": "
            
            while True:
                choice = input(prompt).strip()
                
                if not choice and default_idx:
                    choice = str(default_idx)
                
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(self.available_models):
                        selected_model = self.available_models[idx]
                        break
                    else:
                        print(f"Пожалуйста, введите число от 1 до {len(self.available_models)}")
                except ValueError:
                    print("Пожалуйста, введите корректное число")
            
            self.config.set_model(
                model_name=selected_model.name,
                temperature=0.3,
                family=selected_model.family,
            )
            
            self.agent = SearchAgent(self.config)

            return True
            
        except Exception as e:
            print(f"Ошибка при получении списка моделей: {e}")
            print("Убедитесь, что Ollama запущен и доступен")
            return False
    
    async def run_single_query(self, query: str):
        if not self.agent:
            print("Агент не инициализирован")
            return
            
        print(f"\nОбрабатываю запрос: {query}\nПодождите, идёт поиск информации...\n")
        
        sources = {}
        
        async for chunk in self.agent.asearch(query):
            if chunk["type"] == "content":
                print(chunk["content"], end="", flush=True)
            elif chunk["type"] == "sources":
                sources = chunk["sources"]
            elif chunk["type"] == "error":
                print(f"\n\nОшибка: {chunk['content']}")
        
        if sources:
            print("\n\nИсточники:")
            for idx, (url, info) in enumerate(sorted(sources.items()), 1):
                print(f"[{idx}] {info['title']} - {url}")
        
        print()
    
    async def run_interactive(self):
        if not await self.initialize():
            return
        
        print("\nВведите ваш вопрос или 'q' для выхода.")
        
        while True:
            try:
                query = input(f"\nВаш вопрос: ").strip()
                
                if query.lower() == 'q':
                    break
                
                if not query:
                    continue
                
                await self.run_single_query(query)
            
            except KeyboardInterrupt:
                print("\n\nПрервано пользователем")
                break
            except Exception as e:
                print(f"\nОшибка: {e}")


async def main():
    cli = SearchAgentCLI()
    await cli.run_interactive()


if __name__ == "__main__":
    asyncio.run(main())