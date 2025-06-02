import asyncio
from typing import Optional
from agent_graph import SearchAgentGraph
from config import AgentConfig


class SearchAgentCLI:
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.agent = SearchAgentGraph(self.config)
    
    async def run_single_query(self, query: str):
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
        print("Введите ваш вопрос или 'q' для выхода.")
        
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