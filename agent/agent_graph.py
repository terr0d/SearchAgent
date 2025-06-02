from typing import Dict, Any, List, Literal, AsyncIterator
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from json import loads

from config import config, ModelProvider
from tools import ALL_TOOLS
from state import AgentState


class SearchAgentGraph:
    def __init__(self, agent_config=config):
        self.config = agent_config
        self.llm = self._create_llm()
        self.tools = ALL_TOOLS
        self.graph = self._build_graph()
        
    def _create_llm(self):
        model_config = self.config.default_model
        
        if model_config.provider == ModelProvider.OLLAMA:
            return ChatOllama(
                model=model_config.name,
                temperature=model_config.temperature,
                **model_config.extra_params
            )
        else:
            raise ValueError(f"Провайдер {model_config.provider} пока не поддерживается")
    
    def _build_graph(self):
        g = StateGraph(AgentState)
        
        g.add_node("agent", self._agent_node)
        g.add_node("tools", ToolNode(self.tools))
        
        g.set_entry_point("agent")
        
        g.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": END
            }
        )
        
        g.add_edge("tools", "agent")
        
        return g.compile()
    
    def _create_prompt(self) -> ChatPromptTemplate:
        system_prompt = """/no_think Ты - интеллектуальный поисковый ассистент, который помогает находить и анализировать информацию из интернета.

                        Твоя задача:
                        1. Понять, что именно нужно найти пользователю
                        2. Использовать доступные инструменты для поиска информации
                        3. Проанализировать найденную информацию
                        4. Дать структурированный ответ с указанием источников

                        ВАЖНЫЕ ПРАВИЛА:
                        - Всегда начинай с поиска информации через search_and_extract
                        - Можешь делать несколько поисков для уточнения информации
                        - При необходимости используй calculate для вычислений
                        - Используй get_current_datetime для актуальной временной информации
                        - ОБЯЗАТЕЛЬНО сохраняй все URL источников для последующего цитирования
                        - Отвечай на русском языке

                        Доступные инструменты: {tool_names}"""
        
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])
    
    def _agent_node(self, state: AgentState) -> Dict[str, Any]:
        current_step = state.get("current_step", 0)
        if current_step >= self.config.max_iterations:
            return {
                "should_continue": False,
                "output": "Достигнут лимит итераций. Пожалуйста, попробуйте переформулировать запрос.",
                "error": "Max iterations reached"
            }
        
        prompt = self._create_prompt()
        

        llm_with_tools = self.llm.bind_tools(self.tools)
        
        messages = state.get("messages", [])
        if not messages:
            messages = [HumanMessage(content=state["input"])]
        
        response = llm_with_tools.invoke(
            prompt.format_messages(
                messages=messages,
                tool_names=", ".join([tool.name for tool in self.tools])
            )
        )
        
        return {
            "messages": [response],
            "current_step": current_step + 1,
            "should_continue": len(response.tool_calls) > 0
        }
    
    def _should_continue(self, state: AgentState) -> Literal["continue", "end"]:
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None
        
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "continue"
        return "end"
    
    def _extract_sources_from_messages(self, messages: List[BaseMessage]) -> Dict[str, Dict[str, Any]]:
        sources = {}
        
        for msg in messages:
            if isinstance(msg, ToolMessage):
                try:
                    content = msg.content
                    if isinstance(content, str):
                        content = loads(content)
                    
                    if isinstance(content, dict) and "results" in content:
                        for item in content.get("results", []):
                            if item.get("success") and item.get("url"):
                                url = item["url"]
                                sources[url] = {
                                    "title": item.get("title", "Без названия"),
                                    "url": url
                                }
                except:
                    continue
        
        return sources
    
    async def asearch(self, query: str) -> AsyncIterator[Dict[str, Any]]:
        initial_state = {
            "messages": [],           
            "input": query,
            "output": "",
            "current_step": 0,
            "max_steps": self.config.max_iterations,
            "should_continue": True,
        }
        
        try:
            current_state = initial_state
            
            while True:
                result = await self.graph.ainvoke(current_state)
                current_state = result
                
                messages = result.get("messages", [])
                last_message = messages[-1] if messages else None
                
                if isinstance(last_message, AIMessage) and not last_message.tool_calls:
                    break
            
            sources = self._extract_sources_from_messages(messages)
            
            final_prompt = ChatPromptTemplate.from_messages([
                ("system", """Ты - интеллектуальный поисковый ассистент. 
                На основе проведенного поиска и анализа информации дай подробный и структурированный ответ на вопрос пользователя.
                Отвечай на русском языке."""),
                MessagesPlaceholder(variable_name="messages"),
                ("human", "На основе всей собранной информации, пожалуйста, дай финальный ответ на мой изначальный вопрос: {query}")
            ])
            
            skip_reasoning = True
            reasoning_content = ""
            
            async for chunk in self.llm.astream(
                final_prompt.format_messages(
                    messages=messages,
                    query=query
                )
            ):
                if chunk.content:
                    if skip_reasoning:
                        reasoning_content += chunk.content
                        
                        if '</think>' in reasoning_content:
                            parts = reasoning_content.split('</think>', 1)
                            if len(parts) > 1 and parts[1]:
                                yield {"type": "content", "content": parts[1]}
                            skip_reasoning = False
                    else:
                        yield {"type": "content", "content": chunk.content}
            
            if sources:
                yield {"type": "sources", "sources": sources}
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield {"type": "error", "content": f"Произошла ошибка при обработке запроса: {str(e)}"}