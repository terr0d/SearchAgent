import asyncio
import aiohttp
from typing import TypedDict, List, Dict, Optional, AsyncIterator, Any, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
import trafilatura
from duckduckgo_search import DDGS
from json import loads
import time
from asyncio import Semaphore

from config import config as agent_config, ModelProvider
from tools import ALL_TOOLS


class AgentState(TypedDict):
    input: str
    output: str
    error: Optional[str]
    mode: str
    
    # Для ReAct режима
    messages: List[BaseMessage]
    current_step: int
    max_steps: int
    should_continue: bool
    
    # Для пошагового режима
    query: str
    search_results: Optional[List[Dict]]
    page_contents: Optional[Dict[str, str]]
    page_summaries: Optional[Dict[str, str]]
    final_summary: Optional[str]
    needs_clarification: Optional[bool]
    sources_info: Optional[Dict[str, Dict]]


_ddgs_instance = None
_last_search_time = 0
_search_semaphore = Semaphore(1)
MIN_SEARCH_INTERVAL = 2.0


class SearchAgent:
    
    def __init__(self, config=agent_config):
        self.config = config
        self.llm = self._create_llm()
        self.tools = ALL_TOOLS
        self.is_qwen_family = self.config.default_model.is_qwen_family
        self.graph = self._build_unified_graph()
    
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
    
    def _build_unified_graph(self):
        g = StateGraph(AgentState)
        
        g.add_node("route", self._route_node)
        
        # ReAct ветка
        g.add_node("react_agent", self._react_agent_node)
        g.add_node("tools", ToolNode(self.tools))
        
        # Пошаговая ветка
        g.add_node("format_query", self._format_query_node)
        g.add_node("search", self._search_node)
        g.add_node("extract", self._extract_node)
        g.add_node("summarize_pages", self._summarize_pages_node)
        g.add_node("create_final_summary", self._create_final_summary_node)
        
        g.add_node("finalize", self._finalize_node)
        
        g.set_entry_point("route")
        
        g.add_conditional_edges(
            "route",
            lambda x: x["mode"],
            {
                "react": "react_agent",
                "step_by_step": "format_query"
            }
        )
        
        g.add_conditional_edges(
            "react_agent",
            self._should_continue_react,
            {
                "continue": "tools",
                "end": "finalize"
            }
        )
        g.add_edge("tools", "react_agent")
        
        g.add_conditional_edges(
            "format_query",
            self._should_continue_step_by_step,
            {
                "search": "search",
                "error": "finalize"
            }
        )
        
        g.add_conditional_edges(
            "search",
            self._should_continue_step_by_step,
            {
                "extract": "extract",
                "error": "finalize"
            }
        )
        
        g.add_conditional_edges(
            "extract",
            self._should_continue_step_by_step,
            {
                "summarize_pages": "summarize_pages",
                "error": "finalize"
            }
        )
        
        g.add_conditional_edges(
            "summarize_pages",
            self._should_continue_step_by_step,
            {
                "create_final_summary": "create_final_summary",
                "error": "finalize"
            }
        )
        
        g.add_edge("create_final_summary", "finalize")
        
        g.add_edge("finalize", END)
        
        return g.compile()
    
    def _route_node(self, state: AgentState) -> AgentState:
        mode = "react" if self.is_qwen_family else "step_by_step"
        print(f"[DEBUG] Выбран режим: {mode}")
        
        return {
            **state,
            "mode": mode,
            "query": state["input"], 
            "messages": [HumanMessage(content=state["input"])],
            "current_step": 0,
            "max_steps": self.config.max_iterations,
            "sources_info": {}
        }
    
    def _react_agent_node(self, state: AgentState) -> AgentState:
        current_step = state.get("current_step", 0)
        if current_step >= self.config.max_iterations:
            return {
                **state,
                "should_continue": False,
                "error": "Достигнут лимит итераций"
            }
        
        prompt = self._create_react_prompt()
        llm_with_tools = self.llm.bind_tools(self.tools)
        
        messages = state.get("messages", [])
        
        response = llm_with_tools.invoke(
            prompt.format_messages(
                messages=messages,
                tool_names=", ".join([tool.name for tool in self.tools])
            )
        )
        
        return {
            **state,
            "messages": messages + [response],
            "current_step": current_step + 1,
            "should_continue": len(response.tool_calls) > 0
        }
    
    def _should_continue_react(self, state: AgentState) -> Literal["continue", "end"]:
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None
        
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "continue"
        return "end"
    
    def _create_react_prompt(self) -> ChatPromptTemplate:
        system_prompt = """/no_think Ты - интеллектуальный поисковый ассистент, который помогает находить и анализировать информацию из интернета.

                        Твоя задача:
                        1. Понять, что именно нужно найти пользователю
                        2. Использовать доступные инструменты для поиска информации
                        3. Проанализировать найденную информацию
                        4. Дать структурированный ответ с указанием источников

                        ВАЖНЫЕ ПРАВИЛА:
                        - Должен быть как минимум один вызов инструмента.
                        - Всегда начинай с поиска информации через search_and_extract.
                        - Если вопрос пользователя непонятен, уточни, что он имеет в виду 
                        - Можешь делать несколько поисков для уточнения информации
                        - При необходимости используй calculate для вычислений
                        - Используй get_current_datetime для актуальной временной информации
                        - Отвечай на русском языке

                        Доступные инструменты: {tool_names}"""
        
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])
    
    def _strip_think(self, text: str) -> str:
        return text.replace("<think>", "").replace("</think>", "")
    
    def _get_ddgs(self):
        global _ddgs_instance
        if _ddgs_instance is None:
            _ddgs_instance = DDGS()
        return _ddgs_instance
    
    def _jina_ai_url(self, url: str) -> str:
        return "https://r.jina.ai/" + url.replace("https://", "").replace("http://", "")
    
    async def _run_ddg(self, query: str, max_results: int = None) -> List[Dict]:
        if max_results is None:
            max_results = self.config.max_search_results
            
        async with _search_semaphore:
            global _last_search_time
            
            current_time = time.time()
            time_since_last = current_time - _last_search_time
            
            if time_since_last < MIN_SEARCH_INTERVAL:
                await asyncio.sleep(MIN_SEARCH_INTERVAL - time_since_last)
            
            def _search(q):
                ddgs = self._get_ddgs()
                return list(ddgs.text(q, max_results=max_results))
            
            result = await asyncio.to_thread(_search, query)
            _last_search_time = time.time()
            
            return result
    
    async def _format_query_node(self, state: AgentState) -> AgentState:
        query = state["query"].strip()
        if not query:
            return {**state, "needs_clarification": True}

        prompt = f"""
                    Ты – эксперт по поисковым запросам.
                    Если приведённый текст является понятным вопросом, преобразуй его в одну поисковую фразу (3-8 слов) без стоп-слов.
                    Если это не вопрос или он непонятен, ответь строго: NEEDS_CLARIFICATION.
                    Отвечай без кавычек.

                    Ввод: {query}

                    Ответ:"""
        response = await self.llm.ainvoke(prompt)
        answer = self._strip_think(response.content).strip()

        if answer.upper() == "NEEDS_CLARIFICATION":
            return {**state, "needs_clarification": True, "error": "Запрос непонятен"}

        print(f"Поиск: {answer}, количество запросов: {self.config.max_search_results}\n")
        return {**state, "query": answer, "needs_clarification": False}
    
    async def _search_node(self, state: AgentState) -> AgentState:
        """Выполняет поиск и сохраняет полную информацию"""
        try:
            results = await self._run_ddg(state["query"])
            
            sources_info = {}
            for r in results:
                url = r.get("href", "")
                if url:
                    sources_info[url] = {
                        "title": r.get("title", "Без названия"),
                        "snippet": r.get("body", ""),
                        "url": url
                    }
            
            return {
                **state, 
                "search_results": results,
                "sources_info": sources_info
            }
        except Exception as e:
            return {**state, "error": f"Ошибка при поиске: {e}"}
    
    async def _extract_node(self, state: AgentState) -> AgentState:
        search_results = state.get("search_results") or []
        if not search_results:
            return {**state, "error": "Нет результатов поиска."}
        
        urls = [r.get("href", "") for r in search_results if r.get("href")]
        
        async def _fetch_single(session: aiohttp.ClientSession, url: str) -> tuple[str, str]:
            try:
                timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
                async with session.get(url, timeout=timeout) as r:
                    html = await r.text()
                text = trafilatura.extract(
                    html,
                    include_comments=False,
                    include_tables=False,
                ) or ""
                if len(text) > self.config.max_content_length:
                    text = text[:self.config.max_content_length]
                return url, text or "Не удалось извлечь текст со страницы."
            except Exception as e:
                return url, f"Ошибка при извлечении содержимого: {e}"
        
        async with aiohttp.ClientSession() as session:
            tasks = [_fetch_single(session, u) for u in urls]
            page_contents = dict(await asyncio.gather(*tasks))
        
        return {**state, "page_contents": page_contents}
    
    async def _summarize_pages_node(self, state: AgentState) -> AgentState:
        page_contents = state.get("page_contents") or {}
        original_query = state["query"]
        page_summaries: Dict[str, str] = {}
        
        async with aiohttp.ClientSession() as session:
            for url, content in page_contents.items():
                if content.startswith(("Ошибка", "Не удалось")):
                    page_summaries[url] = content
                    continue
                    
                prompt = f"""
                            Ты – аналитик, умеющий делать краткие саммари.
                            Всегда отвечай на русском.
                            Если текст нерелевантен вопросу, ответь: "Нерелевантный источник".
                            Вопрос: {original_query}

                            Текст:
                            {content}

                            Резюме:"""
                try:
                    resp = await self.llm.ainvoke(prompt)
                    summary = self._strip_think(resp.content).strip()

                    if "нерелевантный источник" in summary.lower():
                        print(f"[DEBUG] Пробую Jina для {url}")
                        jina_url = self._jina_ai_url(url)
                        
                        try:
                            timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
                            async with session.get(jina_url, timeout=timeout) as jr:
                                jina_text = await jr.text()
                            
                            jina_prompt = f"""
                                           Ты – аналитик, умеющий делать краткие саммари.
                                           Всегда отвечай на русском.
                                           Если текст нерелевантен вопросу, ответь: "Нерелевантный источник".
                                           Вопрос: {original_query}

                                           Текст:
                                           {jina_text}

                                           Резюме:"""
                            resp = await self.llm.ainvoke(jina_prompt)
                            summary = self._strip_think(resp.content).strip()
                        except Exception as e:
                            print(f"[DEBUG] Ошибка Jina для {url}: {e}")

                    page_summaries[url] = summary
                except Exception as e:
                    page_summaries[url] = f"Ошибка при создании резюме: {e}"
        
        return {**state, "page_summaries": page_summaries}
    
    async def _create_final_summary_node(self, state: AgentState) -> AgentState:
        page_summaries = state.get("page_summaries") or {}
        sources_info = state.get("sources_info") or {}
        
        if not page_summaries:
            return {**state, "error": "Не удалось создать резюме страниц."}
        
        original_query = state["query"]
        
        relevant_summaries = {}
        for url, summary in page_summaries.items():
            if not summary.startswith(("Ошибка", "Не удалось")) and "нерелевантный источник" not in summary.lower():
                relevant_summaries[url] = summary
        
        if not relevant_summaries:
            return {**state, "error": "Не найдено релевантных источников."}
        
        joined = "\n\n".join(f"Источник: {url}\n{summary}" for url, summary in relevant_summaries.items())
        
        prompt = f"""
                    Ты – помощник, который собирает общее саммари из нескольких источников.
                    НЕ добавляй раздел "Источники" в конце - он будет добавлен автоматически.
                    Если мнения расходятся – отрази все точки зрения.
                    Не добавляй личных рассуждений.
                    
                    Вопрос: {original_query}
                    
                    Резюме источников:
                    {joined}
                    
                    Ответ:"""
        try:
            final_parts: List[str] = []
            started = False

            async for chunk in self.llm.astream(prompt):
                token = self._strip_think(chunk.content)

                if not started and token.strip() == "":
                    continue
                started = True
                final_parts.append(token)

            final_text = "".join(final_parts).strip()
            
            filtered_sources = {url: info for url, info in sources_info.items() if url in relevant_summaries}
            
            return {
                **state, 
                "final_summary": final_text,
                "sources_info": filtered_sources
            }
        except Exception as e:
            return {**state, "error": f"Ошибка при финальном резюме: {e}"}
    
    def _should_continue_step_by_step(self, state: AgentState) -> str:
        if state.get("needs_clarification"):
            return "error"
        if state.get("error"):
            return "error"
        if state.get("final_summary"):
            return "end"
        if not state.get("search_results"):
            return "search"
        if not state.get("page_contents"):
            return "extract"
        if not state.get("page_summaries"):
            return "summarize_pages"
        return "create_final_summary"
    

    def _finalize_node(self, state: AgentState) -> AgentState:
        if state["mode"] == "react":
            messages = state.get("messages", [])
            sources = self._extract_sources_from_messages(messages)
            
            last_ai_message = None
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and not msg.tool_calls:
                    last_ai_message = msg
                    break
            
            output = last_ai_message.content if last_ai_message else ""
            
            return {
                **state,
                "output": output,
                "final_summary": output,
                "sources_info": sources
            }
        else:
            return {
                **state,
                "output": state.get("final_summary", "")
            }
    
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
        initial_state: AgentState = {
            "input": query,
            "output": "",
            "error": None,
            "mode": "",
            
            "messages": [],
            "current_step": 0,
            "max_steps": self.config.max_iterations,
            "should_continue": True,
            
            "query": query,
            "search_results": None,
            "page_contents": None,
            "page_summaries": None,
            "final_summary": None,
            "needs_clarification": None,
            "sources_info": None
        }
        
        try:
            result = await self.graph.ainvoke(initial_state)
            
            if result.get("error"):
                yield {"type": "error", "content": result["error"]}
                return
            
            if result["mode"] == "react":
                messages = result.get("messages", [])
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
                    
            else:
                final_text = result.get("final_summary", "")
                sources_info = result.get("sources_info", {})
                
                for char in final_text:
                    yield {"type": "content", "content": char}
                    await asyncio.sleep(0.001)
                
                if sources_info:
                    yield {"type": "sources", "sources": sources_info}
                        
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield {"type": "error", "content": f"Произошла ошибка при обработке запроса: {str(e)}"}