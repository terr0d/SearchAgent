import asyncio
import aiohttp
from typing import TypedDict, List, Dict, Optional
import trafilatura
from duckduckgo_search import DDGS
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
import sys


class AgentState(TypedDict):
    query: str
    search_results: Optional[List[Dict]]
    page_contents: Optional[Dict[str, str]]
    page_summaries: Optional[Dict[str, str]]
    final_summary: Optional[str]
    error: Optional[str]
    needs_clarification: Optional[bool]


llm = ChatOllama(model="qwen3:30b-a3b", temperature=0.3)


def _strip_think(text: str) -> str:
    return text.replace("<think>", "").replace("</think>", "")


async def _run_ddg(query: str, max_results: int = 5) -> List[Dict]:
    def _search(q):
        with DDGS() as ddgs:
            return list(ddgs.text(q, max_results=max_results))
    return await asyncio.to_thread(_search, query)


async def _fetch_single(session: aiohttp.ClientSession, url: str) -> tuple[str, str]:
    try:
        async with session.get(url, timeout=15) as r:
            html = await r.text()
        text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
        ) or ""
        if len(text) > 10000:
            text = text[:10000]
        return url, text or "Не удалось извлечь текст со страницы."
    except Exception as e:
        return url, f"Ошибка при извлечении содержимого: {e}"


async def format_search_query(state: AgentState) -> AgentState:
    query = state["query"].strip()
    if not query:
        return {**state, "needs_clarification": True, "error": "Запрос пустой."}
    prompt = f"""/no_think
                Ты – поисковой оптимизатор. 
                Задача: превратить вопрос пользователя в ОДНУ короткую поисковую фразу (3-8 слов), убрав лишние стоп-слова.
                Отвечай всегда на русском, без кавычек.

                Вопрос: {query}
                Выведи только поисковый запрос:"""
    response = await llm.ainvoke(prompt)
    search_query = _strip_think(response.content).strip()
    print(f"[DEBUG] Сформированный запрос:{search_query}\n")
    return {**state, "query": search_query}


async def search_duckduckgo(state: AgentState) -> AgentState:
    try:
        results = await _run_ddg(state["query"])
        filtered = [{"href": r.get("href", "")} for r in results]
        print(f"[DEBUG] Найдено ссылок:{filtered}\n")
        return {**state, "search_results": filtered}
    except Exception as e:
        return {**state, "error": f"Ошибка при поиске: {e}"}


async def extract_page_content(state: AgentState) -> AgentState:
    search_results = state.get("search_results") or []
    if not search_results:
        return {**state, "error": "Нет результатов поиска."}
    urls = [r["href"] for r in search_results if r.get("href")]
    async with aiohttp.ClientSession() as session:
        tasks = [_fetch_single(session, u) for u in urls]
        page_contents = dict(await asyncio.gather(*tasks))
    return {**state, "page_contents": page_contents}


async def summarize_pages(state: AgentState) -> AgentState:
    page_contents = state.get("page_contents") or {}
    original_query = state["query"]
    page_summaries: Dict[str, str] = {}
    for url, content in page_contents.items():
        if content.startswith(("Ошибка", "Не удалось")):
            page_summaries[url] = content
            continue
        prompt = f"""/no_think
                    Ты – аналитик, умеющий делать краткие саммари.
                    Всегда отвечай на русском.
                    Если текст нерелевантен вопросу, ответь: "Нерелевантный источник".
                    Вопрос: {original_query}

                    Текст:
                    {content}

                    Резюме:"""
        try:
            resp = await llm.ainvoke(prompt)
            summary = _strip_think(resp.content).strip()
            page_summaries[url] = summary
            print(f"[DEBUG] Резюме {url}:\n{summary}\n")
        except Exception as e:
            page_summaries[url] = f"Ошибка при создании резюме: {e}"
    return {**state, "page_summaries": page_summaries}


async def create_final_summary(state: AgentState) -> AgentState:
    page_summaries = state.get("page_summaries") or {}
    if not page_summaries:
        return {**state, "error": "Не удалось создать резюме страниц."}
    original_query = state["query"]
    joined = "\n\n".join(f"Источник: {u}\n{s}" for u, s in page_summaries.items())
    prompt = f"""/no_think
                Ты – помощник, который собирает сводный ответ из нескольких источников.
                В конце сделай раздел "Источники" в формате [N] URL, а в тексте – ссылки на них.
                Если мнения расходятся – отрази все точки зрения.
                Не добавляй личных рассуждений.
                
                Вопрос: {original_query}
                
                Резюме источников:
                {joined}
                
                Ответ:"""
    try:
        final_parts: List[str] = []
        print("\nОтвет:\n")
        started = False

        async for chunk in llm.astream(prompt):
            token = _strip_think(chunk.content)

            if not started:
                if token.strip() == "":
                    continue
                started = True
                token = token.lstrip()

            print(token, end="", flush=True)
            final_parts.append(token)

        print("")
        return {**state, "final_summary": "".join(final_parts).strip()}
    except Exception as e:
        return {**state, "error": f"Ошибка при финальном резюме: {e}"}


def should_continue(state: AgentState) -> str:
    if state.get("needs_clarification"):
        return "clarify"
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


def build_agent_graph():
    g = StateGraph(AgentState)
    g.add_node("format_query", format_search_query)
    g.add_node("search", search_duckduckgo)
    g.add_node("extract", extract_page_content)
    g.add_node("summarize_pages", summarize_pages)
    g.add_node("create_final_summary", create_final_summary)
    g.set_entry_point("format_query")
    g.add_conditional_edges("format_query", should_continue, {"clarify": END, "search": "search", "error": END})
    g.add_conditional_edges("search", should_continue, {"extract": "extract", "error": END})
    g.add_conditional_edges("extract", should_continue, {"summarize_pages": "summarize_pages", "error": END})
    g.add_conditional_edges("summarize_pages", should_continue, {"create_final_summary": "create_final_summary", "error": END})
    g.add_conditional_edges("create_final_summary", should_continue, {"end": END, "error": END})
    return g.compile()


async def run_agent(query: str):
    agent = build_agent_graph()
    print(f"\nОбрабатываю запрос: {query}\nПодождите, идет поиск информации...\n")
    result: AgentState = await agent.ainvoke({"query": query})
    if result.get("needs_clarification"):
        print("Запрос неясен. Пожалуйста, уточните вопрос.")
    elif result.get("error"):
        print(f"Произошла ошибка: {result['error']}")


async def main():
    if len(sys.argv) > 1:
        await run_agent(" ".join(sys.argv[1:]))
        return
    print("Введите ваш вопрос или 'q' для выхода.")
    while True:
        query = input("\nВаш вопрос: ").strip()
        if query.lower() == "q":
            break
        if not query:
            print("Пожалуйста, введите вопрос.")
            continue
        await run_agent(query)


if __name__ == "__main__":
    asyncio.run(main())