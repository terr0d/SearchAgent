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


llm = ChatOllama(model="qwen3:8b", temperature=0.3)


def _strip_think(text: str) -> str:
    return text.replace("<think>", "").replace("</think>", "")


async def _run_ddg(query: str, max_results: int = 5) -> List[Dict]:
    def _search(q):
        with DDGS() as ddgs:
            return list(ddgs.text(q, max_results=max_results))
    return await asyncio.to_thread(_search, query)


def _jina_ai_url(url: str) -> str:
    return "https://r.jina.ai/http://" + url.replace("https://", "").replace("http://", "")


async def _fetch_single(session: aiohttp.ClientSession, url: str) -> tuple[str, str]:
    try:
        async with session.get(url, timeout=8) as r:
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
        return {**state, "needs_clarification": True}

    prompt = f"""/no_think
                Ты – эксперт по поисковым запросам.
                Если приведённый текст является понятным вопросом, преобразуй его в одну поисковую фразу (3-8 слов) без стоп-слов.
                Если это не вопрос или он непонятен, ответь строго: NEEDS_CLARIFICATION.
                Отвечай без кавычек.

                Ввод: {query}

                Ответ:"""
    response = await llm.ainvoke(prompt)
    answer = _strip_think(response.content).strip()

    if answer.upper() == "NEEDS_CLARIFICATION":
        return {**state, "needs_clarification": True}

    print(f"[DEBUG] Сформированный запрос: {answer}\n")
    return {**state, "query": answer, "needs_clarification": False}


async def clarify_question(state: AgentState) -> AgentState:
    new_query = await asyncio.to_thread(
        input, "Запрос непонятен. Пожалуйста, уточните формулировку: "
    )
    new_query = new_query.strip()
    return {
        **state,
        "query": new_query,
        "needs_clarification": False,
        "error": None,
        "search_results": None,
        "page_contents": None,
        "page_summaries": None,
        "final_summary": None,
    }


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
    async with aiohttp.ClientSession() as session:
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

                if summary == "Нерелевантный источник.":
                    jina_url = _jina_ai_url(url)
                    async with session.get(jina_url, timeout=8) as jr:
                        jina_text = await jr.text()
                    jina_prompt = f"""/no_think
                                   Ты – аналитик, умеющий делать краткие саммари.
                                   Всегда отвечай на русском.
                                   Если текст нерелевантен вопросу, ответь: "Нерелевантный источник".
                                   Вопрос: {original_query}

                                   Текст:
                                   {jina_text}

                                   Резюме:"""
                    resp = await llm.ainvoke(jina_prompt)
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
                Ты – помощник, который собирает общее саммари из нескольких источников.
                В конце сделай раздел "Источники" в формате [N] URL, а в тексте – ссылки на них.
                Если мнения расходятся – отрази все точки зрения.
                Если не удалось получить релевантную информацию из источника, не упоминай его в ответе.
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

            if not started and token.strip() == "":
                continue
            started = True
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
    g.add_node("clarify", clarify_question)
    g.add_node("search", search_duckduckgo)
    g.add_node("extract", extract_page_content)
    g.add_node("summarize_pages", summarize_pages)
    g.add_node("create_final_summary", create_final_summary)

    g.set_entry_point("format_query")

    g.add_conditional_edges(
        "format_query",
        should_continue,
        {"clarify": "clarify", "search": "search", "error": END},
    )
    g.add_edge("clarify", "format_query")

    g.add_conditional_edges(
        "search", should_continue, {"extract": "extract", "error": END}
    )
    g.add_conditional_edges(
        "extract", should_continue, {"summarize_pages": "summarize_pages", "error": END}
    )
    g.add_conditional_edges(
        "summarize_pages",
        should_continue,
        {"create_final_summary": "create_final_summary", "error": END},
    )
    g.add_edge("create_final_summary", END)
    return g.compile()


async def run_agent(query: str):
    agent = build_agent_graph()
    print(f"\nОбрабатываю запрос: {query}\nПодождите, идёт поиск информации...\n")
    result: AgentState = await agent.ainvoke({"query": query})
    if result.get("error") and not result.get("needs_clarification"):
        print(f"Произошла ошибка: {result['error']}")


async def main():
    if len(sys.argv) > 1:
        await run_agent(" ".join(sys.argv[1:]))
        return
    print("Введите ваш вопрос или 'q' для выхода.")
    while True:
        q = input("\nВаш вопрос: ").strip()
        if q.lower() == "q":
            break
        await run_agent(q)


if __name__ == "__main__":
    asyncio.run(main())