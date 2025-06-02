import asyncio
import aiohttp
from typing import Dict, Any
import trafilatura
from duckduckgo_search import DDGS
from datetime import datetime
import re
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import time
import json
from asyncio import Semaphore

from config import config as agent_config


class SearchQuery(BaseModel):
    query: str = Field(description="Поисковый запрос")
    max_results: int = Field(default=3, description="Максимальное количество результатов")

class CalculatorQuery(BaseModel):
    expression: str = Field(description="Математическое выражение для вычисления")


_ddgs_instance = None
_last_search_time = 0
_search_semaphore = Semaphore(1) 
MIN_SEARCH_INTERVAL = 2.0


def _get_ddgs():
    global _ddgs_instance
    if _ddgs_instance is None:
        _ddgs_instance = DDGS()
    return _ddgs_instance


async def _search(query: str, max_results: int = 3):
    if max_results is None:
        max_results = agent_config.max_search_results
        
    async with _search_semaphore:
        global _last_search_time
        
        current_time = time.time()
        time_since_last = current_time - _last_search_time
        
        if time_since_last < MIN_SEARCH_INTERVAL:
            await asyncio.sleep(MIN_SEARCH_INTERVAL - time_since_last)
        
        def _search(q):
            ddgs = _get_ddgs()
            return list(ddgs.text(q, max_results=max_results))
        
        result = await asyncio.to_thread(_search, query)
        _last_search_time = time.time()
        
        return result


async def _search_and_extract_impl(query: str, max_results: int = 3) -> Dict[str, Any]:
    """Внутренняя реализация поиска и извлечения"""
    if max_results is None:
        max_results = agent_config.max_search_results
        
    try:
        search_results = await _search(query, max_results)
        
        if not search_results:
            return {"error": "Не найдено результатов", "results": []}
        
        timeout = aiohttp.ClientTimeout(total=agent_config.request_timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = []
            for result in search_results:
                url = result.get("href", "")
                if url:
                    tasks.append(_fetch_content(session, url, result.get("title", ""), result.get("body", "")))
            
            contents = await asyncio.gather(*tasks)
        
        return {
            "query": query,
            "results": contents,
            "total_found": len(contents)
        }
    
    except Exception as e:
        return {"error": f"Ошибка при поиске: {str(e)}", "results": []}


@tool("search_and_extract", args_schema=SearchQuery)
async def search_and_extract(query: str, max_results: int = 3) -> Dict[str, Any]:
    """
    Выполняет поиск в интернете и извлекает содержимое найденных страниц.
    Возвращает словарь с результатами поиска и извлеченным контентом.
    """
    if isinstance(query, dict):
        max_results = query.get("max_results", max_results)
        query = query.get("query", "")
    
    elif isinstance(query, str) and query.strip().startswith("{"):
        try:
            query_dict = json.loads(query)
            if isinstance(query_dict, dict):
                max_results = query_dict.get("max_results", max_results)
                query = query_dict.get("query", query)
        except:
            pass
    
    print(f"Поиск: {query}, количество запросов: {max_results}\n")
    
    return await _search_and_extract_impl(query, max_results)


async def _fetch_content(session: aiohttp.ClientSession, url: str, title: str, snippet: str) -> Dict[str, Any]:
    try:
        async with session.get(url) as response:
            html = await response.text()
        
        text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            include_links=False
        ) or ""
        
        if len(text) > agent_config.max_content_length:
            text = text[:agent_config.max_content_length] + "..."
        
        return {
            "url": url,
            "title": title,
            "snippet": snippet,
            "content": text,
            "content_length": len(text),
            "success": True
        }
    
    except Exception as e:
        return {
            "url": url,
            "title": title,
            "snippet": snippet,
            "content": f"Ошибка извлечения: {str(e)}",
            "success": False
        }


@tool("calculate", args_schema=CalculatorQuery)
async def calculate(expression: str) -> Dict[str, Any]:
    """
    Выполняет математические вычисления.
    Поддерживает базовые операции: +, -, *, /, **, (), sqrt, sin, cos, tan, log.
    """
    try:
        import math
        
        if isinstance(expression, dict):
            expression = expression.get("expression", str(expression))
        
        allowed_names = {
            k: v for k, v in math.__dict__.items() 
            if not k.startswith("_")
        }
        allowed_names.update({
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum
        })
        
        if re.search(r'__|import|exec|eval|open|file|input|raw_input', expression):
            return {
                "error": "Недопустимое выражение",
                "success": False
            }
        
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        
        return {
            "expression": expression,
            "result": result,
            "success": True
        }
    
    except Exception as e:
        return {
            "expression": expression,
            "error": f"Ошибка вычисления: {str(e)}",
            "success": False
        }


@tool("get_current_datetime")
async def get_current_datetime() -> Dict[str, str]:
    """
    Возвращает текущую дату и время.
    Полезно для контекста временных запросов.
    """
    now = datetime.now()
    return {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "weekday": now.strftime("%A"),
        "timezone": "local"
    }


ALL_TOOLS = [
    search_and_extract,
    calculate,
    get_current_datetime
]