import gradio as gr
import asyncio
import threading
from typing import Optional, List, Dict, Any
from agent import SearchAgent
from config import AgentConfig
from ollama_utils import OllamaAPI, OllamaModel


class SearchAgentGradio:
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.agent = None
        self.ollama_api = OllamaAPI(base_url=self.config.ollama_base_url)
        self.available_models: List[OllamaModel] = []
        self.current_model_name = None
        self.loop = None
        self.thread = None
        self._start_event_loop()
        
    def _start_event_loop(self):
        def run_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()
        
        self.thread = threading.Thread(target=run_loop, daemon=True)
        self.thread.start()
        while self.loop is None:
            threading.Event().wait(0.01)
    
    def _run_async(self, coro):
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result()
        
    def initialize(self):
        return self._run_async(self._initialize_async())
    
    async def _initialize_async(self):
        try:
            self.available_models = await self.ollama_api.get_models()
            if not self.available_models:
                return False, [], None
            
            model_choices = []
            default_model = None
            
            for model in self.available_models:
                display_name = model.name
                model_choices.append(display_name)
                
                if model.name == self.config.default_model.name:
                    default_model = display_name
            
            if not default_model and model_choices:
                default_model = model_choices[0]
            
            return True, model_choices, default_model
            
        except Exception as e:
            print(f"Ошибка при получении списка моделей: {e}")
            return False, [], None
    
    def set_model(self, model_name: str) -> bool:
        for model in self.available_models:
            if model.name == model_name:
                self.config.set_model(
                    model_name=model.name,
                    temperature=0.3,
                    family=model.family,
                )
                self.agent = SearchAgent(self.config)
                self.current_model_name = model_name
                return True
        return False
    
    def process_query(self, query: str, model_display_name: str):
        if not query.strip():
            yield "", "Пожалуйста, введите вопрос"
            return
        
        model_name = model_display_name
        if model_name and model_name != self.current_model_name:
            if not self.set_model(model_display_name):
                yield "", "Ошибка: модель не найдена"
                return
        
        if not self.agent:
            yield "", "Агент не инициализирован. Выберите модель."
            return
        
        response_text = ""
        sources = {}
        status_text = f"Обрабатываю запрос: {query}\n\nПодождите, идёт поиск информации..."
        
        yield response_text, status_text
        
        async def process_async():
            async for chunk in self.agent.asearch(query):
                yield chunk
        
        gen = process_async()
        
        try:
            while True:
                future = asyncio.run_coroutine_threadsafe(gen.__anext__(), self.loop)
                try:
                    chunk = future.result(timeout=60)
                    
                    if chunk["type"] == "content":
                        response_text += chunk["content"]
                        yield response_text, status_text
                        
                    elif chunk["type"] == "sources":
                        sources = chunk["sources"]
                        sources_text = self._format_sources(sources)
                        status_text = f"Запрос обработан\n\n{sources_text}"
                        yield response_text, status_text
                        
                    elif chunk["type"] == "error":
                        error_msg = f"**Ошибка:** {chunk['content']}"
                        yield response_text, error_msg
                        break
                        
                except StopAsyncIteration:
                    if response_text and not sources:
                        status_text = "Запрос обработан"
                        yield response_text, status_text
                    break
                except Exception as e:
                    error_msg = f"**Ошибка при обработке:** {str(e)}"
                    yield response_text, error_msg
                    break
                    
        except Exception as e:
            error_msg = f"**Ошибка:** {str(e)}"
            yield response_text, error_msg
        finally:
            future = asyncio.run_coroutine_threadsafe(gen.aclose(), self.loop)
            try:
                future.result(timeout=5)
            except:
                pass
    
    def _format_sources(self, sources: Dict[str, Dict[str, Any]]) -> str:
        if not sources:
            return ""
        
        formatted = "### Источники:\n\n"
        for idx, (url, info) in enumerate(sorted(sources.items()), 1):
            title = info.get('title', 'Без названия')
            formatted += f"{idx}. [{title}]({url})\n"
        
        return formatted
    
    def cleanup(self):
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread:
            self.thread.join(timeout=5)


def create_gradio_interface():
    agent_ui = SearchAgentGradio()
    
    success, model_choices, default_model = agent_ui.initialize()
    
    if not success:
        with gr.Blocks(title="Search Agent - Ошибка") as interface:
            gr.Markdown("# Ошибка подключения к Ollama")
            gr.Markdown(
                "Не удалось подключиться к Ollama. Убедитесь, что:\n"
                "1. Ollama установлен и запущен\n"
                "2. Сервис доступен по адресу http://localhost:11434\n"
                "3. У вас загружена хотя бы одна модель\n\n"
            )
        return interface
    
    if default_model:
        agent_ui.set_model(default_model)
    
    with gr.Blocks(
        title="Search Agent",
        theme=gr.themes.Base(),
        css="""
        .gradio-container {
            max-width: 1200px;
            margin: auto;
        }
        .status-area {
            border-left: 3px solid #e0e0e0;
            padding-left: 10px;
            margin-top: 10px;
        }
        """
    ) as interface:
        gr.Markdown("# Search Agent")
        
        with gr.Row():
            model_dropdown = gr.Dropdown(
                    choices=model_choices,
                    value=default_model,
                    label="Модель",
                    interactive=True,
                    info="Выберите языковую модель"
                )
        
        with gr.Row():
            with gr.Column(scale=3):
                query_input = gr.Textbox(
                    label="Ваш вопрос",
                    lines=1,
                    autofocus=True
                )
                
                with gr.Row():
                    submit_btn = gr.Button("Отправить", variant="primary", scale=2)
                    clear_btn = gr.Button("Очистить", scale=1)
        
        response_output = gr.Markdown(
            label="Ответ",
            value="",
            elem_classes="response-area"
        )
        
        status_output = gr.Markdown(
            label="Статус",
            value="Готов к работе. Введите ваш вопрос и нажмите 'Отправить'.",
            elem_classes="status-area"
        )
        
        def process_with_ui_updates(query, model):
            yield "", "", gr.update(interactive=False), gr.update(interactive=False)
            
            for response, status in agent_ui.process_query(query, model):
                yield response, status, gr.update(interactive=False), gr.update(interactive=False)
            
            yield response, status, gr.update(interactive=True), gr.update(interactive=True)
        
        submit_btn.click(
            fn=process_with_ui_updates,
            inputs=[query_input, model_dropdown],
            outputs=[response_output, status_output, submit_btn, clear_btn],
            show_progress=False
        )
        
        query_input.submit(
            fn=process_with_ui_updates,
            inputs=[query_input, model_dropdown],
            outputs=[response_output, status_output, submit_btn, clear_btn],
            show_progress=False
        )
        
        clear_btn.click(
            fn=lambda: ("", "", "Готов к работе. Введите ваш вопрос и нажмите 'Отправить'."),
            inputs=[],
            outputs=[query_input, response_output, status_output]
        )

        model_dropdown.change(
            fn=lambda model: agent_ui.set_model(model),
            inputs=[model_dropdown],
            outputs=[]
        )
        
        interface.unload(lambda: agent_ui.cleanup())
    
    return interface


if __name__ == "__main__":
    interface = create_gradio_interface()
    
    try:
        interface.launch(
            share=False,
            show_error=True,
            quiet=False
        )
    except KeyboardInterrupt:
        print("\nЗавершение работы...")
    except Exception as e:
        print(f"Ошибка при запуске: {e}")