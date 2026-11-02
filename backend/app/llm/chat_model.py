"""Turn saved LLMSettings into a LangChain chat model.

This is LangChain's first job in the project: one interface (`BaseChatModel`)
over every provider, so agent code never knows which one it's talking to.
"""

from langchain_core.language_models import BaseChatModel

from app.llm.settings import LLMProvider, LLMSettings

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def build_chat_model(settings: LLMSettings, temperature: float = 0.7, max_tokens: int = 4096) -> BaseChatModel:
    """`max_tokens` caps the reply length: without it, a model stuck repeating itself
    inside one JSON string can run for minutes and still produce unparseable output."""
    # Imports are local so only the selected provider's package is loaded.
    key = settings.api_key.get_secret_value() if settings.api_key else None
    match settings.provider:
        case LLMProvider.LOCAL:
            from langchain_ollama import ChatOllama

            # num_ctx: Ollama's default context window is small; design prompts are long.
            return ChatOllama(model=settings.model, base_url=settings.base_url, temperature=temperature,
                              num_ctx=16384, num_predict=max_tokens)
        case LLMProvider.OPENAI:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(model=settings.model, api_key=key, temperature=temperature, max_tokens=max_tokens)
        case LLMProvider.ANTHROPIC:
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(model=settings.model, api_key=key, temperature=temperature, max_tokens=max_tokens)
        case LLMProvider.OPENROUTER:
            from langchain_openai import ChatOpenAI

            # OpenRouter speaks the OpenAI API, so the OpenAI client works with a different URL.
            return ChatOpenAI(model=settings.model, api_key=key, base_url=OPENROUTER_BASE_URL,
                              temperature=temperature, max_tokens=max_tokens)
