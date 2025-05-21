from langchain.chat_models import ChatOllama
from langchain.prompts import PromptTemplate

from ..config import settings

PROMPT_TEMPLATE = (
    "You are Jarvis, a helpful bilingual assistant. Respond in the language of the "
    "user (English or Dutch).\n"
    "{history}\n"
    "Context: {context}\n"
    "User: {input}\n"
    "Jarvis:"
)

prompt = PromptTemplate(
    input_variables=["history", "input", "context"],
    template=PROMPT_TEMPLATE,
)


def get_llm():
    return ChatOllama(base_url=settings.ollama_base_url, model="llama3")
