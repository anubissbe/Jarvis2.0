class SimpleLLM:
    """A minimal LLM implementation used for testing."""

    def generate(self, message: str) -> str:
        """Return a simple echo response."""
        return f"Echo: {message}"


PROMPT_TEMPLATE = (
    "You are Jarvis, a helpful bilingual assistant. Respond in the language of the user (English or Dutch).\n"
    "{history}\nUser: {input}\nJarvis:"
)

# ``prompt`` previously exposed a ``PromptTemplate`` from LangChain. The current
# minimal implementation no longer relies on LangChain, but other modules may
# still attempt to import ``prompt``. Provide a simple string fallback to avoid
# import errors if older code expects this name.
prompt = PROMPT_TEMPLATE


def get_llm() -> SimpleLLM:
    """Return the LLM instance."""
    return SimpleLLM()
