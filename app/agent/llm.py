class SimpleLLM:
    """A minimal LLM implementation used for testing."""

    def generate(self, message: str) -> str:
        """Return a simple echo response."""
        return f"Echo: {message}"


PROMPT_TEMPLATE = (
    "You are Jarvis, a helpful bilingual assistant. Respond in the language of the user (English or Dutch).\n"
    "{history}\nUser: {input}\nJarvis:"
)


def get_llm() -> SimpleLLM:
    """Return the LLM instance."""
    return SimpleLLM()
