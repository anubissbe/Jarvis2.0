class SimpleLLM:
    """A minimal LLM implementation used for testing."""

    def generate(self, message: str) -> str:
        """Return a simple echo response."""
        return f"Echo: {message}"


def get_llm() -> SimpleLLM:
    """Return the LLM instance."""
    return SimpleLLM()
