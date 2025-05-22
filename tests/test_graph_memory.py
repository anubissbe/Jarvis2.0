import pytest

from app.memory.graph_memory import save_interaction


class FakeSession:
    def __init__(self):
        self.calls = []

    async def run(self, query, **params):
        self.calls.append((query, params))


class FakeDriver:
    def __init__(self):
        self.session_obj = FakeSession()

    def session(self):
        driver = self

        class CM:
            async def __aenter__(self_inner):
                return driver.session_obj

            async def __aexit__(self_inner, exc_type, exc, tb):
                pass

        return CM()

    async def close(self):
        pass


@pytest.mark.asyncio
async def test_save_interaction_stores_messages():
    driver = FakeDriver()
    await save_interaction(driver, "hi", "bye")
    assert len(driver.session_obj.calls) == 1
    query, params = driver.session_obj.calls[0]
    assert "MERGE" in query
    assert params["user_msg"] == "hi"
    assert params["ai_msg"] == "bye"


class FailingSession:
    async def run(self, query, **params):
        raise RuntimeError("boom")


class FailingDriver:
    def session(self):
        session = FailingSession()

        class CM:
            async def __aenter__(self_inner):
                return session

            async def __aexit__(self_inner, exc_type, exc, tb):
                pass

        return CM()


@pytest.mark.asyncio
async def test_save_interaction_raises_on_failure():
    driver = FailingDriver()
    with pytest.raises(RuntimeError):
        await save_interaction(driver, "hi", "bye")
