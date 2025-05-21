from app.memory.graph_memory import save_interaction


class FakeSession:
    def __init__(self):
        self.calls = []

    def run(self, query, **params):
        self.calls.append((query, params))


class FakeDriver:
    def __init__(self):
        self.session_obj = FakeSession()

    def session(self):
        driver = self

        class CM:
            def __enter__(self_inner):
                return driver.session_obj

            def __exit__(self_inner, exc_type, exc, tb):
                pass

        return CM()


def test_save_interaction_stores_messages():
    driver = FakeDriver()
    save_interaction(driver, "hi", "bye")
    assert len(driver.session_obj.calls) == 1
    query, params = driver.session_obj.calls[0]
    assert "MERGE" in query
    assert params["user_msg"] == "hi"
    assert params["ai_msg"] == "bye"
