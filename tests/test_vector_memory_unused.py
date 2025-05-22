def test_vector_store_not_initialized():
    import sys
    import importlib
    import app.main
    assert 'app.memory.vector_memory' not in sys.modules
    assert not hasattr(app.main, 'get_vector_store')
