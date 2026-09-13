import src.statetracking as statetracking


def test_statetracking_module_imports():
    # statetracking.py is currently just a stub wiring together the other
    # modules; this smoke test will start failing the moment real functions
    # are added, prompting real test coverage for them.
    assert hasattr(statetracking, "camera")
    assert hasattr(statetracking, "library")
    assert hasattr(statetracking, "recognition")
