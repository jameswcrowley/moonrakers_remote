import json
import src.state as state


def test_update_returns_true_on_new_zone():
    board_state = state.BoardState()
    changed = board_state.update("zone_1", "card_a", 0.9)
    assert changed is True
    assert board_state.zones["zone_1"]["card_id"] == "card_a"
    assert board_state.zones["zone_1"]["confidence"] == 0.9


def test_update_returns_false_when_card_unchanged():
    board_state = state.BoardState()
    board_state.update("zone_1", "card_a", 0.9)
    changed = board_state.update("zone_1", "card_a", 0.95)
    assert changed is False
    assert board_state.zones["zone_1"]["confidence"] == 0.95


def test_update_returns_true_when_card_changes():
    board_state = state.BoardState()
    board_state.update("zone_1", "card_a", 0.9)
    changed = board_state.update("zone_1", "card_b", 0.5)
    assert changed is True
    assert board_state.zones["zone_1"]["card_id"] == "card_b"


def test_to_dict_wraps_zones():
    board_state = state.BoardState()
    board_state.update("zone_1", "card_a", 0.9)
    assert board_state.to_dict() == {"zones": board_state.zones}


def test_save_and_load_round_trip(tmp_path):
    board_state = state.BoardState()
    board_state.update("zone_1", "card_a", 0.9)
    board_state.update("zone_2", None, 0.0)

    path = tmp_path / "board_state.json"
    board_state.save(path)

    with open(path) as f:
        raw = json.load(f)
    assert raw["zones"]["zone_1"]["card_id"] == "card_a"

    loaded = state.BoardState.load(path)
    assert loaded.zones["zone_1"]["card_id"] == "card_a"
    assert loaded.zones["zone_2"]["card_id"] is None
