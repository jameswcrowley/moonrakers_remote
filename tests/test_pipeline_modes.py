import json
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

import src.pipeline as pipeline
import src.state as state
import src.webserver as webserver


class FakeLibrary:
    def __init__(self, card_ids):
        self.cards = {card_id: object() for card_id in card_ids}


def make_libraries():
    return {
        "contracts": FakeLibrary([f"contract_{index}" for index in range(9)]),
        "crew": FakeLibrary([f"crew_{index}" for index in range(4)] + ["Ada_Massa"]),
        "ship_parts": FakeLibrary([f"ship_part_{index}" for index in range(7)]),
    }


def test_manual_updates_keep_empty_slots_and_confidence():
    board_state = state.BoardState()
    libraries = make_libraries()
    manual_input = {
        "contract_slot_1": ["contract_1", None, "contract_2", None, None, None, None, None],
        "crew_slot_1": [None] * 3,
        "ship_part_slot_1": [None] * 6,
    }

    assert pipeline.process_board_from_manual(libraries, board_state, manual_input)
    cards = board_state.zones["contract_slot_1"]["cards"]
    assert [card["card_id"] for card in cards] == manual_input["contract_slot_1"]
    assert [card["confidence"] for card in cards[:3]] == [1.0, 0.0, 1.0]


def test_manual_update_rejects_unknown_card_ids():
    with pytest.raises(ValueError, match="unknown card ID"):
        pipeline.process_board_from_manual(
            make_libraries(),
            state.BoardState(),
            {"contract_slot_1": ["missing"] * 8},
        )


def test_random_initialization_and_reroll_have_no_duplicates_per_zone():
    libraries = make_libraries()
    board_state = state.BoardState()

    assert pipeline.initialize_random_board(libraries, board_state)
    for zone in board_state.zones.values():
        card_ids = [card["card_id"] for card in zone["cards"]]
        assert len(card_ids) == len(set(card_ids))

    previous = board_state.zones["crew_slot_1"]["cards"][0]["card_id"]
    _, replacement = pipeline.update_card_slot(
        libraries, board_state, "crew_slot_1", 0, randomize=True
    )
    still_shown = {
        card["card_id"]
        for card in board_state.zones["crew_slot_1"]["cards"][1:]
    }
    assert replacement != previous
    assert replacement not in still_shown
    assert board_state.zones["crew_slot_1"]["cards"][0]["card_id"] == replacement
    assert previous not in still_shown


def test_slot_update_rejects_invalid_zone_slot_and_card():
    libraries = make_libraries()
    board_state = state.BoardState()
    with pytest.raises(ValueError, match="Unknown zone"):
        pipeline.update_card_slot(libraries, board_state, "missing", 0)
    with pytest.raises(ValueError, match="Invalid slot"):
        pipeline.update_card_slot(libraries, board_state, "crew_slot_1", 3)
    with pytest.raises(ValueError, match="Unknown card ID"):
        pipeline.update_card_slot(libraries, board_state, "crew_slot_1", 0, "missing")


def test_library_loading_excludes_placeholder_artwork(monkeypatch):
    monkeypatch.setattr(pipeline.os.path, "isdir", lambda directory: True)
    monkeypatch.setattr(
        pipeline.library.CardLibrary,
        "from_directory",
        lambda directory, feature_extractor: FakeLibrary([
            "real_card",
            "contracts_placeholder",
            "crew_placeholder",
            "shipParts_placeholder",
        ]),
    )

    libraries = pipeline.load_libraries(None)

    assert all(list(card_library.cards) == ["real_card"] for card_library in libraries.values())


def test_manual_mode_startup_does_not_open_camera(monkeypatch, tmp_path):
    class DummyServer:
        def serve_forever(self):
            return

        def server_close(self):
            return

    args = SimpleNamespace(
        mode="manual",
        matcher="orb",
        output=str(tmp_path / "state.json"),
        images_dir=str(tmp_path / "images"),
        host="127.0.0.1",
        port=0,
    )
    monkeypatch.setattr(pipeline, "parse_arguments", lambda: args)
    monkeypatch.setattr(pipeline, "load_libraries", lambda extractor: make_libraries())
    monkeypatch.setattr(webserver, "create_server", lambda **kwargs: DummyServer())
    monkeypatch.setattr(
        pipeline.cv,
        "VideoCapture",
        lambda *_: pytest.fail("manual mode must not open a camera"),
    )

    pipeline.main()

    saved = json.loads((tmp_path / "state.json").read_text())
    assert len(saved["zones"]["crew_slot_1"]["cards"]) == 3


def test_manual_api_updates_state_and_exports_slot_image(tmp_path):
    board_state = state.BoardState()
    libraries = make_libraries()
    for zone_name, slot_count in (
        ("contract_slot_1", 8),
        ("crew_slot_1", 3),
        ("ship_part_slot_1", 6),
    ):
        board_state.initialize_zone(zone_name, slot_count)
    server = webserver.create_server(
        host="127.0.0.1",
        port=0,
        mode="manual",
        libraries=libraries,
        board_state=board_state,
        state_path=str(tmp_path / "state.json"),
        images_dir=str(tmp_path / "images"),
    )
    thread = webserver.start_server_thread(server)
    address, port = server.server_address
    try:
        request = Request(
            f"http://{address}:{port}/api/slot",
            data=json.dumps({
                "zone": "crew_slot_1",
                "slot": 0,
                "action": "select",
                "card_id": "Ada_Massa",
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            assert response.status == 200
            assert json.load(response)["ok"] is True

        saved = json.loads((tmp_path / "state.json").read_text())
        assert saved["zones"]["crew_slot_1"]["cards"][0]["card_id"] == "Ada_Massa"
        assert (tmp_path / "images" / "current_crew1.jpg").is_file()

        clear = Request(
            f"http://{address}:{port}/api/slot",
            data=json.dumps({"zone": "crew_slot_1", "slot": 0, "action": "clear"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(clear) as response:
            assert response.status == 200
        saved = json.loads((tmp_path / "state.json").read_text())
        assert saved["zones"]["crew_slot_1"]["cards"][0]["card_id"] is None
        assert (tmp_path / "images" / "current_crew1.jpg").read_bytes() == (
            Path("data/saved_images/crew/crew_placeholder.jpg").read_bytes()
        )

        invalid = Request(
            f"http://{address}:{port}/api/slot",
            data=json.dumps({"zone": "unknown", "slot": 0, "action": "clear"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(HTTPError) as error:
            urlopen(invalid)
        assert error.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)