import pytest
from unittest.mock import MagicMock, patch
import src.library as library


@pytest.fixture
def empty_library():
    return library.CardLibrary()


@pytest.fixture
def test_card():
    return library.Card("Card1", "Description1")


def test_library_cards_initially_empty(empty_library):
    assert len(empty_library.cards) == 0


def test_add_card(empty_library, test_card):
    empty_library.add_card(test_card)
    assert "Card1" in empty_library.cards
    assert empty_library.cards["Card1"] == test_card
    assert len(empty_library.cards) == 1


def test_get_nonexistent_card(empty_library):
    assert empty_library.cards.get("NonExistentCard") is None


@patch("os.listdir")
def test_library_from_directory(mock_listdir):
    mock_listdir.return_value = ["Bill_Bendo.jpg"]
    lib = library.CardLibrary.from_directory(
        directory="./data/test_cards", feature_extractor=None
    )
    assert isinstance(lib, library.CardLibrary)
    assert isinstance(lib.cards, dict)