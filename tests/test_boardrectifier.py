import numpy as np
import cv2 as cv
import src.boardrectifier as boardrectifier


def _marker_corners(center, half_size=10.0):
    x, y = center
    return np.array([[
        [x - half_size, y - half_size],
        [x + half_size, y - half_size],
        [x + half_size, y + half_size],
        [x - half_size, y + half_size],
    ]], dtype=np.float32)


def test_card_zone_defaults():
    zone = boardrectifier.CardZone(name="slot_1", roi=(0, 0, 100, 150), library_type="crew")
    assert zone.name == "slot_1"
    assert zone.roi == (0, 0, 100, 150)
    assert zone.library_type == "crew"
    assert zone.width_tolerance == 40.0
    assert zone.height_tolerance == 40.0


def test_board_config_defaults():
    config = boardrectifier.BoardConfig(board_id="test_board", corner_marker_ids={0: "top_left"})
    assert config.rectified_size == (1000, 1000)
    assert config.zones == []


def test_board_configs_have_four_distinct_corners():
    for config in boardrectifier.BOARD_CONFIGS.values():
        assert sorted(config.corner_marker_ids.values()) == sorted(boardrectifier.CORNER_ORDER)


def test_rectify_board_missing_ids_returns_none():
    board_config = boardrectifier.BoardConfig(
        board_id="board",
        corner_marker_ids={0: "top_left", 1: "top_right", 2: "bottom_right", 3: "bottom_left"},
        rectified_size=(100, 100),
    )
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    aruco_ids = np.array([[0], [1], [2]])  # missing marker id 3
    aruco_corners = [_marker_corners((10, 10)), _marker_corners((190, 10)), _marker_corners((190, 190))]

    assert boardrectifier.rectify_board(frame, aruco_corners, aruco_ids, board_config) is None


def test_rectify_board_none_ids_returns_none():
    board_config = boardrectifier.BoardConfig(board_id="board", corner_marker_ids={0: "top_left"})
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    assert boardrectifier.rectify_board(frame, None, None, board_config) is None


def test_rectify_board_warps_to_rectified_size():
    board_config = boardrectifier.BoardConfig(
        board_id="board",
        corner_marker_ids={0: "top_left", 1: "top_right", 2: "bottom_right", 3: "bottom_left"},
        rectified_size=(100, 100),
    )
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    cv.rectangle(frame, (10, 10), (190, 190), (255, 255, 255), -1)

    aruco_ids = np.array([[0], [1], [2], [3]])
    aruco_corners = [
        _marker_corners((10, 10)),
        _marker_corners((190, 10)),
        _marker_corners((190, 190)),
        _marker_corners((10, 190)),
    ]

    rectified = boardrectifier.rectify_board(frame, aruco_corners, aruco_ids, board_config)

    assert rectified is not None
    assert rectified.shape[:2] == (100, 100)


def test_identify_card_subarea_empty_crop_returns_none():
    rectified_board = np.zeros((50, 50, 3), dtype=np.uint8)
    zone = boardrectifier.CardZone(name="slot_1", roi=(100, 100, 50, 50), library_type="crew")

    assert boardrectifier.identify_card_subarea(rectified_board, zone) is None


def test_identify_card_subarea_no_contour_returns_none():
    rectified_board = np.zeros((300, 300, 3), dtype=np.uint8)
    zone = boardrectifier.CardZone(name="slot_1", roi=(0, 0, 300, 300), library_type="crew")

    assert boardrectifier.identify_card_subarea(rectified_board, zone) is None


def test_identify_card_subarea_finds_and_warps_card():
    rectified_board = np.zeros((300, 300, 3), dtype=np.uint8)
    cv.rectangle(rectified_board, (50, 50), (250, 250), (255, 255, 255), -1)
    zone = boardrectifier.CardZone(name="slot_1", roi=(0, 0, 300, 300), library_type="crew")

    card_image = boardrectifier.identify_card_subarea(rectified_board, zone, output_size=(60, 84))

    assert card_image is not None
    assert card_image.shape[:2] == (84, 60)
