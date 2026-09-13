# code for taking the camera feed, identifying the board, and rectifying the images using OpenCV.

from dataclasses import dataclass, field
import cv2 as cv
import numpy as np

try:
    from . import camera
except ImportError:  # running as a script, or with src/ on sys.path directly
    import camera

# Canonical size (px) that every detected card is warped to before matching.
CARD_OUTPUT_SIZE = (300, 420)


def detect_aruco_markers(frame, dictionary=cv.aruco.DICT_6X6_250):
    """
    Detect ArUco markers in the given frame.

    Args:
        frame (np.ndarray): The input camera frame.
        dictionary (int): The ArUco dictionary id to use for detection.

    Returns:
        tuple: (corners, ids) as returned by cv.aruco.ArucoDetector.detectMarkers.
    """
    aruco_dict = cv.aruco.getPredefinedDictionary(dictionary)
    aruco_params = cv.aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(aruco_dict, aruco_params)
    corners, ids, _ = detector.detectMarkers(frame)
    return corners, ids


@dataclass
class CardZone:
    """A single hard-coded card slot on a rectified board.

    Args:
        name: Unique zone identifier (e.g. "shop_slot_1").
        roi: (x, y, w, h) crop rectangle in rectified-board pixel coordinates.
        library_type: Which CardLibrary this zone should be matched against
            (e.g. "contracts", "crew", "ship_parts").
    """
    name: str
    roi: tuple
    library_type: str

    expected_width: float = None
    expected_height: float = None
    width_tolerance: float = 40.0
    height_tolerance: float = 40.0


@dataclass
class BoardConfig:
    """Static configuration for one physical board.

    corner_marker_ids maps ArUco marker id -> corner name, and must contain
    exactly the four ids that sit at the board's corners. The corner names
    ("top_left", "top_right", "bottom_right", "bottom_left") determine the
    ordering used to build the rectified image.
    """
    board_id: str
    corner_marker_ids: dict
    rectified_size: tuple = (1000, 1000)
    zones: list = field(default_factory=list)


CORNER_ORDER = ("top_left", "top_right", "bottom_right", "bottom_left")

# TODO: fill in with the real ArUco marker ids painted on each physical board
# and the real pixel ROIs (in rectified_size coordinates) for every card slot.
BOARD_CONFIGS = {
    "contracts_board": BoardConfig(
        board_id="contracts_board",
        corner_marker_ids={
            0: "top_left",
            1: "top_right",
            2: "bottom_right",
            3: "bottom_left",
        },
        rectified_size=(1000, 1000),
        zones=[
            # CardZone(name="contract_slot_1", roi=(x, y, w, h), library_type="contracts"),
        ],
    ),
    "shop_board": BoardConfig(
        board_id="shop_board",
        corner_marker_ids={
            4: "top_left",
            5: "top_right",
            6: "bottom_right",
            7: "bottom_left",
        },
        rectified_size=(1000, 1000),
        zones=[
            # CardZone(name="crew_slot_1", roi=(x, y, w, h), library_type="crew"),
            # CardZone(name="ship_part_slot_1", roi=(x, y, w, h), library_type="ship_parts"),
        ],
    ),
}


def rectify_board(frame, aruco_corners, aruco_ids, board_config: BoardConfig):
    """
    Rectify a board to a top-down view using its four known corner markers.

    Args:
        frame (np.ndarray): The input camera frame.
        aruco_corners (list): Corners as returned by detect_aruco_markers.
        aruco_ids (np.ndarray): Ids as returned by detect_aruco_markers.
        board_config (BoardConfig): Which board to rectify, and which marker
            ids correspond to its corners.

    Returns:
        np.ndarray or None: The rectified board image, or None if this
        board's corner markers were not all found.
    """
    if aruco_ids is None:
        return None

    ids_flat = aruco_ids.flatten()
    id_to_corner_pts = {}
    for marker_id, corner_name in board_config.corner_marker_ids.items():
        matches = np.where(ids_flat == marker_id)[0]
        if len(matches) == 0:
            return None  # missing a required corner marker for this board
        # use the marker's own center as the board corner point
        id_to_corner_pts[corner_name] = aruco_corners[matches[0]][0].mean(axis=0)

    w, h = board_config.rectified_size
    src_points = np.array([id_to_corner_pts[name] for name in CORNER_ORDER], dtype=np.float32)
    dst_points = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1],
    ], dtype=np.float32)

    # TODO: check this on the real board 
    # display the BoardConfig zones on the rectified board for debugging:
    for zone in board_config.zones:
        x, y, w, h = zone.roi
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.putText(frame, zone.name, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    matrix = cv.getPerspectiveTransform(src_points, dst_points)
    return cv.warpPerspective(frame, matrix, (w, h))


def identify_card_subarea(rectified_board, zone: CardZone, output_size=CARD_OUTPUT_SIZE):
    """
    Crop a zone out of a rectified board and locate/warp the card inside it.

    Args:
        rectified_board (np.ndarray): The rectified board image.
        zone (CardZone): The hard-coded zone to inspect.
        output_size (tuple): Size to warp the detected card to.

    Returns:
        np.ndarray or None: The warped card image, or None if no card-shaped
        contour was found in this zone.
    """
    x, y, w, h = zone.roi
    crop = rectified_board[y:y + h, x:x + w]
    if crop.size == 0:
        return None

    gray = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
    edges = camera.edge_detection_canny(gray)
    contours = camera.approximate_contours(camera.find_contours(edges), max_contours=1, min_area=500)
    if not contours:
        return None

    pts_src = np.float32(contours[0].reshape(4, 2))
    pts_src = camera.order_corners(pts_src)
    out_w, out_h = output_size
    pts_dst = np.float32([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]])
    matrix = cv.getPerspectiveTransform(pts_src, pts_dst)
    return cv.warpPerspective(crop, matrix, (out_w, out_h))

