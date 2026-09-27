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
    # mark them on the frame for vizualization:
    if corners is not None:
        for corner in corners:
            int_corners = corner.astype(int)
            cv.polylines(frame, int_corners, True, (0, 255, 0), 5)
    return corners, ids


@dataclass
class CardZone:
    """A single hard-coded card slot on a rectified board.

    Args:
        name: Unique zone identifier (e.g. "shop_slot_1").
        roi: (x, y, w, h) crop rectangle in rectified-board pixel coordinates.
        library_type: Which CardLibrary this zone should be matched against
            (e.g. "contracts", "crew", "ship_parts").
        expected_number_cards: The number of cards expected in this zone.
    """
    name: str
    roi: tuple
    library_type: str
    expected_number_cards: int = 1

    expected_width: float = None
    expected_height: float = None
    width_tolerance: float = 40.0
    height_tolerance: float = 40.0

    expected_aspect_ratio: float = None
    aspect_ratio_tolerance: float = 0.2


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

# TODO: decide whether to use the real pixel ROIs (in rectified_size coordinates) for every card slot or for each section of the board...
BOARD_CONFIGS = {
    "contracts_board": BoardConfig(
        board_id="contracts_board",
        corner_marker_ids={
            5: "top_left",
            6: "top_right",
            7: "bottom_right",
            8: "bottom_left",
        },
        rectified_size=(1000, 1000),
        zones=[
            CardZone(name="contract_slot_1", roi=(0, 0, 700, 1000), library_type="contracts", expected_number_cards=8, expected_aspect_ratio=0.7),
        ],
    ),
    "shop_board": BoardConfig(
        board_id="shop_board",
        corner_marker_ids={
            1: "top_left",
            2: "top_right",
            3: "bottom_right",
            4: "bottom_left",
        },
        rectified_size=(1500, 1000),
        zones=[
            CardZone(name="crew_slot_1", roi=(600, 0, 900, 430), library_type="crew", expected_number_cards=3, expected_aspect_ratio=1.35),
            CardZone(name="ship_part_slot_1", roi=(600, 430, 900, 570), library_type="ship_parts", expected_number_cards=6, expected_aspect_ratio=1),
        ],
    ),
}


def rectify_board(frame, aruco_corners, aruco_ids, board_config: BoardConfig, stabilizer=None):
    """
    Rectify a board to a top-down view using its four known corner markers.

    Args:
        frame (np.ndarray): The input camera frame.
        aruco_corners (list): Corners as returned by detect_aruco_markers.
        aruco_ids (np.ndarray): Ids as returned by detect_aruco_markers.
        board_config (BoardConfig): Which board to rectify, and which marker
            ids correspond to its corners.
        stabilizer (camera.ContourStabilizer, optional): Smooths the board's
            corner points across frames once they've been confirmed stable.

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

    if stabilizer is not None:
        confirmed = stabilizer.update([src_points])
        if confirmed:
            src_points = confirmed[0]

    dst_points = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1],
    ], dtype=np.float32)


    matrix = cv.getPerspectiveTransform(src_points, dst_points)
    return cv.warpPerspective(frame, matrix, (w, h))

# TODO: test this and add unit test for it. 
def order_cards(cards):
    """
    Order detected card contours by their coordinates (left to right, then top to bottom).

    Args:
        cards (list): List of card contours, each represented as a quadrilateral (4 points).

    Returns:
        list: The sorted list of card contours.
    """
    # if there are multiple cards detected, sort them by their coordinates (left to right, then top to bottom):
    if not cards:
        return []
    # Sort primarily by y (top to bottom), then by x (left to right)
    return sorted(cards, key=lambda c: (c[:, 0, 1].mean(), c[:, 0, 0].mean()))

def identify_card_subarea(rectified_board, zone: CardZone, output_size=CARD_OUTPUT_SIZE, edge_detection ='canny', stabilizer=None):
    #TODO: I probably want to return all candidates and sort/match them later, not cut off here....
    """
    Crop a zone out of a rectified board and locate/warp the card(s) inside it.

    Args:
        rectified_board (np.ndarray): The rectified board image.
        zone (CardZone): The hard-coded zone to inspect.
        output_size (tuple): Size to warp each detected card to.
        stabilizer (camera.ContourStabilizer, optional): Smooths the zone's
            card outlines across frames once they've been confirmed stable.

    Returns:
        list or None: A list of (card_image, (x, y, w, h)) tuples, one per
        detected card, or None if no card-shaped contour was found in this zone.
    """
    number_cards = zone.expected_number_cards
    identified_cards = []
    x, y, w, h = zone.roi
    crop = rectified_board[y:y + h, x:x + w]
    if crop.size == 0:
        return None

    if zone.expected_aspect_ratio == 1:
        output_size = (output_size[0], output_size[0])

    gray = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
    if edge_detection == 'canny':
        edges = camera.edge_detection_canny(gray)
    elif edge_detection == 'threshold':
        edges = camera.edge_detection_thresholding(gray, threshold1 = 180, threshold2 = 255)
    else:
        raise ValueError(f"Unsupported edge detection method: {edge_detection}")
    # show the edges for debugging:
    cv.imshow(f"edges_debug_{zone.name}", edges)
    contours = camera.approximate_contours(camera.find_contours(edges), max_contours=number_cards, min_area=500, expected_aspect_ratio=zone.expected_aspect_ratio)
    if not contours:
        return None

    # once every expected card has a confirmed, stabilized track, use those
    # smoothed points instead of the raw (frame-to-frame jittery) contours.
    quads = contours
    if stabilizer is not None:
        confirmed = stabilizer.update(contours)
        if len(confirmed) >= number_cards:
            quads = confirmed

    # order the quads consistently (left to right, then top to bottom)
    quads = order_cards(np.array(quads))

    for i in range(number_cards):
        if i >= len(quads):
            return None  # not enough card-shaped contours found
        pts_src = np.float32(np.array(quads[i]).reshape(4, 2))
        pts_src = camera.order_corners(pts_src)
        out_w, out_h = output_size
        pts_dst = np.float32([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]])
        matrix = cv.getPerspectiveTransform(pts_src, pts_dst)
        identified_cards.append((cv.warpPerspective(crop, matrix, (out_w, out_h)), (x + pts_src[0][0], y + pts_src[0][1], out_w, out_h)))
        cx, cy, cw, ch = x + pts_src[0][0], y + pts_src[0][1], out_w, out_h
        # TODO: figure out how to not make this affect the actual frame because it's messing up whichever zone is second:
        # cv.rectangle(rectified_board, (int(cx), int(cy)), (int(cx + cw), int(cy + ch)), (255, 0, 0), 2)

    return identified_cards if identified_cards else None

