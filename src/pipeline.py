# Ties camera capture, ArUco board rectification, per-zone card detection, and
# library matching together into a single loop that persists the current
# board state to a JSON file.

import argparse
import os
import time

import cv2 as cv

try:
    from . import boardrectifier, library, recognition, state
except ImportError:  # running as a script, or with src/ on sys.path directly
    import boardrectifier
    import library
    import recognition
    import state

# TODO: point these at the real image directories for each card type.
LIBRARY_DIRECTORIES = {
    "contracts": "data/saved_images/contracts",
    "crew": "data/saved_images/crew",
    "ship_parts": "data/saved_images/ship_parts",
}

MATCHERS = {
    "sift": (recognition.SIFTMatcher(), recognition.sift_extractor),
    "orb": (recognition.ORBMatcher(), recognition.orb_extractor),
}

MIN_MATCH_CONFIDENCE = 0.05


def load_libraries(feature_extractor):
    """Build one CardLibrary per zone type, keyed the same way as LIBRARY_DIRECTORIES."""
    libraries = {}
    for library_type, directory in LIBRARY_DIRECTORIES.items():
        if not os.path.isdir(directory):
            continue
        libraries[library_type] = library.CardLibrary.from_directory(directory, feature_extractor=feature_extractor)
    return libraries


def process_boards(frame, libraries, matcher, board_state: state.BoardState):
    """Detect, rectify, and match every configured board/zone for one frame.

    Returns:
        bool: True if any zone's recorded card changed this frame.
    """
    aruco_corners, aruco_ids = boardrectifier.detect_aruco_markers(frame)

    dirty = False
    for board_config in boardrectifier.BOARD_CONFIGS.values():
        rectified = boardrectifier.rectify_board(frame, aruco_corners, aruco_ids, board_config)
        if rectified is None:
            continue

        for zone in board_config.zones:
            card_library = libraries.get(zone.library_type)
            if card_library is None or not card_library.cards:
                continue

            card_image = boardrectifier.identify_card_subarea(rectified, zone)
            if card_image is None:
                changed = board_state.update(zone.name, None, 0.0)
                dirty = dirty or changed
                continue

            result = recognition.match_against_library(card_image, card_library, matcher)
            card_id, confidence = result.best_match()
            if confidence < MIN_MATCH_CONFIDENCE:
                card_id = None

            changed = board_state.update(zone.name, card_id, confidence)
            dirty = dirty or changed

    return dirty


def parse_arguments():
    """Parse camera and pipeline options."""
    parser = argparse.ArgumentParser(description="Track Moonrakers board state from a camera feed.")
    parser.add_argument("--camera", type=int, default=1, help="Camera device index (default: 1).")
    parser.add_argument("--matcher", choices=list(MATCHERS.keys()), default="sift", help="Feature matcher to use.")
    parser.add_argument("--output", default="board_state.json", help="Path to write the board state JSON.")
    return parser.parse_args()


def main():
    """Run the capture/rectify/match loop until the user presses q, saving state on change."""
    args = parse_arguments()
    matcher, feature_extractor = MATCHERS[args.matcher]
    libraries = load_libraries(feature_extractor)
    board_state = state.BoardState()

    capture = cv.VideoCapture(args.camera)
    if not capture.isOpened():
        print("Cannot open camera")
        return

    try:
        while True:
            ret, frame = capture.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            if process_boards(frame, libraries, matcher, board_state):
                board_state.save(args.output)

            cv.imshow("frame", frame[:, ::-1])
            if cv.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
