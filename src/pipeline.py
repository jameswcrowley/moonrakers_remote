# Ties camera capture, ArUco board rectification, per-zone card detection, and
# library matching together into a single loop that persists the current
# board state to a JSON file.

import argparse
import os
import time

import cv2 as cv

try:
    from . import boardrectifier, camera, library, recognition, state
except ImportError:  # running as a script, or with src/ on sys.path directly
    import boardrectifier
    import camera
    import library
    import recognition
    import state


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


def process_boards(frame, libraries, matcher, board_state: state.BoardState, board_stabilizers, zone_stabilizers):
    """Detect, rectify, and match every configured board/zone for one frame.

    Args:
        board_stabilizers (dict): board_id -> camera.ContourStabilizer, persisted
            across frames so board corners are smoothed over time.
        zone_stabilizers (dict): zone name -> camera.ContourStabilizer, persisted
            across frames so each zone's card outlines are smoothed over time.

    Returns:
        bool: True if any zone's recorded cards changed this frame.
    """

    aruco_corners, aruco_ids = boardrectifier.detect_aruco_markers(frame)

    dirty = False # flag for whether any zone's recorded cards changed this frame
    for board_config in boardrectifier.BOARD_CONFIGS.values():
        board_stabilizer = board_stabilizers.setdefault(board_config.board_id, camera.ContourStabilizer())
        rectified = boardrectifier.rectify_board(frame, aruco_corners, aruco_ids, board_config, stabilizer=board_stabilizer)
        if rectified is None:
            continue

        for zone in board_config.zones:
            print(f"Processing zone: {zone.name}")
            card_library = libraries.get(zone.library_type)
            if card_library is None or not card_library.cards:
                continue

            # draw box on rectified board for this zone:
            x, y, w, h = zone.roi
            cv.rectangle(rectified, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.putText(rectified, zone.name, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 5)
            cv.imshow("rectified_board_debug", rectified)

            zone_stabilizer = zone_stabilizers.setdefault(zone.name, camera.ContourStabilizer())
            card_images_and_outlines = boardrectifier.identify_card_subarea(rectified, zone, stabilizer=zone_stabilizer)
            print(f"Identified {len(card_images_and_outlines) if card_images_and_outlines else 0} card(s) in zone: {zone.name}")

            if card_images_and_outlines is None:
                changed = board_state.update(zone.name, [])
                dirty = dirty or changed
                continue
            else:
                cards = []
                for i, card_image in enumerate(card_images_and_outlines):
                    card_image, (cx, cy, cw, ch) = card_image
                    cv.imshow(f"card_debug_{zone.name}_{i}", card_image)
                    result = recognition.match_against_library(card_image, card_library, matcher)
                    card_id, confidence = result.best_match()
                    if confidence < MIN_MATCH_CONFIDENCE:
                        card_id = None

                    cards.append((card_id, confidence))

                changed = board_state.update(zone.name, cards)
                dirty = dirty or changed

    return dirty


def parse_arguments():
    """Parse camera and pipeline options."""
    parser = argparse.ArgumentParser(description="Track Moonrakers board state from a camera feed.")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 1).")
    parser.add_argument("--matcher", choices=list(MATCHERS.keys()), default="sift", help="Feature matcher to use.")
    parser.add_argument("--output", default="board_state.json", help="Path to write the board state JSON.")
    return parser.parse_args()


def main():
    """Run the capture/rectify/match loop until the user presses q, saving state on change."""
    args = parse_arguments()
    matcher, feature_extractor = MATCHERS[args.matcher]
    libraries = load_libraries(feature_extractor)
    board_state = state.BoardState()
    board_stabilizers = {}
    zone_stabilizers = {}

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

            if process_boards(frame, libraries, matcher, board_state, board_stabilizers, zone_stabilizers):
                board_state.save(args.output)
                print("Board state updated.")

            cv.imshow("frame", frame[:, ::-1])
            if cv.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
