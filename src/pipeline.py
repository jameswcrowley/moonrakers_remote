# Ties camera capture, ArUco board rectification, per-zone card detection, and
# library matching together into a single loop that persists the current
# board state to a JSON file.

import argparse
import os
import random
import shutil
import time

import cv2 as cv

try:
    from . import boardrectifier, camera, library, recognition, state, webserver
except ImportError:  # running as a script, or with src/ on sys.path directly
    import boardrectifier
    import camera
    import library
    import recognition
    import state
    import webserver


MODES = ["detect", "manual", "random"]

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

# Mapping zone identifiers to (library_key, output_filename_prefix, max_slots).
# Used to propagate recognized card images to the web server directory.
ZONE_EXPORT_CONFIG = {
    "contract_slot_1": ("contracts", "current_contract", 8),
    "crew_slot_1": ("crew", "current_crew", 3),
    "ship_part_slot_1": ("ship_parts", "current_ship_part", 6),
}

PLACEHOLDER_IMAGES = {
    "contracts": "contracts_placeholder.jpg",
    "crew": "crew_placeholder.jpg",
    "ship_parts": "shipParts_placeholder.jpg",
}
PLACEHOLDER_CARD_IDS = {"contracts_placeholder", "crew_placeholder", "shipParts_placeholder"}


def export_server_images(board_state_data, output_dir="src/server_images", library_dirs=LIBRARY_DIRECTORIES):
    """
    Copy recognized card reference images into the server images directory.

    For each slot in each board zone:
      - If a valid card_id was detected, copies its high-res image from data/saved_images/{zone_type}/.
      - Ensures the destination image exists for web clients to render.
    """
    os.makedirs(output_dir, exist_ok=True)
    zones = board_state_data.get("zones", {})

    # TODO: if confidence is too low, I want to export a placeholder image

    for zone_name, (lib_type, file_prefix, max_slots) in ZONE_EXPORT_CONFIG.items():
        cards = zones.get(zone_name, {}).get("cards", [])
        lib_dir = library_dirs.get(lib_type, "")

        for i in range(max_slots):
            dst_path = os.path.join(output_dir, f"{file_prefix}{i + 1}.jpg")
            card_id = cards[i]["card_id"] if i < len(cards) else None

            source_path = None
            if card_id:
                for ext in [".jpg", ".jpeg", ".png"]:
                    candidate = os.path.join(lib_dir, f"{card_id}{ext}")
                    if os.path.isfile(candidate):
                        source_path = candidate
                        break
            if source_path is None:
                source_path = os.path.join(lib_dir, PLACEHOLDER_IMAGES[lib_type])
            if os.path.isfile(source_path):
                shutil.copyfile(source_path, dst_path)


def load_libraries(feature_extractor):
    """Build one CardLibrary per zone type, keyed the same way as LIBRARY_DIRECTORIES."""
    libraries = {}
    for library_type, directory in LIBRARY_DIRECTORIES.items():
        if not os.path.isdir(directory):
            continue
        card_library = library.CardLibrary.from_directory(directory, feature_extractor=feature_extractor)
        for placeholder_id in PLACEHOLDER_CARD_IDS:
            card_library.cards.pop(placeholder_id, None)
        libraries[library_type] = card_library
    return libraries

def process_board_from_manual(libraries, board_state: state.BoardState, manual_card_input: dict):
    """Apply zone-to-card-ID lists, preserving blank slots as None."""
    dirty = False
    for board_config in boardrectifier.BOARD_CONFIGS.values():
        for zone in board_config.zones:
            card_library = libraries.get(zone.library_type)
            if card_library is None:
                raise ValueError(f"No card library is loaded for '{zone.library_type}'.")
            card_ids = manual_card_input.get(zone.name, [None] * zone.expected_number_cards)
            if len(card_ids) != zone.expected_number_cards:
                raise ValueError(f"Zone '{zone.name}' requires {zone.expected_number_cards} card slots.")
            if any(card_id is not None and card_id not in card_library.cards for card_id in card_ids):
                raise ValueError(f"Zone '{zone.name}' contains an unknown card ID.")
            dirty = board_state.update(
                zone.name,
                [(card_id, 1.0 if card_id is not None else 0.0) for card_id in card_ids],
            ) or dirty
    return dirty


def update_card_slot(libraries, board_state, zone_name, slot_index, card_id=None, randomize=False):
    """Validate and apply a manual, clear, or random slot update."""
    zone = next(
        (zone for config in boardrectifier.BOARD_CONFIGS.values() for zone in config.zones if zone.name == zone_name),
        None,
    )
    if zone is None:
        raise ValueError(f"Unknown zone '{zone_name}'.")
    if not isinstance(slot_index, int) or not 0 <= slot_index < zone.expected_number_cards:
        raise ValueError(f"Invalid slot index for zone '{zone_name}'.")
    card_library = libraries.get(zone.library_type)
    if card_library is None or not card_library.cards:
        raise ValueError(f"No cards are available for '{zone.library_type}'.")

    if randomize:
        current_cards = board_state.zones.get(zone_name, {}).get("cards", [])
        excluded = {
            card["card_id"]
            for card in current_cards
            if card["card_id"] is not None
        }
        excluded |= set(board_state.used_cards.get(zone_name, []))
        options = [candidate for candidate in card_library.cards if candidate not in excluded]
        if not options:
            raise ValueError(f"No unused cards remain for '{zone.library_type}'.")
        card_id = random.choice(options)
        board_state.mark_used(zone_name, card_id)
    elif card_id is not None and card_id not in card_library.cards:
        raise ValueError(f"Unknown card ID for '{zone.library_type}'.")

    changed = board_state.set_slot(
        zone_name,
        slot_index,
        zone.expected_number_cards,
        card_id,
        confidence=1.0,
    )
    return changed, card_id


def initialize_random_board(libraries, board_state):
    """Fill empty configured slots while avoiding duplicates within each zone."""
    dirty = False
    for board_config in boardrectifier.BOARD_CONFIGS.values():
        for zone in board_config.zones:
            board_state.initialize_zone(zone.name, zone.expected_number_cards)
            current = board_state.zones[zone.name]["cards"]
            for slot_index, card in enumerate(current):
                if card["card_id"] is None:
                    changed, _ = update_card_slot(libraries, board_state, zone.name, slot_index, randomize=True)
                    dirty = dirty or changed
    return dirty



def process_boards(frame, libraries, matcher, board_state: state.BoardState, board_stabilizers, zone_stabilizers, card_stabilizers, edge_detection='canny'):
    """Detect, rectify, and match every configured board/zone for one frame.

    Args:
        board_stabilizers (dict): board_id -> camera.ContourStabilizer, persisted
            across frames so board corners are smoothed over time. 
        zone_stabilizers (dict): zone name -> camera.ContourStabilizer, persisted
            across frames so each zone's card outlines are smoothed over time.
        card_stabilizers (dict): (zone name, slot index) -> (BestGuessStabilizer,
            RollingAverageStabilizer), persisted across frames so each card
            slot's identity is smoothed over time instead of flickering frame
            to frame or vanishing on a single dropped detection.
        edge_detection (str): Edge detection method to use for identifying card contours ('canny' or 'threshold').

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
            card_library = libraries.get(zone.library_type)
            if card_library is None or not card_library.cards:
                continue

            # draw box on rectified board for this zone:
            x, y, w, h = zone.roi
            cv.rectangle(rectified, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.putText(rectified, zone.name, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 5)
            cv.imshow("rectified_board_debug", rectified)

            zone_stabilizer = zone_stabilizers.setdefault(zone.name, camera.ContourStabilizer())
            card_images_and_outlines = boardrectifier.identify_card_subarea(rectified, zone, edge_detection=edge_detection, stabilizer=zone_stabilizer)
            print(f"Identified {len(card_images_and_outlines) if card_images_and_outlines else 0} card(s) in zone: {zone.name}")

            # every slot is fed through its own stabilizer, even slots with no
            # observation this frame, so a single bad frame (or a whole-zone
            # detection dropout) doesn't instantly blank out the zone.
            cards = []
            for i in range(zone.expected_number_cards):
                best_guess, rolling_average = card_stabilizers.setdefault(
                    (zone.name, i), (recognition.BestGuessStabilizer(confidence_threshold=MIN_MATCH_CONFIDENCE), recognition.RollingAverageStabilizer())
                )

                observation = (None, 0.0)
                if card_images_and_outlines is not None and i < len(card_images_and_outlines):
                    card_image, (cx, cy, cw, ch) = card_images_and_outlines[i]
                    cv.imshow(f"card_debug_{zone.name}_{i}", card_image)
                    result = recognition.match_against_library(card_image, card_library, matcher)
                    observation = result.best_match()

                    label = f"{observation[0] or 'unknown'}: {observation[1]:.2f}"
                    cv.putText(rectified, label, (int(cx), max(int(cy) - 10, 0)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)

                card_id, confidence = best_guess.update([observation])
                debug_card_id, debug_confidence = rolling_average.update(observation)
                print(f"  slot {i}: held={card_id}:{confidence:.2f} rolling_avg={debug_card_id}:{debug_confidence:.2f}")

                cards.append((card_id, confidence))

            changed = board_state.update(zone.name, cards)
            dirty = dirty or changed

            cv.imshow("rectified_board_debug", rectified)

    return dirty


def parse_arguments():
    """Parse camera and pipeline options."""
    parser = argparse.ArgumentParser(description="Track Moonrakers board state from a camera feed.")
    parser.add_argument("--mode", choices=MODES, default="manual", help="Pipeline mode (default: manual).")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0).")
    parser.add_argument("--edge-detection", type=str, choices=["canny", "threshold"], default="threshold", help="Edge detection method to use (default: canny).")
    parser.add_argument("--matcher", choices=list(MATCHERS.keys()), default="orb", help="Feature matcher to use.")
    parser.add_argument("--output", default="board_state.json", help="Path to write the board state JSON.")
    parser.add_argument("--images-dir", default="src/server_images", help="Path to write web server card images.")
    parser.add_argument("--host", default="127.0.0.1", help="Web server bind address (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8000, help="Web server port (default: 8000).")
    return parser.parse_args()


def prepare_non_detect_mode(mode, libraries, board_state):
    """Initialize configured slots for camera-free modes."""
    if mode == "manual":
        for config in boardrectifier.BOARD_CONFIGS.values():
            for zone in config.zones:
                board_state.initialize_zone(zone.name, zone.expected_number_cards)
        return False
    if mode == "random":
        return initialize_random_board(libraries, board_state)
    raise ValueError(f"Unsupported non-detect mode: {mode}")


def main():
    """Run the configured board mode and serve the live board webpage."""
    args = parse_arguments()
    matcher, feature_extractor = MATCHERS[args.matcher] if args.mode == "detect" else (None, None)
    libraries = load_libraries(feature_extractor)
    if args.mode != "detect" and os.path.isfile(args.output):
        board_state = state.BoardState.load(args.output)
    else:
        board_state = state.BoardState()

    if args.mode != "detect":
        prepare_non_detect_mode(args.mode, libraries, board_state)
        board_state.save(args.output)
        export_server_images(board_state.to_dict(), output_dir=args.images_dir)

    server = webserver.create_server(
        host=args.host,
        port=args.port,
        mode=args.mode,
        libraries=libraries,
        board_state=board_state,
        state_path=args.output,
        images_dir=args.images_dir,
    )
    print(f"Board page: http://{args.host}:{args.port}/src/index.html")

    if args.mode != "detect":
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
        return

    edge_detection = args.edge_detection
    board_stabilizers = {}
    zone_stabilizers = {}
    card_stabilizers = {}
    server_thread = webserver.start_server_thread(server)

    capture = cv.VideoCapture(args.camera)
    if not capture.isOpened():
        print("Cannot open camera")
        server.shutdown()
        return

    try:
        while True:
            ret, frame = capture.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            if process_boards(frame, libraries, matcher, board_state, board_stabilizers, zone_stabilizers, card_stabilizers, edge_detection=edge_detection):
                board_state.save(args.output)
                # Propagate best-guess images to server directory for live web client consumption
                export_server_images(board_state.to_dict(), output_dir=args.images_dir)
                print("Board state and server images updated.")

            cv.imshow("frame", frame[:, :])
            if cv.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv.destroyAllWindows()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)


if __name__ == "__main__":
    main()
