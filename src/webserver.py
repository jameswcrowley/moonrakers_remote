"""
Local web UI server and validated board-state API.

This script serves the local web interface for the board state and handles the API for 
interacting with the webpage to refresh and retrieve new cards or updates to the board state.
"""

import json
import mimetypes
import os
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

try:
    from . import boardrectifier
except ImportError:  # running as a script, or with src/ on sys.path directly
    import boardrectifier


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _pipeline_module():
    try:
        from . import pipeline
    except ImportError:  # running as a top-level script
        import pipeline
    return pipeline


def create_server(host, port, mode, libraries, board_state, state_path, images_dir):
    """Create a static-file server with a small JSON API for slot updates."""
    lock = threading.RLock()

    class BoardRequestHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(PROJECT_ROOT), **kwargs)

        def _send_json(self, status, payload):
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):
            request_path = urlparse(self.path).path
            if request_path.startswith("/src/server_images/"):
                image_name = request_path.removeprefix("/src/server_images/")
                image_path = (Path(images_dir) / image_name).resolve()
                if Path(image_name).name == image_name and image_path.is_file():
                    content_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
                    self.send_response(200)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(image_path.stat().st_size))
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    with image_path.open("rb") as image_file:
                        self.wfile.write(image_file.read())
                    return
                return super().do_GET()
            if request_path != "/api/state":
                return super().do_GET()

            with lock:
                zones = {}
                for config in boardrectifier.BOARD_CONFIGS.values():
                    for zone in config.zones:
                        zones[zone.name] = {
                            "library_type": zone.library_type,
                            "slot_count": zone.expected_number_cards,
                        }
                payload = {
                    "mode": mode,
                    "zones": board_state.zones,
                    "zone_config": zones,
                    "catalog": {
                        library_type: sorted(card_library.cards)
                        for library_type, card_library in libraries.items()
                    },
                }
            self._send_json(200, payload)

        def do_POST(self):
            if urlparse(self.path).path != "/api/slot":
                self._send_json(404, {"error": "Unknown API endpoint."})
                return
            if mode == "detect":
                self._send_json(409, {"error": "Slot controls are unavailable in detect mode."})
                return

            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 16_384:
                    raise ValueError("Request body must be between 1 and 16384 bytes.")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("Request body must be a JSON object.")
                zone_name = payload.get("zone")
                slot_index = payload.get("slot")
                action = payload.get("action")
                if action == "clear":
                    card_id = None
                    randomize = False
                elif mode == "manual" and action == "select":
                    card_id = payload.get("card_id")
                    if not isinstance(card_id, str):
                        raise ValueError("A card ID is required for manual selection.")
                    randomize = False
                elif mode == "random" and action == "reroll":
                    card_id = None
                    randomize = True
                else:
                    raise ValueError(f"Action '{action}' is not available in {mode} mode.")

                pipeline = _pipeline_module()
                with lock:
                    pipeline.update_card_slot(
                        libraries,
                        board_state,
                        zone_name,
                        slot_index,
                        card_id=card_id,
                        randomize=randomize,
                    )
                    board_state.save(state_path)
                    pipeline.export_server_images(board_state.to_dict(), output_dir=images_dir)
                    updated_zone = board_state.zones[zone_name]
                self._send_json(200, {"ok": True, "zone": zone_name, "state": updated_zone})
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
                self._send_json(400, {"error": str(error)})

    return ThreadingHTTPServer((host, port), BoardRequestHandler)


def start_server_thread(server):
    """Run the HTTP server in a daemon thread and return that thread."""
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread