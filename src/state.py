# Tracks the current best-guess board state (per zone) and persists it to JSON.

import json
import time


class BoardState:
    """
    Current best-guess card occupying each hard-coded zone, across all boards.

    zones maps zone_name -> {"card_id": str or None, "confidence": float, "last_updated": epoch seconds}.
    """

    def __init__(self):
        self.zones = {}

    def update(self, zone_name, card_id, confidence):
        """
        Record the latest observation for a zone.

        Returns:
            bool: True if this changed the zone's recorded card_id (i.e. the
            state is now "dirty" and worth persisting).
        """
        previous = self.zones.get(zone_name)
        changed = previous is None or previous["card_id"] != card_id
        self.zones[zone_name] = {
            "card_id": card_id,
            "confidence": confidence,
            "last_updated": time.time(),
        }
        return changed

    def to_dict(self):
        return {"zones": self.zones}

    def save(self, path):
        """Write the current state to a JSON file at path."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path):
        """Load a previously saved state from a JSON file."""
        instance = cls()
        with open(path, "r") as f:
            data = json.load(f)
        instance.zones = data.get("zones", {})
        return instance
