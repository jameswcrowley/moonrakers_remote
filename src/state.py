# Tracks the current best-guess board state (per zone) and persists it to JSON.

import json
import time


class BoardState:
    """
    Current best-guess cards occupying each hard-coded zone, across all boards.

    zones maps zone_name -> {"cards": [{"card_id": str or None, "confidence": float}, ...], "last_updated": epoch seconds}.
    One entry in "cards" per card slot detected in the zone that frame.
    """

    def __init__(self):
        self.zones = {}

    def update(self, zone_name, cards):
        """
        Record the latest observation for a zone.

        Args:
            zone_name (str): Which zone this observation is for.
            cards (list): List of (card_id, confidence) tuples, one per card
                slot detected in the zone this frame (empty if none detected).

        Returns:
            bool: True if this changed the zone's recorded card_ids (i.e. the
            state is now "dirty" and worth persisting).
        """
        previous = self.zones.get(zone_name)
        previous_ids = [card["card_id"] for card in previous["cards"]] if previous else None
        new_ids = [card_id for card_id, _ in cards]
        changed = previous_ids != new_ids
        self.zones[zone_name] = {
            "cards": [{"card_id": card_id, "confidence": confidence} for card_id, confidence in cards],
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
