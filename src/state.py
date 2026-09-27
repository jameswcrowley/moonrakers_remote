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
        self.used_cards = {}  # zone_name -> card_ids ever drawn, so reroll/clear never reuses them

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

    def initialize_zone(self, zone_name, slot_count):
        """Ensure a zone has a stable number of empty card slots."""
        previous = self.zones.get(zone_name)
        if previous and len(previous["cards"]) == slot_count:
            return False
        return self.update(zone_name, [(None, 0.0)] * slot_count)

    def set_slot(self, zone_name, slot_index, slot_count, card_id, confidence=1.0):
        """Set one card slot, initializing the zone to its configured size."""
        if not 0 <= slot_index < slot_count:
            raise IndexError(f"Slot index {slot_index} is outside zone '{zone_name}'.")
        self.initialize_zone(zone_name, slot_count)
        cards = [
            (card["card_id"], card["confidence"])
            for card in self.zones[zone_name]["cards"]
        ]
        cards[slot_index] = (card_id, confidence if card_id is not None else 0.0)
        return self.update(zone_name, cards)

    def mark_used(self, zone_name, card_id):
        """Record a card as drawn so a later reroll/clear can't reuse it."""
        used = self.used_cards.setdefault(zone_name, [])
        if card_id not in used:
            used.append(card_id)

    def to_dict(self):
        return {"zones": self.zones, "used_cards": self.used_cards}

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
        instance.used_cards = data.get("used_cards", {})
        return instance
