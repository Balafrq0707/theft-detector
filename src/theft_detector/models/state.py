from enum import Enum


class ParcelState(str, Enum):
    UNKNOWN = "unknown"

    VISIBLE = "visible"

    INTERACTING = "interacting"

    MISSING = "missing"

    SUSPICIOUSLY_REMOVED = "suspiciously_removed"