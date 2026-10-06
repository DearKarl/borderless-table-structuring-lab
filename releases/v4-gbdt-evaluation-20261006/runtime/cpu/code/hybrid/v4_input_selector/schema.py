"""Deployment-safe records and a strict numeric feature contract."""
from dataclasses import dataclass
import math
import re

SEED = 20261002
RANDOM_CONTROL_SEEDS = tuple(range(SEED, SEED + 5))
GBDT_PARAMETERS = dict(loss="squared_error", learning_rate=.05, max_iter=100,
    max_leaf_nodes=7, min_samples_leaf=10, l2_regularization=1,
    random_state=SEED, early_stopping=False, warm_start=False)
FEATURES = ("width", "height", "aspect", "glyph_height", "glyph_reliability",
    "blur", "contrast", "whitespace", "density", "action_code",
    "preflight_grid_t", "preflight_grid_h", "preflight_grid_w", "is_original_pdf")
PDF_FEATURES = ("pdf_has_text_layer", "pdf_font_size", "pdf_character_count")
ACTION_CODES = {"A": 0, "B": 1, "C": 2}

@dataclass(frozen=True)
class SourceRecord:
    original_file_sha256: str
    source_type: str
    original_page_ordinal: int
    oriented_size: tuple
    pdf_cropbox: tuple | None = None
    pdf_rotation: int | None = None
    pdf_content_type: str = "unknown"

    def __post_init__(self):
        if not re.fullmatch(r"[0-9a-f]{64}", self.original_file_sha256):
            raise ValueError("Original SHA256 required")
        if self.source_type not in ("original_pdf", "raster"):
            raise ValueError("Unknown original source type")
        if self.original_page_ordinal < 0 or len(self.oriented_size) != 2:
            raise ValueError("Invalid source geometry")
        if not all(math.isfinite(x) and x > 0 for x in self.oriented_size):
            raise ValueError("Invalid oriented dimensions")
        if self.source_type == "raster" and self.original_page_ordinal != 0:
            raise ValueError("Raster input has one page")
        if self.source_type == "original_pdf":
            if self.pdf_cropbox is None or len(self.pdf_cropbox) != 4 or self.pdf_rotation not in (0,90,180,270):
                raise ValueError("Original PDF CropBox and rotation required")

    def available_actions(self):
        return ("A", "B", "C") if self.source_type == "original_pdf" else ("A", "B")

def numeric_features(values, *, allow_pdf_text=False):
    names = FEATURES + (PDF_FEATURES if allow_pdf_text else ())
    if set(values) != set(names):
        raise ValueError("Missing or forbidden feature fields")
    result = [float(values[k]) for k in names]
    if not all(math.isfinite(v) for v in result):
        raise ValueError("Nonfinite feature")
    return result

def choose_action(gains, available, fixed_action, margin, *, features_valid=True):
    """Select only supplied available actions; no inference or evaluation access."""
    if fixed_action not in available:
        raise ValueError("Fixed fallback is unavailable")
    if not features_valid or not gains:
        return fixed_action
    if not math.isfinite(margin) or margin not in (0, .25, .5):
        raise ValueError("Unfrozen margin")
    if set(gains) - set(available) or any(not math.isfinite(v) for v in gains.values()):
        return fixed_action
    best = max(sorted(gains), key=lambda a: gains[a])
    return best if gains[best] > max(0, margin) else fixed_action

def fit_row_weight(family_count, pages_in_family, available_actions):
    if min(family_count, pages_in_family, available_actions) <= 0:
        raise ValueError("Positive counts required")
    return 1 / (family_count * pages_in_family * available_actions)

