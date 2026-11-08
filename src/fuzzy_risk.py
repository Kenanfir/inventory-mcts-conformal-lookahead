"""Simplified Mamdani-style fuzzy inference for order-quantity risk adjustment.

Two inputs, one output, triangular membership functions, Sugeno-style weighted
average defuzzification (singleton output sets instead of full Mamdani
centroid-of-area — a documented simplification for speed, not a design
we'd defend as final).

Inputs:
  buffer_ratio: (inventory_position - conformal_lower) / max(upper-lower, eps)
                <0 means already below the calibrated lower bound (stockout
                risk imminent); ~1 means sitting near the calibrated upper
                bound (comfortable).
  rel_width:    (upper - lower) / max(point_forecast, eps), i.e. how wide the
                calibrated interval is relative to the forecast (uncertainty).

Output:
  risk_multiplier: scales the base order-up-to quantity. >1 orders more
  (hedging against stockout / high uncertainty), <1 orders less.
"""

import numpy as np


def _tri(x, a, b, c):
    """Triangular membership, peak at b, zero outside [a, c]."""
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)


def _shoulder_low(x, b, c):
    """Open-ended "low" membership: 1.0 for x <= b (the original triangular
    peak), ramping down to 0.0 at c, 0.0 beyond. Fixes a bug (found in
    independent review) where LOW/NARROW were defined via `_tri` with a
    degenerate or clip-boundary-coincident left endpoint: `_tri`'s `x <= a`
    guard is checked before its `x == b` peak check, so whenever a == b (the
    intended flat shoulder) or the peak sits exactly on the input's clip
    floor, the "peak" value was unreachable and silently evaluated to 0.0
    instead of 1.0 -- collapsing all three memberships to zero at that point
    and tripping `risk_multiplier`'s no-rule-fired fallback to a flat,
    wrong 1.0. See research/M5_EVALUATION.md for how often this fired in
    practice before the fix.
    """
    if x <= b:
        return 1.0
    if x >= c:
        return 0.0
    return (c - x) / (c - b)


def _shoulder_high(x, a, b):
    """Open-ended "high" membership: 0.0 for x <= a, ramping up to 1.0 at b
    (the original triangular peak), 1.0 beyond. See `_shoulder_low` for why
    this replaces a `_tri` call whose peak was unreachable at/beyond the
    input's clip boundary.
    """
    if x <= a:
        return 0.0
    if x >= b:
        return 1.0
    return (x - a) / (b - a)


def _buffer_memberships(x):
    x = np.clip(x, -1.0, 2.0)
    return {
        "LOW": _shoulder_low(x, -0.2, 0.4),
        "MEDIUM": _tri(x, 0.0, 0.6, 1.2),
        "HIGH": _shoulder_high(x, 0.8, 1.6),
    }


def _width_memberships(x):
    x = np.clip(x, 0.0, 1.5)
    return {
        "NARROW": _shoulder_low(x, 0.0, 0.5),
        "MEDIUM": _tri(x, 0.2, 0.55, 0.9),
        "WIDE": _shoulder_high(x, 0.6, 1.5),
    }


# Output singleton values per rule consequent.
_OUT = {"LOW": 0.6, "MEDIUM": 1.0, "HIGH": 1.4, "VERY_HIGH": 1.7}


def risk_multiplier(buffer_ratio: float, rel_width: float) -> float:
    b = _buffer_memberships(buffer_ratio)
    w = _width_memberships(rel_width)

    # Rule base (each rule: firing strength = min of antecedents, Mamdani AND).
    rules = [
        (min(b["LOW"], 1.0), "VERY_HIGH"),          # near/below lower bound -> order a lot more
        (min(b["MEDIUM"], w["WIDE"]), "HIGH"),       # mid buffer but uncertain -> hedge up
        (min(b["MEDIUM"], w["MEDIUM"]), "MEDIUM"),
        (min(b["MEDIUM"], w["NARROW"]), "MEDIUM"),
        (min(b["HIGH"], w["NARROW"]), "LOW"),        # comfortable buffer, confident -> order less
        (min(b["HIGH"], w["WIDE"]), "MEDIUM"),       # comfortable buffer but uncertain -> don't cut too much
    ]

    num = 0.0
    den = 0.0
    for strength, label in rules:
        if strength <= 0:
            continue
        num += strength * _OUT[label]
        den += strength

    if den == 0.0:
        return 1.0  # no rule fired (shouldn't happen given full membership coverage) -> neutral
    return num / den
