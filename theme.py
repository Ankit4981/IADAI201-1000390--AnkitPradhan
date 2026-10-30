"""Colour palettes for ParkVision AI (light = the approved design, dark = matching night palette).

Every text/background pair that the app uses is listed in PAIRS and verified by
check_contrast.py (WCAG AA: 4.5:1 for text).
"""

LIGHT = {
    "bg": "#EEF3F1", "card": "#FFFFFF", "border": "#D5E0DC",
    "ink": "#14201D", "muted": "#42544F",
    "accent": "#0F6B5C", "btn_ink": "#FFFFFF",
    "chip_bg": "#D9F0E8", "chip_ink": "#0B4F43",
    "reco_bg": "#0F3D35", "reco_ink": "#FFFFFF", "reco_sub": "#D9EDE6", "reco_lbl": "#B8DDD2",
    "low": "#BFE8D0", "mod": "#F3D58A", "high": "#E8A0A0",
    "lot": "#2A3236", "blob1": "#B5E6D2", "blob2": "#F5E3A8", "blob_op": "0.6",
    "empty_sw": "#1E9E5A", "occ_sw": "#C0392B", "occ_fill": "#F5B7B1",
}

DARK = {
    "bg": "#0D1412", "card": "#16211E", "border": "#2D3E39",
    "ink": "#E8F1EE", "muted": "#A9BDB6",
    "accent": "#3DD9B4", "btn_ink": "#05231C",
    "chip_bg": "#183A31", "chip_ink": "#C3EEE0",
    "reco_bg": "#0A3229", "reco_ink": "#FFFFFF", "reco_sub": "#CFE8E0", "reco_lbl": "#A9D8CA",
    "low": "#2F7157", "mod": "#8F7420", "high": "#9E4E4E",
    "lot": "#0A0F10", "blob1": "#134035", "blob2": "#3B3214", "blob_op": "0.55",
    "empty_sw": "#4ADE80", "occ_sw": "#FF7A7A", "occ_fill": "#6B2F2F",
}

# Fixed (theme-independent) colours: congestion badges and the dark "analysing" panel.
BADGES = {
    "Low": ("#BFE8D0", "#0B3B2D"),
    "Moderate": ("#F3D58A", "#4A3300"),
    "High": ("#E8A0A0", "#4A0F0F"),
}
PANEL = {"bg": "#14201D", "text": "#FFFFFF", "sub": "#B8DDD2"}

# (foreground, background) keys that must reach 4.5:1
PAIRS = [
    ("ink", "bg"), ("ink", "card"), ("muted", "bg"), ("muted", "card"),
    ("accent", "bg"), ("accent", "card"), ("btn_ink", "accent"),
    ("chip_ink", "chip_bg"), ("reco_ink", "reco_bg"), ("reco_sub", "reco_bg"),
    ("reco_lbl", "reco_bg"),
]