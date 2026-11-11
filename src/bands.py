"""Channel names and physically meaningful band groups."""

S1_BANDS = ["ASC_VV", "ASC_VH", "DSC_VV", "DSC_VH"]
S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", "CLP"]

GROUPS = {
    # Sentinel-1
    "S1": S1_BANDS,
    "VV": ["ASC_VV", "DSC_VV"],
    "VH": ["ASC_VH", "DSC_VH"],
    "ASC": ["ASC_VV", "ASC_VH"],
    "DSC": ["DSC_VV", "DSC_VH"],
    # Sentinel-2
    "S2": [b for b in S2_BANDS if b != "CLP"],
    "Visible": ["B2", "B3", "B4"],
    "RedEdge": ["B5", "B6", "B7"],
    "NIR": ["B8", "B8A"],
    "SWIR": ["B11", "B12"],
    "CLP": ["CLP"],
}


def expand(names):
    """Expand a list of band and/or group names into band names (order preserved, no duplicates)."""
    out = []
    for n in names:
        for b in GROUPS.get(n, [n]):
            if b not in S1_BANDS + S2_BANDS:
                raise ValueError(f"Unknown band or group: {n}")
            if b not in out:
                out.append(b)
    return out
