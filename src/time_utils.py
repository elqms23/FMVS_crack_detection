from typing import Tuple, List

def parse_hms_to_seconds(s: str) -> float:
    """
    Accepts:
      - "H:M:S"
      - "M:S"
      - "S"
    """
    parts = s.strip().split(":")
    parts = [p.strip() for p in parts if p.strip() != ""]
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        m = int(parts[0])
        sec = float(parts[1])
        return m * 60 + sec
    if len(parts) == 3:
        h = int(parts[0])
        m = int(parts[1])
        sec = float(parts[2])
        return h * 3600 + m * 60 + sec
    raise ValueError(f"Invalid time format: {s}")

def parse_ranges(ranges_str: str):
    """
    "01:00:00-01:10:00,02:00:00-02:05:00"
    -> [(3600,4200),(7200,7500)]
    """
    ranges = []

    parts = ranges_str.split(",")

    for part in parts:
        start_s, end_s = part.split("-")

        start = parse_hms_to_seconds(start_s.strip())
        end = parse_hms_to_seconds(end_s.strip())

        if end <= start:
            raise ValueError(f"Invalid range: {part}")

        ranges.append((start, end))

    return ranges


def hms_from_seconds(total_seconds: float) -> Tuple[int, int, int]:
    if total_seconds < 0:
        total_seconds = 0
    whole = int(total_seconds)
    h = whole // 3600
    m = (whole % 3600) // 60
    s = whole % 60
    return h, m, s


def safe_filename_from_hms(h: int, m: int, s: int) -> str:
    # Windows-safe (':' is not allowed in filenames)
    return f"{h:02d}_{m:02d}_{s:02d}"
