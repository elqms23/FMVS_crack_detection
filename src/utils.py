import os

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def unique_path(outdir: str, base: str, ext: str = ".png") -> str:
    path = os.path.join(outdir, base + ext)
    if not os.path.exists(path):
        return path
    i = 2
    while True:
        p = os.path.join(outdir, f"{base}_{i}{ext}")
        if not os.path.exists(p):
            return p
        i += 1