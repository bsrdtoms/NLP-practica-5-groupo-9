# Carga del corpus de texto
#
# PLN 2025/2026 (FDI UCM)

from pathlib import Path


def load_corpus(path="alicia"):
    """Lee todos los .txt del directorio y los concatena."""
    p = Path(path)
    files = sorted(p.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No se encontraron .txt en {path}")
    return "\n\n".join(f.read_text(encoding="utf-8") for f in files)
