from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "pyproject.toml").exists():
            return p
    return Path.cwd()


def resolve_path(p: str | Path) -> Path:
    """Absolute -> as is. Relative -> relative to CWD if it exists, else relative to the repository root."""
    p = Path(p).expanduser()
    if p.is_absolute() or p.exists():
        return p
    alt = project_root() / p
    return alt if alt.exists() else p
