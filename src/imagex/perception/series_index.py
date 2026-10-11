"""Build a seriesuid -> LUNA16 subset index from the extracted directory tree.

After extraction, LUNA16 subsets live under names like:
  data/raw/LUNA16/hf_full/data/subset0/
  data/raw/LUNA16/hf_full/data/subset1/
  ...

Each .mhd file is one series. The filename is the seriesuid.
"""

from __future__ import annotations

from pathlib import Path


def build_subset_index(root: Path) -> dict[str, int]:
    """Walk a LUNA16 root and map each seriesuid to its subset number.

    `root` must contain directories named `subset0` through `subset9`.
    Each subset directory contains .mhd files named `<seriesuid>.mhd`.

    Raises NotADirectoryError if root is missing.
    Raises FileNotFoundError if a subset directory is missing.
    Raises ValueError if a .mhd stem does not look like a UID.
    """
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"not a directory: {root}")

    index: dict[str, int] = {}
    seen: set[str] = set()

    for k in range(10):
        sub = root / f"subset{k}"
        if not sub.is_dir():
            raise FileNotFoundError(f"missing subset directory: {sub}")

        for mhd in sub.glob("*.mhd"):
            seriesuid = mhd.stem
            if not seriesuid or "." not in seriesuid:
                raise ValueError(f"unexpected seriesuid: {mhd.name}")
            if seriesuid in seen:
                raise ValueError(f"duplicate seriesuid across subsets: {seriesuid}")
            seen.add(seriesuid)
            index[seriesuid] = k

    if not index:
        raise ValueError(f"no .mhd files found under {root}")

    return index


def list_seriesuids(index: dict[str, int], subset: int | None = None) -> list[str]:
    """Return seriesuids, optionally filtered to a single subset."""
    if subset is not None and subset not in range(10):
        raise ValueError(f"subset must be 0..9, got {subset}")
    if subset is None:
        return sorted(index.keys())
    return sorted(s for s, k in index.items() if k == subset)


def mhd_path_for(root: Path, seriesuid: str, subset: int) -> Path:
    """Return the expected .mhd path for a seriesuid inside a subset."""
    return Path(root) / f"subset{subset}" / f"{seriesuid}.mhd"
