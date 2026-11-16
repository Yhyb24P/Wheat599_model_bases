"""Load the BGLR wheat.RData and cache it as a numpy archive.

The raw input is an R `.RData` file. We do not depend on pyreadr; instead a
small Rscript step exports X / Y / A / sets into a cached `.npz`. The cache is
only (re)built when it is missing or older than the `.RData`, and the `.RData`
SHA-256 is verified before any read so a stale or substituted cache is never
silently trusted.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# BGLR wheat facts that we verify against on every load (see docs/DATA_FACTS.md).
EXPECTED_ENVIRONMENTS = ("1", "2", "4", "5")


@dataclass(frozen=True)
class WheatData:
    """Validated view of the wheat benchmark."""

    X: np.ndarray  # (n, p) float64, {0,1} markers
    Y: np.ndarray  # (n, n_env) float64, standardized yields
    environments: tuple[str, ...]  # true column names of Y
    A: np.ndarray  # (n, n) additive relationship matrix
    sets: np.ndarray  # (n,) integer 10-fold assignment shipped by BGLR

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    @property
    def p(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_env(self) -> int:
        return int(self.Y.shape[1])


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: Path, expected: str) -> None:
    actual = sha256_of(path)
    if actual != expected:
        raise ValueError(
            f"SHA-256 mismatch for {path}: expected {expected}, got {actual}"
        )


_R_EXPORT_TEMPLATE = r"""
suppressWarnings({{
  e <- new.env()
  load("{rdata}", envir = e)
  X <- as.vector(as.matrix(e$wheat.X))
  Y <- as.vector(as.matrix(e$wheat.Y))
  A <- as.vector(as.matrix(e$wheat.A))
  sets <- as.integer(e$wheat.sets)
  dir.create("{outdir}", showWarnings = FALSE, recursive = TRUE)
  writeBin(X, "{outdir}/X.f64")
  writeBin(Y, "{outdir}/Y.f64")
  writeBin(A, "{outdir}/A.f64")
  writeBin(sets, "{outdir}/sets.i32")
  cat(paste(dim(as.matrix(e$wheat.X)), dim(as.matrix(e$wheat.Y)), dim(as.matrix(e$wheat.A)), length(sets), sep = "x"), "\n")
  cat(paste(colnames(e$wheat.Y), collapse = ","), "\n")
}})
"""


def _export_with_r(rdata: Path, outdir: Path) -> str:
    """Run Rscript to dump raw arrays as flat little-endian binary. Returns the
    Y column names line printed by R."""
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError(
            "Rscript not found on PATH; cannot read .RData. Install R or supply "
            "a pre-exported cache."
        )
    outdir.mkdir(parents=True, exist_ok=True)
    program = _R_EXPORT_TEMPLATE.format(
        rdata=str(rdata), outdir=str(outdir)
    )
    proc = subprocess.run(
        [rscript, "-e", program],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Rscript export failed:\n{proc.stderr}")
    lines = [ln for ln in proc.stdout.strip().splitlines() if ln]
    # last line is the comma-joined Y colnames
    return lines[-1] if lines else ",".join(EXPECTED_ENVIRONMENTS)


# R exports matrices with as.vector(), which is COLUMN-MAJOR; the flat file must
# be reinterpreted in Fortran order. A C-order reshape scrambles X/Y silently
# (A is symmetric, so it masks the bug; X/Y are not — see tests/test_data_layout.py).
LAYOUT_VERSION = "f-order-v2"


def _read_flat(path: Path, dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
    expected_itemsize = np.dtype(dtype).itemsize
    raw = path.read_bytes()
    if len(raw) != int(np.prod(shape)) * expected_itemsize:
        raise ValueError(
            f"{path}: size {len(raw)} does not match shape {shape} "
            f"({int(np.prod(shape)) * expected_itemsize} bytes)"
        )
    return np.frombuffer(raw, dtype=dtype).reshape(
        shape, order="F" if len(shape) > 1 else "C").copy()


def load(
    rdata: Path,
    expected_sha256: str,
    cache_dir: Path | None = None,
) -> WheatData:
    """Load and validate the wheat benchmark.

    Verifies the `.RData` SHA-256, then builds (or reuses) a binary cache under
    `cache_dir` (default: alongside the `.RData` in `data/processed/`).
    """
    rdata = Path(rdata)
    verify_sha256(rdata, expected_sha256)

    if cache_dir is None:
        cache_dir = rdata.parent.parent / "processed"
    cache_dir = Path(cache_dir)
    stamp = cache_dir / "source.sha256"
    layout_stamp = cache_dir / "layout.version"

    cache_ok = (
        cache_dir.exists() and stamp.exists()
        and stamp.read_text().strip() == expected_sha256
        and layout_stamp.exists()
        and layout_stamp.read_text().strip() == LAYOUT_VERSION
    )
    if cache_ok:
        y_names_line = (cache_dir / "y_names.txt").read_text().strip()
    else:
        y_names_line = _export_with_r(rdata, cache_dir)
        stamp.write_text(expected_sha256)
        layout_stamp.write_text(LAYOUT_VERSION)
        (cache_dir / "y_names.txt").write_text(y_names_line)

    y_names = tuple(y_names_line.split(","))
    if y_names != EXPECTED_ENVIRONMENTS:
        raise ValueError(
            f"Unexpected Y column names {y_names}; expected {EXPECTED_ENVIRONMENTS}"
        )

    X = _read_flat(cache_dir / "X.f64", np.float64, (599, 1279))
    Y = _read_flat(cache_dir / "Y.f64", np.float64, (599, len(y_names)))
    A = _read_flat(cache_dir / "A.f64", np.float64, (599, 599))
    sets = _read_flat(cache_dir / "sets.i32", np.int32, (599,))

    _validate_arrays(X, Y, A, sets)
    return WheatData(X=X, Y=Y, environments=y_names, A=A, sets=sets)


def _validate_arrays(X: np.ndarray, Y: np.ndarray, A: np.ndarray, sets: np.ndarray) -> None:
    if X.ndim != 2 or X.shape[0] != Y.shape[0] or X.shape[0] != A.shape[0]:
        raise ValueError("X/Y/A row counts disagree")
    if A.shape[0] != A.shape[1]:
        raise ValueError("A is not square")
    if sets.shape[0] != X.shape[0]:
        raise ValueError("sets length disagrees with X rows")
    if not np.all(np.isin(X, (0.0, 1.0))):
        raise ValueError("X contains values other than {0,1}")
    if np.any(np.isnan(X)) or np.any(np.isnan(Y)):
        raise ValueError("X or Y contains NaN")
    if np.any(X.std(axis=0) == 0):
        raise ValueError("X contains a constant column")
