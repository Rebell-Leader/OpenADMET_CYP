#!/usr/bin/env python3
"""Local submission validator for the OpenADMET CYP Inhibition Blind Challenge.

Reimplements every check the HuggingFace Space actually performs on upload
(``submission.py::submit_predictions`` / ``_read_tabular_submission``), so a file can
be cleared before it is uploaded and before the 4-hour-per-track rate limit is spent.

Every rule below is annotated CONFIRMED (a literal check in the Space source) or
STRICTER-THAN-SPACE (a check we add because the FAQ demands it but the Space's code
does not actually enforce it). Two of the latter come from real defects in the Space:

* Classification NaNs pass the Space's check, because it is
  ``df[col].dropna().isin([0, 1, True, False]).all()`` -- the NaNs are dropped before
  testing. The FAQ says columns "must be fully populated for every row".
* A non-numeric regression column makes the Space raise ``TypeError`` inside
  ``np.isfinite`` rather than returning its intended message.

Extra columns are deliberately NOT an error: the Space checks
``set(required) - set(df.columns)``, a one-sided difference. Column order is not
checked either. Column names are compared by exact string equality and so are
case-sensitive.

Usage:
    python validate_submission.py --track regression --file preds.parquet
    python validate_submission.py --track classification --file preds.csv --lenient
    python validate_submission.py --track structure --file poses.zip

Exit code 0 = pass, 1 = fail, 2 = usage error.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# --- CONFIRMED constants, mirrored from the Space's config.py ----------------------

ACTIVITY_DATASET_SIZE = 750
STRUCTURE_DATASET_SIZE = 184
IDENTIFIER_COLUMNS = ["SMILES", "Molecule_Name"]
REGRESSION_ENDPOINTS = [
    "CYP1A2_pIC50_direct_inhibition",
    "CYP2C9_pIC50_direct_inhibition",
    "CYP2D6_pIC50_direct_inhibition",
    "CYP3A4_pIC50_direct_inhibition",
]
CLASSIFICATION_ENDPOINTS = ["CYP2D6_is_TDI", "CYP3A4_is_TDI"]
REQUIRED_REGRESSION_COLUMNS = IDENTIFIER_COLUMNS + REGRESSION_ENDPOINTS
REQUIRED_CLASSIFICATION_COLUMNS = IDENTIFIER_COLUMNS + CLASSIFICATION_ENDPOINTS

TRACKS = ("regression", "classification", "structure")

_REQUIRED = {
    "regression": REQUIRED_REGRESSION_COLUMNS,
    "classification": REQUIRED_CLASSIFICATION_COLUMNS,
}
_ENDPOINTS = {
    "regression": REGRESSION_ENDPOINTS,
    "classification": CLASSIFICATION_ENDPOINTS,
}


@dataclass
class ValidationReport:
    """Outcome of validating one submission file."""

    track: str
    path: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when no error was recorded."""
        return not self.errors

    def error(self, msg: str) -> None:
        """Record a blocking failure."""
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        """Record a non-blocking observation."""
        self.warnings.append(msg)

    def render(self) -> str:
        """Human-readable report."""
        lines = [
            f"track : {self.track}",
            f"file  : {self.path}",
        ]
        for k, v in self.info.items():
            lines.append(f"  {k}: {v}")
        for w in self.warnings:
            lines.append(f"WARN  {w}")
        for e in self.errors:
            lines.append(f"FAIL  {e}")
        lines.append("PASS — file satisfies every rule the Space enforces."
                     if self.ok else
                     f"FAILED — {len(self.errors)} error(s).")
        return "\n".join(lines)


def _read_tabular(path: Path, report: ValidationReport) -> pd.DataFrame | None:
    """Read a .parquet or .csv submission, mirroring ``_read_tabular_submission``."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception as exc:  # CONFIRMED: Space surfaces the exception text
            report.error(f"Could not read parquet file — {exc}")
            return None
    if suffix == ".csv":
        try:
            return pd.read_csv(path)
        except Exception as exc:
            report.error(f"Could not read CSV file — {exc}")
            return None
    report.error(
        f"{report.track.capitalize()} submissions must be a .parquet or .csv file "
        f"(got {path.suffix!r})."
    )
    return None


def validate_tabular(
    path: Path, track: str, *, strict: bool = True, expected_rows: int = ACTIVITY_DATASET_SIZE
) -> ValidationReport:
    """Validate a regression or classification submission file.

    Args:
        path: File to check.
        track: ``"regression"`` or ``"classification"``.
        strict: Apply the STRICTER-THAN-SPACE checks (recommended; the FAQ requires
            them even though the Space's code does not enforce them). With
            ``strict=False`` the validator reproduces the Space's behaviour exactly,
            including its two defects, and downgrades those findings to warnings.
        expected_rows: Row count required. Defaults to the confirmed 750.

    Returns:
        :class:`ValidationReport`.
    """
    report = ValidationReport(track=track, path=str(path))
    if not path.exists():
        report.error(f"file does not exist: {path}")
        return report

    df = _read_tabular(path, report)
    if df is None:
        return report

    report.info["rows"] = len(df)
    report.info["columns"] = list(df.columns)

    # CONFIRMED rule 7: exact row count, both tracks.
    if len(df) != expected_rows:
        report.error(f"Expected {expected_rows} rows, got {len(df)}.")

    # CONFIRMED rule 8: required columns present, exact case-sensitive names.
    required = _REQUIRED[track]
    missing = [c for c in required if c not in df.columns]
    if missing:
        report.error(f"Missing required columns: {set(missing)}")
        # A case-only mismatch is the most common cause; call it out.
        lower_map = {str(c).lower(): c for c in df.columns}
        for m in missing:
            hit = lower_map.get(m.lower())
            if hit is not None and hit != m:
                report.error(
                    f"column {hit!r} looks like a case mismatch for required {m!r} "
                    "— names are case-sensitive."
                )
        return report

    # CONFIRMED rule 9: extra columns are NOT rejected by the Space.
    extra = [c for c in df.columns if c not in required]
    if extra:
        report.warn(
            f"{len(extra)} column(s) beyond the required set: {extra}. The Space's "
            "check is a one-sided set difference, so these are accepted — but the "
            "Submit tab documents a "
            f"{len(required)}-column file, so the backend may be stricter."
        )

    endpoints = _ENDPOINTS[track]

    if track == "regression":
        for col in endpoints:
            s = df[col]
            # STRICTER-THAN-SPACE: the Space calls np.isfinite on the raw column,
            # which raises TypeError on object dtype instead of reporting.
            if not pd.api.types.is_numeric_dtype(s):
                coerced = pd.to_numeric(s, errors="coerce")
                n_bad = int(coerced.isna().sum() - s.isna().sum())
                report.error(
                    f"{col} is dtype {s.dtype} rather than a float dtype "
                    f"({n_bad} value(s) not parseable as a number). The Space would "
                    "raise an uncaught TypeError on this file."
                )
                continue
            # CONFIRMED rule 10.
            if s.isnull().any():
                report.error(
                    f"{col} column contains NaN values "
                    f"({int(s.isnull().sum())} row(s))."
                )
            arr = s.to_numpy(dtype=float)
            if not np.isfinite(arr[~np.isnan(arr)]).all():
                report.error(
                    f"{col} column contains infinite values "
                    f"({int(np.isinf(arr).sum())} row(s))."
                )
            finite = arr[np.isfinite(arr)]
            if finite.size:
                report.info[f"{col} range"] = (
                    f"{finite.min():.3f} .. {finite.max():.3f}"
                )
                # Advisory only: pIC50 is a log-molar potency, so values outside
                # roughly 2..11 are almost certainly a units or sign error.
                if finite.min() < 2 or finite.max() > 11:
                    report.warn(
                        f"{col} has values outside the plausible pIC50 range 2..11 "
                        f"({finite.min():.3f} .. {finite.max():.3f}) — check units "
                        "and sign. Not enforced by the Space."
                    )
    else:  # classification
        for col in endpoints:
            s = df[col]
            n_null = int(s.isnull().sum())
            # CONFIRMED rule 11 (note the dropna() in the Space's version).
            non_binary = s.dropna()
            ok_binary = bool(non_binary.isin([0, 1, True, False]).all())
            if not ok_binary:
                offenders = sorted(
                    {repr(v) for v in non_binary[~non_binary.isin([0, 1, True, False])]}
                )[:5]
                report.error(
                    f"{col} column contains non-binary values (e.g. "
                    f"{', '.join(offenders)})."
                )
            if n_null:
                msg = (
                    f"{col} column contains {n_null} null value(s). The FAQ requires "
                    "every row populated, but the Space drops NaNs before its "
                    "binary check, so this file would be ACCEPTED on upload and "
                    "then likely fail or be mis-scored in the backend."
                )
                # STRICTER-THAN-SPACE.
                report.error(msg) if strict else report.warn(msg)
            counts = s.dropna().astype("object").map(
                lambda v: bool(v) if not isinstance(v, str) else v
            ).value_counts(dropna=False)
            report.info[f"{col} value counts"] = counts.to_dict()
            if s.dropna().nunique() <= 1:
                report.warn(
                    f"{col} predicts a single class — MCC's denominator is 0 for a "
                    "single-class prediction and would score 0.0 under our "
                    "convention (backend behaviour unconfirmed)."
                )

    # Identifier sanity — advisory, the Space checks none of this.
    for col in IDENTIFIER_COLUMNS:
        if df[col].isnull().any():
            report.warn(
                f"{col} has {int(df[col].isnull().sum())} null value(s). Not checked "
                "by the Space, but the backend must join on Molecule_Name."
            )
    if df["Molecule_Name"].duplicated().any():
        n = int(df["Molecule_Name"].duplicated().sum())
        report.warn(
            f"Molecule_Name has {n} duplicate value(s). Not checked by the Space; the "
            "backend joins ground truth on this column, so duplicates are risky."
        )
    return report


def validate_structure(
    path: Path, *, expected_files: int = STRUCTURE_DATASET_SIZE
) -> ValidationReport:
    """Validate a structure-track zip.

    CONFIRMED rule 13: extension ``.zip`` and ``len(ZipFile.namelist()) == 184``.
    ``namelist()`` counts directory entries, so a zip made from a folder reports 185
    and is rejected — we warn about that specifically. The ``.pdb`` naming and the
    ``LIG`` ligand residue name are documented on the Submit tab but NOT enforced by
    the Space; we report them as warnings.
    """
    report = ValidationReport(track="structure", path=str(path))
    if not path.exists():
        report.error(f"file does not exist: {path}")
        return report
    if path.suffix.lower() != ".zip":
        report.error("Structure submissions must be a .zip file.")
        return report
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
    except Exception as exc:
        report.error(f"Could not read zip file — {exc}")
        return report
    report.info["entries"] = len(names)
    if len(names) != expected_files:
        report.error(f"Expected {expected_files} files in zip, got {len(names)}.")
        dirs = [n for n in names if n.endswith("/")]
        if dirs:
            report.warn(
                f"{len(dirs)} directory entry/entries counted toward the total "
                f"({dirs[:3]}). Zip the .pdb files directly, not their parent folder."
            )
    non_pdb = [n for n in names if not n.lower().endswith(".pdb") and not n.endswith("/")]
    if non_pdb:
        report.warn(
            f"{len(non_pdb)} entry/entries are not .pdb (e.g. {non_pdb[:3]}). Not "
            "enforced by the Space, but the Submit tab requires one .pdb per compound."
        )
    return report


def validate(path: Path, track: str, *, strict: bool = True) -> ValidationReport:
    """Dispatch to the right validator for ``track``."""
    if track not in TRACKS:
        raise ValueError(f"track must be one of {TRACKS}, got {track!r}")
    if track == "structure":
        return validate_structure(path)
    return validate_tabular(path, track, strict=strict)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    p = argparse.ArgumentParser(
        description=(
            "Validate an OpenADMET CYP challenge submission against every rule the "
            "HuggingFace Space enforces on upload."
        )
    )
    p.add_argument("--track", required=True, choices=TRACKS)
    p.add_argument("--file", required=True, type=Path)
    p.add_argument(
        "--lenient",
        action="store_true",
        help=(
            "Reproduce the Space's behaviour exactly, including its two known "
            "defects (classification NaNs pass). Default is stricter."
        ),
    )
    p.add_argument(
        "--quiet", action="store_true", help="Print only the final PASS/FAILED line."
    )
    args = p.parse_args(argv)

    report = validate(args.file, args.track, strict=not args.lenient)
    text = report.render()
    print(text.splitlines()[-1] if args.quiet else text)
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
