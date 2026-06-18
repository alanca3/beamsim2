"""Parser for NumCalc be.out output files.

Reads per-frequency complex pressure at evaluation-grid nodes and extracts
per-step convergence flags from the NC.out solver log.

File layout (VERIFIED against Mesh2HRTF commit e45d0436a output2hrtf.py
and NC_PostProcessing.cpp lines 598–772):

  be.out/be.<N>/pEvalGrid   — complex pressure at eval-grid nodes, step N (1-based)
  NC.out                    — full solver log; one CGS summary line per step

pEvalGrid format:
  Mesh2HRTF <version>
  <group_id>  <num_nodes>
  <node_id>  <real_pressure>  <imag_pressure>
  ...

Convergence strings (NC_CommonFunctions.cpp lines 1162–1184):
  Success : "CGS solver: number of iterations = <n>, relative error = <e>"
  Failure : "Warning: Maximum number of iterations is reached!"
"""

from __future__ import annotations

import os
import re

import numpy as np

# ── Convergence sentinel strings (verified from NC_CommonFunctions.cpp) ──────
_CONVERGENCE_OK_RE = re.compile(r"CGS solver: number of iterations")
_CONVERGENCE_FAIL_STR = "Maximum number of iterations is reached"


def read_eval_pressure(work_dir: str, n_freq: int, n_obs: int) -> np.ndarray:
    """Parse all per-frequency pEvalGrid files into a complex pressure array.

    In acoustics terms: reads the complex transfer-function values at each
    microphone direction, for every frequency step.

    Parameters
    ----------
    work_dir : str
        Directory containing NC.inp and be.out/. Files are at
        be.out/be.<k>/pEvalGrid for k = 1 … n_freq.
    n_freq : int
        Number of frequency steps (= number of be.out/be.N/ directories).
    n_obs : int
        Expected number of evaluation-grid nodes (observation points).
        Asserted against actual file count to catch silent point-count mismatches.

    Returns
    -------
    np.ndarray, shape [n_freq, n_obs], complex128
        Complex pressure at each observation point for each frequency.
        Pressure is raw — not re-zeroed or minimum-phased (cardinal rule §3.4).

    Raises
    ------
    FileNotFoundError
        If a pEvalGrid file is missing for any step.
    ValueError
        If the number of nodes in a file doesn't match n_obs.
    """
    pressure = np.zeros((n_freq, n_obs), dtype=np.complex128)  # [F, N] complex128

    for step in range(n_freq):
        be_dir = os.path.join(work_dir, "be.out", f"be.{step + 1}")
        fpath = os.path.join(be_dir, "pEvalGrid")

        if not os.path.isfile(fpath):
            raise FileNotFoundError(
                f"NumCalc output missing: {fpath}\n"
                f"Frequency step {step + 1} did not produce output."
            )

        step_pressure = _parse_peval_grid(fpath)

        if len(step_pressure) != n_obs:
            raise ValueError(
                f"pEvalGrid at step {step + 1} has {len(step_pressure)} nodes; "
                f"expected {n_obs}. Eval-grid point count mismatch."
            )

        pressure[step, :] = step_pressure

    return pressure


def read_convergence(work_dir: str, n_freq: int) -> np.ndarray:
    """Parse the NumCalc solver log and return per-step convergence flags.

    NumCalc names the log file NC{istart}-{iend}.out. For a single invocation
    covering steps 1..n_freq the file is NC1-{n_freq}.out. This function searches
    for that file first, then falls back to any NC*.out file in work_dir.

    Splits the log into per-step sections and checks each for the max-iteration
    warning. A step is flagged False (not converged) if its section contains
    the warning string or if its be.out/be.N/ directory is missing.

    Parameters
    ----------
    work_dir : str
        Directory containing the NC log and be.out/.
    n_freq : int
        Number of frequency steps.

    Returns
    -------
    np.ndarray, shape [n_freq], bool
        True where the CGS solver converged within the iteration cap.
    """
    flags = np.zeros(n_freq, dtype=bool)  # [F] bool — default: not converged

    log_path = _find_nc_log(work_dir, n_freq)

    if log_path is None:
        # No log file at all — mark everything failed.
        return flags

    with open(log_path, "r", errors="replace") as fh:
        log_text = fh.read()

    # Split into per-step sections by the "Step <N>" header NumCalc prints.
    # Each section covers one frequency step.
    step_sections = re.split(r"(?=Step\s+\d+)", log_text)

    for step in range(n_freq):
        # Check that the output directory exists as a secondary indicator.
        be_dir = os.path.join(work_dir, "be.out", f"be.{step + 1}")
        if not os.path.isdir(be_dir):
            flags[step] = False
            continue

        # Find the section that mentions "Step <step+1>".
        section = _find_step_section(step_sections, step + 1)
        if section is None:
            flags[step] = False
            continue

        converged_line_found = bool(_CONVERGENCE_OK_RE.search(section))
        max_iter_hit = _CONVERGENCE_FAIL_STR in section

        flags[step] = converged_line_found and not max_iter_hit

    return flags


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _parse_peval_grid(fpath: str) -> np.ndarray:
    """Parse one pEvalGrid file and return complex pressure values.

    Format (NC_PostProcessing.cpp):
        Mesh2HRTF <version>
        <group_id>  <num_nodes>
        <node_id>  <real>  <imag>
        ...

    Parameters
    ----------
    fpath : str
        Absolute path to the pEvalGrid file.

    Returns
    -------
    np.ndarray, shape [N], complex128
        Complex pressures in file order (node_id order).
    """
    pressures: list[complex] = []
    in_data = False

    with open(fpath, "r") as fh:
        for line in fh:
            tokens = line.split()
            if not tokens:
                continue

            # Skip the version header.
            if tokens[0].startswith("Mesh"):
                continue

            # The "group_id  num_nodes" line: exactly 2 integer tokens,
            # first is not a node_id (those come after).
            if not in_data and len(tokens) == 2:
                try:
                    int(tokens[0])
                    int(tokens[1])
                    in_data = True
                    continue
                except ValueError:
                    continue

            # Data lines: node_id  real  imag
            if in_data and len(tokens) >= 3:
                try:
                    re_val = float(tokens[1])
                    im_val = float(tokens[2])
                    pressures.append(complex(re_val, im_val))
                except ValueError:
                    continue

    return np.array(pressures, dtype=np.complex128)


def _find_nc_log(work_dir: str, n_freq: int) -> str | None:
    """Locate the NumCalc solver log file in work_dir.

    NumCalc names the log NC{istart}-{iend}.out. For a single all-frequency
    invocation with -istart 1 -iend n_freq the file is NC1-{n_freq}.out.
    Falls back to any NC*.out file present if the expected name is missing.

    Parameters
    ----------
    work_dir : str
        Directory where NC log files are written.
    n_freq : int
        Number of frequency steps (used to predict the filename).

    Returns
    -------
    str or None
        Absolute path to the log file, or None if not found.
    """
    import glob

    # Primary: the log name for a single -istart 1 -iend n_freq invocation.
    primary = os.path.join(work_dir, f"NC1-{n_freq}.out")
    if os.path.isfile(primary):
        return primary

    # Fallback: any NC*.out in work_dir (handles edge cases or future invocation patterns).
    candidates = sorted(glob.glob(os.path.join(work_dir, "NC*.out")))
    return candidates[0] if candidates else None


def _find_step_section(sections: list[str], step_number: int) -> str | None:
    """Return the log section that mentions 'Step <step_number>'.

    Parameters
    ----------
    sections : list[str]
        Log text split on 'Step N' boundaries.
    step_number : int
        1-based step index to find.

    Returns
    -------
    str or None
    """
    pattern = re.compile(rf"\bStep\s+{step_number}\b")
    for sec in sections:
        if pattern.search(sec):
            return sec
    return None
