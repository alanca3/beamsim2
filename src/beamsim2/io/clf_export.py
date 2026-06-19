"""CLF balloon-format exporter — DEFERRED.

CLF (Common Loudspeaker Format) supports two on-disk representations:

1. The compiled ``.cf2`` binary — no open-source writer exists.
2. An open tab-delimited text data file — requires loudspeaker data on a *regular*
   lat/lon angular grid (e.g., 5° × 5°).

Our BEM results live on a scattered **Lebedev** quadrature grid.  Writing a faithful
CLF file therefore requires resampling from the Lebedev grid onto a regular
lat/lon grid via spherical-harmonic (SH) interpolation.  That resampling is a
non-trivial step (SH transform, zero-pad to target order, inverse transform to
regular grid) and is out of scope for build-order item 9.

Revisit CLF when a real balloon consumer (e.g., room-acoustics tools that ingest CLF)
is actually needed in the workflow.

References
----------
CLF Group: https://www.clfgroup.org.  VERIFIED (open format description).
DATA_CONTRACT.md §3.6.
"""

from __future__ import annotations

from pathlib import Path

from beamsim2.assembly.tensor import RadiationDataset


def write_clf(
    out_dir: str | Path,
    ds: RadiationDataset,
    **kwargs: object,
) -> None:
    """Write a CLF balloon export.

    .. note::
        **Not implemented.**  CLF export requires resampling the Lebedev sphere
        grid onto a regular lat/lon angular grid (via SH interpolation), and the
        compiled ``.cf2`` binary has no open-source writer.  See module docstring.

    Raises
    ------
    NotImplementedError
        Always.
    """
    raise NotImplementedError(
        "CLF export is deferred.\n"
        "\n"
        "Reason: the open CLF text-data format requires directivity data on a regular\n"
        "lat/lon angular grid (e.g., 5°×5°), but BeamSimII stores results on a scattered\n"
        "Lebedev quadrature grid.  Resampling to a regular grid requires SH interpolation\n"
        "(transform → zero-pad → inverse transform) which is out of scope for item 9.\n"
        "The compiled .cf2 binary also has no open-source writer.\n"
        "\n"
        "Revisit when a CLF balloon consumer is actually needed.  The native HDF5 file\n"
        "(io/hdf5_store.write_dataset) and VituixCAD .frd files (io/frd_export.write_frd)\n"
        "are the working export formats for now."
    )
