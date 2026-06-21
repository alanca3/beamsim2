"""IO: on-disk format readers and writers — HDF5 (native), VituixCAD .frd, SOFA, and CLF (deferred).

Public API
----------
``write_dataset`` / ``read_dataset``   — native HDF5 (§3.6 contract)
``write_frd``                          — VituixCAD .frd per-driver/per-angle text files
``write_sofa``                         — SOFA AES69 GeneralTF (multi-driver directivity)
``write_clf``                          — CLF balloon (raises NotImplementedError; deferred)
``export_filter_design``               — Phase-2 audit export (filtered/combined .frd + weights)
``load_design_weights``                — reload exported weights to reconstruct a beam
"""

from beamsim2.io.clf_export import write_clf
from beamsim2.io.filter_export import export_filter_design, load_design_weights
from beamsim2.io.frd_export import write_frd
from beamsim2.io.hdf5_store import read_dataset, write_dataset
from beamsim2.io.sofa_export import write_sofa

__all__ = [
    "write_dataset",
    "read_dataset",
    "write_frd",
    "write_sofa",
    "write_clf",
    "export_filter_design",
    "load_design_weights",
]
