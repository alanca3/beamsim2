"""CAD import: STEP/STL/OBJ via gmsh/OpenCASCADE.

**Deferred** — user-confirmed deferral (build-order item 5 planning, 2026-06-18).
The parametric path (``geometry.assemble``, ``geometry.primitives``) covers
all Stage-0 and Stage-1 use cases.

To import an enclosure from a STEP/STL file, implement:
1. Load via ``gmsh.model.occ.importShapes(path)`` (STEP/IGES) or
   ``gmsh.merge(path)`` (STL/OBJ).
2. Report bounding box + prompt user to confirm scale (mm vs. m).
3. Check for non-watertight surfaces; surface them with located plain-English errors.
4. Return tagged Mesh + BoundaryConditions with driver faces marked by the user
   (GUI face-picking or programmatic face-ID override).

Build-order item 5 (DR-03, Stage A import path).
"""

from __future__ import annotations


def load_step(path: str) -> None:
    """Load a STEP file as a BEM enclosure mesh.

    Not yet implemented.  Use ``geometry.assemble.assemble_box_driver`` for
    parametric enclosures.

    Raises
    ------
    NotImplementedError
        Always.
    """
    raise NotImplementedError(
        "CAD import is deferred. "
        "Use geometry.assemble.assemble_box_driver for parametric enclosures. "
        "STEP/STL/OBJ import is planned as a follow-up to build-order item 5."
    )


def load_stl(path: str) -> None:
    """Load an STL file as a BEM enclosure mesh.

    Not yet implemented.

    Raises
    ------
    NotImplementedError
        Always.
    """
    raise NotImplementedError(
        "CAD import is deferred. "
        "Use geometry.assemble.assemble_box_driver for parametric enclosures."
    )
