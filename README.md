# BeamSimII

BEM-based loudspeaker directivity simulator for beamforming array design.

BeamSimII computes the complex pressure field `H[driver × frequency × direction]` for
multi-driver loudspeaker arrays using boundary element methods (NumCalc/Mesh2HRTF).
The output feeds directly into Phase 2 beamformer optimization.

## Setup

Requires Python 3.12 and [uv](https://github.com/astral-sh/uv).

```bash
uv sync --group dev
```

## Run (headless)

```bash
uv run python -m beamsim2.pipeline.run --help
```

## Docs

See `docs/` for the project overview, first research report, gameplan, data contract, and
coding standards.
