"""Run-monitor re-export for backward-compatibility and discoverability.

The ``RunMonitorWidget`` lives in ``parameters_panel`` because it is directly
embedded in ``SimulationTab``.  This module re-exports it so any code that does
``from beamsim2.gui.run_monitor import RunMonitorWidget`` still works.

Build-order item 10 (§6 Gameplan — run-monitor display, §2 Stage E progress).
"""

from beamsim2.gui.parameters_panel import RunMonitorWidget  # noqa: F401
