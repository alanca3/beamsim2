"""Preferences dialog: logging settings + NumCalc binary path.

Provides the "Settings → Preferences…" dialog.  On OK it persists changes
via the section-preserving ``config.update_settings`` layer and immediately
applies the logging configuration so the current session reflects the change.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class PreferencesDialog(QDialog):
    """Preferences dialog: logging settings + NumCalc binary path.

    Sections
    --------
    Logging
        Enable/disable logging, choose level (DEBUG/INFO/WARNING/ERROR),
        and optionally write to a log file.
    NumCalc
        The absolute path to the NumCalc binary.  Validated on accept.

    Usage
    -----
    ::

        dlg = PreferencesDialog(parent=main_window)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            # settings persisted; logging already re-applied
            pass
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setMinimumWidth(480)

        layout = QVBoxLayout(self)

        # ------------------------------------------------------------------
        # Logging group
        # ------------------------------------------------------------------
        log_box = QGroupBox("Logging")
        log_form = QFormLayout(log_box)

        self._log_enable = QCheckBox("Enable logging")
        log_form.addRow("", self._log_enable)

        self._log_level = QComboBox()
        for lvl in ("DEBUG", "INFO", "WARNING", "ERROR"):
            self._log_level.addItem(lvl)
        log_form.addRow("Level:", self._log_level)

        logfile_row = QHBoxLayout()
        self._log_file = QLineEdit()
        self._log_file.setPlaceholderText("(no file — console only)")
        logfile_row.addWidget(self._log_file)
        browse_log = QPushButton("Browse…")
        browse_log.clicked.connect(self._browse_logfile)
        logfile_row.addWidget(browse_log)
        clear_log = QPushButton("Clear")
        clear_log.clicked.connect(lambda: self._log_file.setText(""))
        logfile_row.addWidget(clear_log)
        log_form.addRow("Log file:", logfile_row)

        self._log_enable.toggled.connect(self._update_log_widgets)
        layout.addWidget(log_box)

        # ------------------------------------------------------------------
        # NumCalc group
        # ------------------------------------------------------------------
        nc_box = QGroupBox("NumCalc")
        nc_form = QFormLayout(nc_box)

        nc_row = QHBoxLayout()
        self._nc_path = QLineEdit()
        self._nc_path.setPlaceholderText("(not configured)")
        nc_row.addWidget(self._nc_path)
        browse_nc = QPushButton("Browse…")
        browse_nc.clicked.connect(self._browse_numcalc)
        nc_row.addWidget(browse_nc)
        nc_form.addRow("Binary path:", nc_row)

        nc_note = QLabel(
            "The NumCalc binary is the solver that runs BEM simulations.\n"
            "See docs/SETUP_NOTES.md for the correct path on this machine."
        )
        nc_note.setWordWrap(True)
        nc_note.setStyleSheet("color: #888888; font-size: 11px;")
        nc_form.addRow("", nc_note)
        layout.addWidget(nc_box)

        # ------------------------------------------------------------------
        # OK / Cancel
        # ------------------------------------------------------------------
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # ------------------------------------------------------------------
        # Load current settings
        # ------------------------------------------------------------------
        self._load_current()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_current(self) -> None:
        """Populate widgets from the persisted settings."""
        from beamsim2.backends.numcalc.config import (
            read_logging_prefs,
            resolve_numcalc_binary,
        )

        prefs = read_logging_prefs()
        self._log_enable.setChecked(prefs["enabled"])
        idx = self._log_level.findText(prefs["level"])
        if idx >= 0:
            self._log_level.setCurrentIndex(idx)
        self._log_file.setText(prefs["logfile"])

        try:
            current_bin = resolve_numcalc_binary()
            self._nc_path.setText(current_bin)
        except FileNotFoundError:
            self._nc_path.setText("")

        self._update_log_widgets(prefs["enabled"])

    def _update_log_widgets(self, enabled: bool) -> None:
        """Grey out level / file controls when logging is disabled."""
        self._log_level.setEnabled(enabled)
        self._log_file.setEnabled(enabled)

    def _browse_logfile(self) -> None:
        """Open a save-file dialog to choose the log file location."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Choose log file",
            str(Path.home()),
            "Log files (*.log *.txt);;All files (*)",
        )
        if path:
            self._log_file.setText(path)

    def _browse_numcalc(self) -> None:
        """Open a file-picker to locate the NumCalc binary."""
        start = os.path.expanduser("~")
        if self._nc_path.text().strip():
            parent_dir = str(Path(self._nc_path.text().strip()).parent)
            if os.path.isdir(parent_dir):
                start = parent_dir
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Locate NumCalc Binary",
            start,
            "NumCalc binary (NumCalc);;All files (*)",
        )
        if path:
            self._nc_path.setText(path)

    def _on_accept(self) -> None:
        """Validate, persist, and apply settings; close on success."""
        from beamsim2.backends.numcalc.config import (
            _write_numcalc_config,
            write_logging_prefs,
        )
        from beamsim2.core.logging_setup import configure_logging

        # --- NumCalc binary -----------------------------------------------
        nc_path = self._nc_path.text().strip()
        if nc_path and not os.path.isfile(nc_path):
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.warning(
                self,
                "Invalid path",
                f"NumCalc binary not found at:\n{nc_path}\n\n"
                "Please pick a valid file or leave the field empty.",
            )
            return

        if nc_path:
            _write_numcalc_config(nc_path)
            os.environ["BEAMSIM2_NUMCALC_BIN"] = nc_path

        # --- Logging -------------------------------------------------------
        enabled = self._log_enable.isChecked()
        level_name = self._log_level.currentText()
        logfile = self._log_file.text().strip()

        write_logging_prefs(enabled=enabled, level=level_name, logfile=logfile)

        if enabled:
            level = getattr(logging, level_name, logging.INFO)
            configure_logging(level, logfile=logfile if logfile else None, console=True)
        else:
            # Silence logging: remove owned handlers, set level above any real record.
            configure_logging(logging.CRITICAL, logfile=None, console=False)

        self.accept()
