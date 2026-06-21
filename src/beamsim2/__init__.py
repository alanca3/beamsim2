"""BeamSimII — BEM-based loudspeaker directivity simulator."""

from beamsim2.core.logging_setup import install_null_handler

# Library discipline: attach a NullHandler to the package logger so modules can
# emit log records before (or without) any application/CLI configure_logging()
# call, without printing to stderr or warning about missing handlers.
install_null_handler()
