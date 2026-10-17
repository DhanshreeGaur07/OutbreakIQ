"""Console + rotating-file logging, configured once at process start."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(log_dir: str | Path, verbose: bool = False) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(FMT, datefmt="%H:%M:%S"))
    root.addHandler(console)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    fh = RotatingFileHandler(Path(log_dir) / "pipeline.log", maxBytes=5_000_000, backupCount=3)
    fh.setFormatter(logging.Formatter(FMT))
    root.addHandler(fh)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
