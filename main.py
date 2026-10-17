#!/usr/bin/env python
"""Single entry point for the OutbreakIQ ML pipeline.

    python main.py all                  # ingest -> preprocess -> features -> train -> predict
    python main.py ingest [--mobility]  # any stage can be run on its own
    python main.py train
    python main.py predict [--as-of 2022-03-01] [--run-id 20260101T000000Z]
    python main.py all --synthetic      # offline demo on simulated data (isolated paths)
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

from src.config import Config
from src.pipeline import stages
from src.utils.logging_utils import setup_logging

ORDER = ["ingest", "preprocess", "features", "train", "predict"]
log = logging.getLogger("main")


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=ORDER + ["all"])
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--synthetic", action="store_true", help="use simulated data and isolated output dirs")
    ap.add_argument("--mobility", action="store_true", help="also fetch Google Mobility (large) and use it")
    ap.add_argument("--force-download", action="store_true")
    ap.add_argument("--skip-seir", action="store_true")
    ap.add_argument("--as-of", help="predict stage: score this date instead of the latest")
    ap.add_argument("--run-id", help="predict stage: use a specific registered model")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = Config.load(args.config, use_mobility=True if args.mobility else None)
    if args.synthetic:
        cfg.for_synthetic()
    setup_logging(cfg.log_dir, args.verbose)

    runners = {
        "ingest": lambda: stages.run_ingest(cfg, args.synthetic, args.mobility, args.force_download),
        "preprocess": lambda: stages.run_preprocess(cfg),
        "features": lambda: stages.run_features(cfg),
        "train": lambda: stages.run_train(cfg),
        "predict": lambda: stages.run_predict(cfg, args.skip_seir, args.as_of, args.run_id),
    }
    todo = ORDER if args.stage == "all" else [args.stage]
    try:
        for name in todo:
            t0 = time.perf_counter()
            log.info("=== stage: %s ===", name)
            runners[name]()
            log.info("=== %s done in %.1fs ===", name, time.perf_counter() - t0)
    except Exception:
        log.exception("stage failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
