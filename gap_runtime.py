"""JSONL logging for the synthetic gap-switch experiment."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np


class GapLog:
    def __init__(self, root="logs/gap", source="simulation", config=None):
        self.directory = Path(root)/(datetime.now().strftime("%Y%m%d_%H%M%S_%f")+f"_{os.getpid()}")
        self.directory.mkdir(parents=True, exist_ok=False)
        self.path = self.directory/"events.jsonl"
        self.source, self.sequence = source, 0
        root_dir = Path(__file__).resolve().parent
        hashes = {name: hashlib.sha256((root_dir/name).read_bytes()).hexdigest() for name in
                  ("gap_jump_servo.py", "gap_runtime.py", "gap_simulation.py",
                   "ibvs_jacobian_estimator.py", "servo_control.py")}
        self("configuration", config=config, schema_version=1, code_sha256=hashes,
             python=platform.python_version(), numpy=np.__version__)

    def __call__(self, event, **data):
        from circle_visualization import clean
        self.sequence += 1
        row = clean(dict(event=event, sequence=self.sequence, source=self.source,
                         host_time=time.monotonic(), **data))
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
            stream.flush()
