#!/usr/bin/env python3

import json
from pathlib import Path
import sys
from .runner import execute_backend

if __name__ == "__main__":
    task = json.loads(Path(sys.argv[1]).read_text())
    execute_backend(Path(task["path"]), task["argv"], task["overrides"])
