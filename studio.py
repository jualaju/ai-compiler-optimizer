"""Launch the local Compiler Studio using the existing project environment."""
import importlib.util
import os
from pathlib import Path
import sys

if importlib.util.find_spec('torch') is None:
    python = Path(__file__).resolve().parent / '.venv' / 'bin' / 'python'
    if python.is_file() and Path(sys.prefix).resolve() != python.parent.parent:
        os.execv(str(python), [str(python), str(Path(__file__).resolve()), *sys.argv[1:]])

from studio.server import main

if __name__ == '__main__':
    main()
