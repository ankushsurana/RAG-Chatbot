"""ragbot: low-latency RAG over large private PDF corpora."""

import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

__version__ = "1.0.0"
