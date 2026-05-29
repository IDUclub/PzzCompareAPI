from __future__ import annotations

import copy
import json
import logging
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .runtime_settings import *  # noqa: F401,F403

logger = logging.getLogger("pipeline_modules")


def progress_kwargs(*, leave: bool = False) -> dict[str, Any]:
    """Common tqdm kwargs for cleaner progress bars."""
    return {
        "dynamic_ncols": True,
        "leave": leave,
        "mininterval": 0.5,
        "smoothing": 0.1,
    }


# Re-export text normalization helpers for legacy business modules that still do
# `from .common import *`.
from .text_utils import *  # noqa: F401,F403,E402
