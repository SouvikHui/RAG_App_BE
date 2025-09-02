"""
Backend package initializer.
This allows imports like `from backend.models import ...`
"""

# You could also re-export common classes if you want:
from backend.models import URLRequest, QueryRequest, QueryResponse, YT_URL
from typing import List, Optional
import os