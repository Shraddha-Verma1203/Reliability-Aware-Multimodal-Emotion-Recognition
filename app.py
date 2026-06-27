"""Root Streamlit entrypoint.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_streamlit_app():
    module_path = Path(__file__).parent / "app" / "streamlit_app.py"
    spec = importlib.util.spec_from_file_location("mer_streamlit_app", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Streamlit app from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_load_streamlit_app().main()
