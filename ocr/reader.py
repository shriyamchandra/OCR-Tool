"""EasyOCR reader initialization with GPU auto-detection."""

import streamlit as st
import easyocr


def _detect_gpu() -> bool:
    """Check if a CUDA-capable GPU is available.

    Returns:
        True if CUDA is available, False otherwise.
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


@st.cache_resource
def load_reader(languages: list[str] | None = None) -> easyocr.Reader:
    """Load and cache the EasyOCR reader instance.

    Args:
        languages: List of language codes to support. Defaults to ['en', 'hi'].

    Returns:
        A cached EasyOCR Reader instance.
    """
    if languages is None:
        languages = ['en', 'hi']
    use_gpu = _detect_gpu()
    return easyocr.Reader(languages, gpu=use_gpu)
