"""Enhanced OCR Application — Streamlit UI entry point.

Extracts text from images containing Hindi and English text with
layout-aware processing (single-column, multi-column, table).
"""

import streamlit as st
import numpy as np
from PIL import Image

from ocr.reader import load_reader
from ocr.layout import detect_layout
from ocr.processors import process_single_column, process_multi_column, process_table
from utils.text import convert_hindi_numerals_to_english, correct_ocr_errors, sanitize_for_html, highlight_keywords

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Enhanced OCR Application",
    layout="wide",
    page_icon="📝",
)

# ---------------------------------------------------------------------------
# Sidebar — settings and help
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png",
        width=150,
    )
    st.title("OCR Settings")

    # Callback to reset search state when a new file is uploaded
    def _reset_search():
        st.session_state["search_query_input"] = ""
        st.session_state["highlighted_text"] = ""

    uploaded_file = st.file_uploader(
        "📂 Upload an Image",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        help="Upload images containing Hindi and English text.",
        on_change=_reset_search,
        key="file_uploader",
    )

    st.markdown("---")
    st.header("🔍 Search Options")
    enable_regex = st.checkbox(
        "Enable Regex Search",
        value=False,
        help="Allow using regular expressions for advanced search.",
    )

    st.markdown("---")
    if st.button("🗑️ Clear Cache & Reprocess", use_container_width=True,
                 help="Force fresh OCR on next upload — clears all cached results"):
        st.cache_data.clear()
        for key in ["extracted_text", "highlighted_text", "search_query_input"]:
            st.session_state[key] = ""
        st.success("✅ Cache cleared! Re-upload your image to reprocess.")

    st.markdown("---")
    with st.expander("ℹ️ How to Use"):
        st.write(
            """
            1. **Upload an Image**: Upload an image containing Hindi and English text.
            2. **View Images**: The original and processed images are displayed side by side.
            3. **Extracted Text**: The OCR-extracted text appears below the images.
            4. **Search Text**: Use the search bar to find and highlight keywords.
            5. **Download Results**: Download the extracted text with the button below.
            """
        )

# ---------------------------------------------------------------------------
# Main area — title
# ---------------------------------------------------------------------------
st.title("📝 Enhanced OCR with Advanced Layout Detection")
st.markdown(
    """
Upload an image containing text in **Hindi** and **English**.
The application detects the layout and performs OCR while preserving the text structure.
Numerals are displayed in English for consistency.
"""
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
for key, default in [
    ("search_query_input", ""),
    ("highlighted_text", ""),
    ("extracted_text", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# Cached OCR pipeline — prevents re-running OCR on every interaction
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_ocr(image_bytes: bytes) -> tuple[str, np.ndarray, str]:
    """Run the full OCR pipeline on raw image bytes and cache the result.

    Returns:
        (extracted_text, processed_image_array, layout_type)
    """
    image = Image.open(__import__("io").BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)

    layout_type, column_splits = detect_layout(image_np)

    if layout_type == "single-column":
        extracted_text, processed_image = process_single_column(image_np.copy())
    elif layout_type == "multi-column":
        extracted_text, processed_image = process_multi_column(image_np.copy(), column_splits)
    elif layout_type == "table":
        extracted_text, processed_image = process_table(image_np.copy())
    else:
        extracted_text = "Could not determine layout."
        processed_image = image_np.copy()

    extracted_text = convert_hindi_numerals_to_english(extracted_text)
    extracted_text = correct_ocr_errors(extracted_text)
    return extracted_text, processed_image, layout_type

# ---------------------------------------------------------------------------
# Main processing logic
# ---------------------------------------------------------------------------
if uploaded_file is not None:
    # Read file bytes once (hashable → enables caching)
    file_bytes = uploaded_file.getvalue()
    original_image = Image.open(uploaded_file).convert("RGB")

    # Run OCR (cached — won't re-run on search interactions)
    try:
        with st.spinner("🔍 Performing OCR..."):
            extracted_text, processed_image, layout_type = run_ocr(file_bytes)
    except Exception as exc:
        st.error(f"❌ OCR processing failed: {exc}")
        st.stop()

    # Store extracted text in session state
    st.session_state["extracted_text"] = extracted_text
    st.session_state["highlighted_text"] = ""

    # --- Display images side by side ---
    st.markdown("### 📷 Uploaded and Processed Images")
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_image, caption="🖼️ Uploaded Image", use_container_width=True)
    with col2:
        st.write(f"**Detected Layout Type:** {layout_type}")
        st.image(
            processed_image,
            caption="🖼️ Processed Image with Detected Text Regions",
            use_container_width=True,
        )

    # --- Extracted text (HTML-escaped to prevent XSS) ---
    st.markdown("### 📝 Extracted Text")
    safe_text = sanitize_for_html(st.session_state["extracted_text"])
    st.markdown(
        f'<div style="white-space: pre-wrap; font-family: monospace; font-size: 0.95rem; line-height: 1.6;">{safe_text}</div>',
        unsafe_allow_html=True,
    )

    # --- Download button (native Streamlit widget) ---
    st.download_button(
        label="📥 Download Extracted Text",
        data=st.session_state["extracted_text"],
        file_name="extracted_text.txt",
        mime="text/plain",
    )

    # --- Search functionality ---
    st.markdown("---")
    st.header("🔍 Search Extracted Text")

    search_query = st.text_input(
        "🔑 Enter keyword(s) to search:",
        help="Enter multiple keywords separated by commas. Enable regex for advanced searches.",
        key="search_query_input",
    )

    if search_query:
        # Parse keywords
        if enable_regex:
            keywords = [search_query.strip()]
        else:
            keywords = [kw.strip() for kw in search_query.split(",") if kw.strip()]

        # Highlight on the HTML-safe version of the text
        highlighted, match_counts, errors = highlight_keywords(
            safe_text, keywords, use_regex=enable_regex
        )

        # Show errors for invalid regex patterns
        for err_kw in errors:
            st.error(f"❌ Invalid regex pattern: `{err_kw}`")

        st.session_state["highlighted_text"] = highlighted

        # Display match counts
        st.markdown("### **Search Results:**")
        for kw, count in match_counts.items():
            st.write(f"**{kw}**: {count} match{'es' if count != 1 else ''}")

        # Display highlighted text
        st.markdown("### **Highlighted Text:**")
        st.markdown(
            f'<div style="white-space: pre-wrap;">{st.session_state["highlighted_text"].replace(chr(10), "<br>")}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.session_state["highlighted_text"] = ""
