import cv2
import re
import streamlit as st
import easyocr
from PIL import Image
import numpy as np
import base64
import io

# Set page configuration
st.set_page_config(
    page_title="Enhanced OCR Application",
    layout="wide",
    page_icon="📝"
)

# Sidebar for navigation and settings
with st.sidebar:
    # Display a logo or image (optional)
    st.image("https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png", width=150)
    st.title("OCR Settings")
    
    # Callback function to reset search query and highlighted text
    def reset_search():
        st.session_state['search_query_input'] = ''
        st.session_state['highlighted_text'] = ''
    
    # File uploader with callback to reset search
    uploaded_file = st.file_uploader(
        "📂 Upload an Image",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        help="Upload images containing Hindi and English text.",
        on_change=reset_search,
        key="file_uploader"
    )
    
    st.markdown("---")
    st.header("🔍 Search Options")
    enable_regex = st.checkbox(
        "Enable Regex Search",
        value=False,
        help="Allow using regular expressions for advanced search."
    )
    
    st.markdown("---")
    with st.expander("ℹ️ How to Use"):
        st.write("""
            1. **Upload an Image**: Upload an image containing Hindi and English text.
            2. **View Images**: The original and processed images will be displayed side by side.
            3. **Extracted Text**: The OCR-extracted text will appear below the images.
            4. **Search Text**: Use the search bar to find and highlight keywords in the extracted text.
            5. **Download Results**: Download the extracted text using the provided button.
        """)

# Title and Instructions in the main area
st.title("📝 Enhanced OCR with Advanced Layout Detection")
st.markdown("""
Upload an image containing text in **Hindi** and **English**.
The application detects the layout and performs OCR while preserving the text structure.
Numerals are displayed in English for consistency.
""")

# Initialize EasyOCR reader with both Hindi and English
@st.cache_resource
def load_reader():
    return easyocr.Reader(['en', 'hi'], gpu=True)

reader = load_reader()

# Function to replace Hindi numerals with English numerals
def convert_hindi_numerals_to_english(text):
    hindi_numerals = '०१२३४५६७८९'
    english_numerals = '0123456789'
    
    # Replace Hindi numerals with English numerals
    translation_table = str.maketrans(''.join(hindi_numerals), ''.join(english_numerals))
    return text.translate(translation_table)

# Improved Layout detection function with adaptive kernel
def detect_layout(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)[1]
    
    # Check for potential table layout (presence of grid-like lines)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (image.shape[1] // 2, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, image.shape[0] // 2))
    
    horizontal_lines = cv2.dilate(cv2.erode(thresh, horizontal_kernel, iterations=2), horizontal_kernel, iterations=2)
    vertical_lines = cv2.dilate(cv2.erode(thresh, vertical_kernel, iterations=2), vertical_kernel, iterations=2)
    
    table_like_areas = cv2.countNonZero(horizontal_lines & vertical_lines)
    
    # Count vertical column-like structures
    vertical_kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (1, image.shape[1] // 20))
    dilated_vertical = cv2.dilate(thresh, vertical_kernel_small, iterations=3)
    contours_vertical, _ = cv2.findContours(dilated_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    column_count = sum(1 for contour in contours_vertical if cv2.boundingRect(contour)[2] > image.shape[1] * 0.2)
    
    if table_like_areas > 1000:  # Table-like structure
        return 'table'
    elif column_count > 1:  # Multiple columns detected
        return 'multi-column'
    else:  # Default to single-column layout
        return 'single-column'

# Function to handle single-column layout with adaptive green box feedback
def process_single_column(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)[1]

    line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (image.shape[1] // 2, 1))
    dilated_lines = cv2.dilate(thresh, line_kernel, iterations=1)

    contours_lines, _ = cv2.findContours(dilated_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = sorted([cv2.boundingRect(contour) for contour in contours_lines], key=lambda l: l[1])

    extracted_text = ""
    for line in lines:
        x, y, w, h = line
        text_roi = image[y:y+h, x:x+w]
        result = reader.readtext(text_roi)
        line_text = " ".join([res[1] for res in result])
        extracted_text += line_text + "\n"

        # Draw green boxes around detected text
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return extracted_text, image

# Improved function to handle multi-column layout with adaptive green box feedback
def process_multi_column(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)[1]

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, image.shape[1] // 20))
    dilated = cv2.dilate(thresh, vertical_kernel, iterations=3)
    
    contours_columns, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    columns = sorted([cv2.boundingRect(contour) for contour in contours_columns], key=lambda c: c[0])

    extracted_text = ""
    for column in columns:
        x, y, w, h = column
        column_roi = image[y:y+h, x:x+w]
        column_text, _ = process_single_column(column_roi)  # Treat each column as single-column
        extracted_text += column_text + "\n"

        # Draw green boxes around detected columns
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return extracted_text, image

# Improved function to handle table layout with adaptive green box feedback
def process_table(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)[1]

    # Use morphology to detect horizontal and vertical lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (image.shape[1] // 2, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, image.shape[0] // 2))
    
    horizontal_lines = cv2.dilate(cv2.erode(thresh, horizontal_kernel, iterations=2), horizontal_kernel, iterations=2)
    vertical_lines = cv2.dilate(cv2.erode(thresh, vertical_kernel, iterations=2), vertical_kernel, iterations=2)
    
    grid = horizontal_lines & vertical_lines

    contours_cells, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cells = sorted([cv2.boundingRect(contour) for contour in contours_cells], key=lambda c: (c[1], c[0]))

    extracted_text = ""
    for cell in cells:
        x, y, w, h = cell
        cell_roi = image[y:y+h, x:x+w]
        result = reader.readtext(cell_roi)
        cell_text = " ".join([res[1] for res in result])
        extracted_text += cell_text + " | "  # Separate cells with pipe symbols

        # Draw green boxes around detected cells
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return extracted_text, image

# Function to allow users to download extracted text
def get_text_download_link(text):
    """Generates a link to download the extracted text as a .txt file."""
    b64 = base64.b64encode(text.encode()).decode()  # Some strings
    href = f'<a href="data:file/txt;base64,{b64}" download="extracted_text.txt">📥 Download Extracted Text</a>'
    return href

# Initialize session state for search_query and highlighted_text if not present
if 'search_query_input' not in st.session_state:
    st.session_state['search_query_input'] = ''
if 'highlighted_text' not in st.session_state:
    st.session_state['highlighted_text'] = ''
if 'extracted_text' not in st.session_state:
    st.session_state['extracted_text'] = ''

# Main Processing Logic
if uploaded_file is not None:
    # Load image and convert to RGB
    image = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(image)

    # Display uploaded image and processed image side by side
    st.markdown("### 📷 Uploaded and Processed Images")
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption='🖼️ Uploaded Image', use_column_width=True)
    with col2:
        # Detect layout type
        layout_type = detect_layout(image_np)
        st.write(f"**Detected Layout Type:** {layout_type}")

        # Perform OCR based on layout type
        with st.spinner('🔍 Performing OCR...'):
            if layout_type == 'single-column':
                extracted_text, processed_image = process_single_column(image_np.copy())
            elif layout_type == 'multi-column':
                extracted_text, processed_image = process_multi_column(image_np.copy())
            elif layout_type == 'table':
                extracted_text, processed_image = process_table(image_np.copy())
            else:
                extracted_text = "Could not determine layout"
                processed_image = image_np.copy()

            # Convert Hindi numerals to English numerals
            extracted_text = convert_hindi_numerals_to_english(extracted_text)
    
    # Update session state for extracted text and reset highlighted text
    st.session_state['extracted_text'] = extracted_text
    st.session_state['highlighted_text'] = ''

    # Display processed image with green boxes
    st.image(processed_image, caption="🖼️ Processed Image with Detected Text Regions", use_column_width=True)

    # Display extracted text and download option
    st.markdown("### 📝 Extracted Text")
    # Use a div with an id for JavaScript to target
    extracted_text_html = f'<div id="extracted_text" style="white-space: pre-wrap;">{st.session_state["extracted_text"]}</div>'
    st.markdown(extracted_text_html, unsafe_allow_html=True)
    
    # Download button for extracted text
    st.markdown(get_text_download_link(st.session_state["extracted_text"]), unsafe_allow_html=True)

    # Enhanced Search Functionality
    st.markdown("---")
    st.header("🔍 Search Extracted Text")

    # Place search functionality in the main area
    search_query = st.text_input(
        "🔑 Enter keyword(s) to search:",
        help="You can enter multiple keywords separated by commas. Enable regex for advanced searches.",
        key="search_query_input"
    )

    if search_query:
        # Split multiple keywords by comma
        if enable_regex:
            # Treat the entire input as a single regex pattern
            keywords = [search_query.strip()]
        else:
            keywords = [kw.strip() for kw in search_query.split(',') if kw.strip()]
        
        highlighted_text = st.session_state['extracted_text']
        match_counts = {}

        for kw in keywords:
            try:
                if enable_regex:
                    pattern = re.compile(kw, re.IGNORECASE)
                else:
                    # Compile regex for case-insensitive search with word boundaries
                    pattern = re.compile(rf'\b({re.escape(kw)})\b', re.IGNORECASE)
                matches = pattern.findall(highlighted_text)
                match_counts[kw] = len(matches)

                # Replace matches with bold and red color using HTML
                highlighted_text = pattern.sub(lambda m: f"<span style='color:red; font-weight:bold;'>{m.group(1)}</span>", highlighted_text)
            except re.error:
                st.error(f"❌ Invalid regex pattern: `{kw}`")
                match_counts[kw] = 0

        # Update session state with highlighted text
        st.session_state['highlighted_text'] = highlighted_text

        # Display match counts
        st.markdown("### **Search Results:**")
        for kw, count in match_counts.items():
            st.write(f"**{kw}**: {count} match{'es' if count != 1 else ''}")

        # Display highlighted text with scrolling
        st.markdown("### **Highlighted Text:**")
        highlighted_text_html = f'<div style="white-space: pre-wrap;">{st.session_state["highlighted_text"].replace("\n", "<br>")}</div>'
        st.markdown(highlighted_text_html, unsafe_allow_html=True)
    else:
        # If no search query, ensure previous search results are not displayed
        st.session_state['highlighted_text'] = ''
        st.empty()
