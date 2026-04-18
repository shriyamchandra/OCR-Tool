# 📝 Enhanced OCR Tool

A web-based OCR application that extracts text from images containing **Hindi** and **English** text, with intelligent layout detection and keyword search.

## Live Demo

🔗 [https://shriyamchandra-ocr-tool-app-j0apqb.streamlit.app/](https://shriyamchandra-ocr-tool-app-j0apqb.streamlit.app/)

## Features

- **Image Upload** — Supports JPEG, PNG, BMP, and TIFF formats
- **Layout Detection** — Automatically detects single-column, multi-column, and table layouts
- **Bilingual OCR** — Extracts Hindi and English text using EasyOCR
- **Visual Feedback** — Draws green bounding boxes around detected text regions
- **Keyword Search** — Plain text and regex search with highlighted results
- **Download** — Export extracted text as a `.txt` file
- **Hindi Numeral Conversion** — Automatically converts Devanagari numerals to English

## Screenshots

| Uploaded Image | Processed Output |
|:-:|:-:|
| ![Upload](Output/Screenshot%202024-09-30%20223743.png) | ![Processed](Output/Screenshot%202024-09-30%20223819.png) |

![Search Results](Output/Screenshot%202024-09-30%20223905.png)

## Tech Stack

| Technology | Purpose |
|---|---|
| **Python 3.8+** | Core language |
| **Streamlit** | Web UI framework |
| **EasyOCR** | Optical Character Recognition engine |
| **OpenCV** | Image processing and layout detection |
| **Pillow** | Image loading and format handling |
| **NumPy** | Array operations |
| **PyTorch** | GPU auto-detection for EasyOCR |

## Project Structure

```
OCR-Tool/
├── app.py                  # Streamlit UI entry point
├── ocr/
│   ├── __init__.py
│   ├── reader.py           # EasyOCR wrapper with GPU auto-detection
│   ├── preprocessing.py    # Shared image preprocessing (Otsu threshold)
│   ├── layout.py           # Layout detection (single/multi-column, table)
│   └── processors.py       # OCR processing for each layout type
├── utils/
│   ├── __init__.py
│   └── text.py             # Numeral conversion, HTML sanitization, search
├── assets/                 # Screenshot assets
├── Output/                 # Sample output screenshots
├── requirements.txt        # Pinned dependencies
├── .gitignore
└── README.md
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/shriyamchandra/OCR-Tool.git
cd OCR-Tool
```

### 2. Create a Virtual Environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

The app will open at [http://localhost:8501](http://localhost:8501).

## Usage

1. **Upload** an image via the sidebar file uploader
2. **View** the original and processed images side by side
3. **Read** the extracted text below the images
4. **Search** for keywords using the search bar (supports regex)
5. **Download** the extracted text as a `.txt` file

## License

This project is open source. See [LICENSE](LICENSE) for details.
