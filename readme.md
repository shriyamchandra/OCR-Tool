## Live Application link

https://shriyamchandra-ocr-tool-app-j0apqb.streamlit.app/

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Deployment](#deployment)
- [Live Demo](#live-demo)
- [Examples](#examples)
- [Assumptions](#assumptions)
- [Future Enhancements](#future-enhancements)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Project Overview

This project is a web-based prototype that demonstrates the ability to perform Optical Character Recognition (OCR) on uploaded images containing text in both **Hindi** and **English**. The application not only extracts text from images but also provides a keyword search functionality to help users quickly find relevant information within the extracted text. The prototype is developed using **Streamlit** and **EasyOCR**, and it is deployed online for easy accessibility.

## Features

- **Image Upload:** Users can upload images in common formats such as JPEG, PNG, BMP, and TIFF.
- **OCR Processing:** Extracts text from uploaded images containing Hindi and English languages.
- **Layout Detection:** Automatically detects the layout of the text (single-column, multi-column, table) to optimize OCR accuracy.
- **Text Highlighting:** Highlights detected text regions within the image for visual confirmation.
- **Keyword Search:** Allows users to search for keywords within the extracted text, supporting both plain text and regex-based searches.
- **Download Extracted Text:** Users can download the extracted text as a `.txt` file for offline use.

## Technologies Used

- **Python 3.8+**
- **Streamlit:** For building the web application interface.
- **EasyOCR:** For performing Optical Character Recognition.
- **OpenCV:** For image processing and layout detection.
- **Pillow:** For image handling.
- **NumPy:** For numerical operations.
- **Base64 & IO:** For encoding and handling download functionalities.

## Installation

Follow these steps to set up the project locally:

1. **Clone the Repository:**

   ```bash
   https://github.com/shriyamchandra/OCR-Tool.git
   cd ocr-document-search-app
