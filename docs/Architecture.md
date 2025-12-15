# Architecture Overview

The Manhwa Auto-Scanlator operates as a linear pipeline processing image data from web source to final translated image.

## Pipeline Steps

### 1. Scraping Module (`scrape_images`)
-   **Input**: URL string.
-   **Process**: 
    -   Fetches HTML content using `requests`.
    -   Parses DOM with `BeautifulSoup` to find `<img>` tags.
    -   Filters for large images (e.g., >300x300) to ignore icons/ads.
    -   Downloads images to `temp_scanlate/`.
-   **Output**: List of local file paths.

### 2. Text Detection (`detect_text_regions`)
-   **Input**: Single image path.
-   **Engine**: `EasyOCR` (Pre-trained Korean model).
-   **Process**:
    -   Runs detection to get raw bounding boxes and text.
    -   Filters out low-confidence (<25%) and tiny regions.
    -   **Clustering**: Calls `merge_text_regions` to group nearby text boxes. This is critical for speech bubbles where text is often split into multiple lines/detection boxes.
-   **Output**: List of `Region` objects (box, text, confidence).

### 3. Translation Module (`batch_translate`)
-   **Input**: List of Korean text strings from one page.
-   **Engine**: Google Gemini 2.0 Flash.
-   **Process**:
    -   Constructs a single prompt containing all extraction text for the page.
    -   Prompt includes instructions for "Manhwa style" (natural dialogue, sound effects).
    -   Requests JSON output from LLM.
-   **Output**: List of English strings mapping 1:1 to input.

### 4. Typesetting/Overlay (`draw_translation`)
-   **Input**: Original Image, Bounding Box, Translated Text.
-   **Tool**: `Pillow` (PIL).
-   **Process**:
    -   Draws a white rectangle over the original text box (with padding).
    -   Calculates optimal font size to fit the text within the box.
    -   Wraps text to multiple lines.
    -   Centers text vertically and horizontally.
-   **Output**: Modified image saved to `final_scanlate/`.

## Directory Structure
```
.
├── auto_scanlate.py    # Main orchestration logic
├── run.sh              # Entry point script (handles venv)
├── requirements.txt    # Python dependencies
├── .env                # Secrets (API Key)
├── temp_scanlate/      # Intermediate storage (Raw images)
└── final_scanlate/     # Final output (Translated images)
```
