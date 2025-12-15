# Manhwa Auto-Scanlator v5 - EasyOCR Edition

A powerful, automated tool for scanlating Manhwa (Korean comics) using **EasyOCR** for text detection and **Google Gemini** for context-aware translation.

## Features

-   **Intelligent Text Detection**: Uses EasyOCR to robustly detect speech bubbles and text overlays.
-   **Smart Merging**: Automatically merges nearby text regions that belong to the same speech bubble for coherent translation.
-   **Context-Aware Translation**: Leverages Google's Gemini 2.0 Flash model to translate dialogue with context, emotion, and nuance.
-   **Auto-Cleaning**: Overlays translations on a clean white background, simulating a professional typeset look.
-   **Web Scraper**: Built-in scraper to download individual chapter images from a given URL.
-   **Local Reader**: Includes a built-in web server to read the translated chapter immediately on any device (e.g., iPad) on your local network.

## Prerequisites

-   **Python 3.8+**
-   **Google Gemini API Key**: You need a free API key from [Google AI Studio](https://makersuite.google.com/).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/MundaneMann1776/Manhwa-Translator.git
    cd Manhwa-Translator
    ```

2.  **Set up the environment:**
    The provided `run.sh` script handles virtual environment creation and dependency installation automatically.
    
    Alternatively, manual setup:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Configure API Key:**
    Create a `.env` file in the root directory:
    ```bash
    echo "GOOGLE_API_KEY=your_api_key_here" > .env
    ```

## Usage

### The Easy Way
Run the wrapper script:
```bash
./run.sh [URL]
```

### Manual Usage
Activate the venv and run the python script:
```bash
source .venv/bin/activate
python auto_scanlate.py [URL] --serve
```

### Arguments
-   `url`: (Optional) The URL of the manhwa chapter to scanlate. If omitted, you will be prompted to enter it.
-   `--serve`: (Optional) Starts a local web server after processing to view the results.

## Output
-   **`temp_scanlate/`**: Contains the raw downloaded images.
-   **`final_scanlate/`**: Contains the fully translated and typeset images.

## Example

```bash
./run.sh https://example.com/comics/my-favorite-manhwa/chapter-1 --serve
```

## Troubleshooting
-   **EasyOCR Model Download**: On the very first run, EasyOCR will download its language models. This may take a few minutes.
-   **Permission Denied**: If `./run.sh` fails, run `chmod +x run.sh`.
