# Manhwa Auto-Scanlator v5 - EasyOCR Edition

A simple tool for scanlating Manhwa using **EasyOCR** for text detection and **Google Gemini** for translation.

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

Currently on-hold indefinitely.
