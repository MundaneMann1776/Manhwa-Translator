# Contributing

We welcome contributions to improve the accuracy and speed of the Manhwa Auto-Scanlator!

## Getting Started

1.  **Fork** the repository on GitHub.
2.  **Clone** your fork locally.
3.  **Install dependencies**:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

## Development Areas

### improving Text Detection
-   The current clustering logic in `merge_text_regions` is heuristic-based (distance threshold).
-   **Goal**: Make it smarter about bubble shapes or reading order.

### Translation Quality
-   Prompt engineering for Gemini is located in `batch_translate`.
-   **Goal**: Improve handling of Sound Effects (SFX) and onomatopoeia.

### Typesetting
-   Currently, we simply blank out the box with white.
-   **Goal**: Implement "inpainting" or intelligent masking to preserve background art outside the immediate text area, or use more shaped bubbles.

## Pull Request Process

1.  Create a new branch for your feature: `git checkout -b feature/amazing-feature`.
2.  Commit your changes.
3.  Push to the branch.
4.  Open a Pull Request.

## Coding Style
-   Use Python type hints.
-   Keep functions small and focused.
-   Add comments for complex logic (especially image processing math).
