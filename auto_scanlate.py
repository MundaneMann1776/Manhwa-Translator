#!/usr/bin/env python3
"""
Manhwa Auto-Scanlator v5 - EasyOCR Edition
- Uses EasyOCR for reliable text detection (like EasyScanlate)
- Uses Gemini for translation only
- Professional quality output
"""
import os
import sys

# Auto-bootstrap venv
VENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv")
if os.path.exists(VENV_PATH) and sys.prefix != VENV_PATH:
    venv_python = os.path.join(VENV_PATH, "bin", "python")
    if os.path.exists(venv_python):
        print(f"[*] Switching to virtual environment...")
        os.execv(venv_python, [venv_python] + sys.argv)

import json
import shutil
import argparse
import time
from typing import List, Dict, Tuple, Any
import requests
from bs4 import BeautifulSoup
import easyocr
import google.generativeai as genai
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv
import numpy as np

# --- Configuration ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = "gemini-2.0-flash"
TEMP_DIR = "temp_scanlate"
OUTPUT_DIR = "final_scanlate"

SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# Global EasyOCR reader (initialized once)
OCR_READER = None

def get_ocr_reader():
    """Initialize EasyOCR reader once."""
    global OCR_READER
    if OCR_READER is None:
        print("[*] Initializing EasyOCR (first run downloads models)...")
        OCR_READER = easyocr.Reader(['ko'], gpu=False, verbose=False)
        print("    -> EasyOCR ready!")
    return OCR_READER

def setup_directories():
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def scrape_images(url: str) -> List[str]:
    """Download images from URL."""
    print(f"[*] Scraping images from {url}...")
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")
        
        img_tags = soup.find_all("img")
        candidates = []
        for img in img_tags:
            src = img.get("src") or img.get("data-src") or img.get("data-original")
            if src and src.strip().startswith("http"):
                candidates.append(src.strip())
        
        candidates = list(dict.fromkeys(candidates))

        saved_paths = []
        for i, src in enumerate(candidates):
            try:
                ext = "jpg"
                if ".png" in src.lower(): ext = "png"
                elif ".webp" in src.lower(): ext = "webp"
                
                filename = f"image_{i:03d}.{ext}"
                local_path = os.path.join(TEMP_DIR, filename)
                
                r = requests.get(src, headers=headers, stream=True, timeout=10)
                if r.status_code == 200:
                    with open(local_path, 'wb') as f:
                        for chunk in r.iter_content(1024):
                            f.write(chunk)
                    
                    with Image.open(local_path) as im:
                        w, h = im.size
                        if w > 300 and h > 300:
                            saved_paths.append(local_path)
                        else:
                            os.remove(local_path)
            except:
                pass
                
        print(f"    Saved {len(saved_paths)} comic pages.")
        return sorted(saved_paths)

    except Exception as e:
        print(f"Scrape failed: {e}")
        return []

def boxes_overlap_or_close(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int], threshold: int = 30) -> bool:
    """Check if two boxes overlap or are very close to each other."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Expand boxes by threshold
    left1, right1 = x1 - threshold, x1 + w1 + threshold
    top1, bottom1 = y1 - threshold, y1 + h1 + threshold
    left2, right2 = x2 - threshold, x2 + w2 + threshold
    top2, bottom2 = y2 - threshold, y2 + h2 + threshold
    
    # Check overlap
    return not (right1 < left2 or right2 < left1 or bottom1 < top2 or bottom2 < top1)

def merge_boxes(boxes: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
    """Merge multiple boxes into one bounding box."""
    if not boxes:
        return (0, 0, 0, 0)
    
    min_x = min(b[0] for b in boxes)
    min_y = min(b[1] for b in boxes)
    max_x = max(b[0] + b[2] for b in boxes)
    max_y = max(b[1] + b[3] for b in boxes)
    
    return (min_x, min_y, max_x - min_x, max_y - min_y)

def merge_text_regions(regions: List[Dict[str, Any]], distance_threshold: int = 40) -> List[Dict[str, Any]]:
    """
    Merge text regions that are close together (likely same speech bubble).
    This is similar to EasyScanlate's group_and_merge_text function.
    """
    if not regions:
        return []
    
    # Sort by y-position (top to bottom reading order)
    regions = sorted(regions, key=lambda r: (r['box'][1], r['box'][0]))
    
    merged = []
    used = set()
    
    for i, region in enumerate(regions):
        if i in used:
            continue
        
        # Find all regions that should be merged with this one
        group = [region]
        group_indices = {i}
        
        # Keep expanding the group until no more matches
        changed = True
        while changed:
            changed = False
            current_merged_box = merge_boxes([r['box'] for r in group])
            
            for j, other in enumerate(regions):
                if j in group_indices:
                    continue
                
                if boxes_overlap_or_close(current_merged_box, other['box'], distance_threshold):
                    group.append(other)
                    group_indices.add(j)
                    changed = True
        
        # Mark all as used
        used.update(group_indices)
        
        # Create merged region
        merged_box = merge_boxes([r['box'] for r in group])
        
        # Sort texts by position (top-to-bottom, left-to-right) for proper reading order
        group_sorted = sorted(group, key=lambda r: (r['box'][1], r['box'][0]))
        merged_text = ' '.join(r['text'] for r in group_sorted)
        avg_confidence = sum(r['confidence'] for r in group) / len(group)
        
        merged.append({
            'box': merged_box,
            'text': merged_text,
            'confidence': avg_confidence
        })
    
    return merged

def detect_text_regions(image_path: str) -> List[Dict[str, Any]]:
    """
    Use EasyOCR to detect text regions with bounding boxes.
    Merges nearby regions that belong to the same speech bubble.
    """
    reader = get_ocr_reader()
    
    # Read image
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # Run OCR
    results = reader.readtext(img_array)
    
    # Convert to our format - catch as much text as possible
    raw_regions = []
    for (bbox, text, confidence) in results:
        # Very low threshold to catch ALL text (0.25)
        if confidence < 0.25:
            continue
        
        # Keep any text with content
        if len(text.strip()) == 0:
            continue
        
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x = int(min(xs))
        y = int(min(ys))
        w = int(max(xs) - x)
        h = int(max(ys) - y)
        
        # Only skip extremely tiny boxes (likely noise artifacts)
        if w < 8 or h < 8:
            continue
        
        raw_regions.append({
            'box': (x, y, w, h),
            'text': text,
            'confidence': confidence
        })
    
    # Merge nearby regions into speech bubbles
    merged_regions = merge_text_regions(raw_regions, distance_threshold=50)
    
    return merged_regions

def batch_translate(model, texts: List[str]) -> List[str]:
    """
    Translate multiple Korean texts to English in one API call.
    More efficient than translating one at a time.
    """
    if not texts:
        return []
    
    # Expert-level translation prompt for human-quality output
    prompt = """You are an expert Korean-to-English translator specializing in Manhwa (Korean comics).
Your translations should read as if written by a native English speaker - natural, flowing, and emotionally resonant.

TRANSLATION RULES:
1. NATURAL DIALOGUE: Translate for how people actually speak, not literal word-for-word.
   - 뭐야 이게 → "What the hell is this?" (not "What is this thing?")
   - 진짜? → "Seriously?" or "For real?" (not "Really?")
   - 어떻게 이런 일이 → "How could this happen?" (natural flow)

2. EMOTIONAL INTENSITY: Match the emotion and intensity of the original.
   - Angry dialogue = use strong words, exclamations
   - Romantic dialogue = tender, intimate phrasing
   - Comedy = preserve the humor and timing

3. CONTEXT AWARENESS: These are sequential speech bubbles from the same scene.
   Use context from other bubbles to improve accuracy.

4. CONCISE: Keep translations brief to fit speech bubbles. Remove filler words.

5. OCR ERRORS: If Korean text looks garbled/incomplete, infer the likely meaning.

6. SOUND EFFECTS: For onomatopoeia (쿵, 끄덕, 철컥, 쾅, 휙, 탁, 쾅쾅, 펑, etc.), return "[SFX]"

7. UNTRANSLATABLE: For fragments or completely unclear text, return "[SKIP]"

Here are the Korean dialogue bubbles to translate:
"""
    for i, text in enumerate(texts):
        prompt += f"{i+1}. \"{text}\"\n"
    
    prompt += """\nRespond with ONLY a JSON array of English translations in the exact same order.
Example format: ["Translation 1", "Translation 2", "[SFX]", "Translation 4"]

JSON array:"""
    
    try:
        response = model.generate_content(prompt, safety_settings=SAFETY_SETTINGS)
        result = response.text.strip()
        
        # Clean up
        if result.startswith("```"):
            result = result.split("\n", 1)[1] if "\n" in result else result[3:]
        if result.startswith("json"):
            result = result[4:]
        if result.endswith("```"):
            result = result[:-3]
        
        translations = json.loads(result.strip())
        
        if len(translations) == len(texts):
            return translations
        else:
            # Fallback: return originals
            return texts
            
    except Exception as e:
        print(f"    [!] Translation error: {e}")
        return texts

def get_font(size=24):
    """Get a reliable system font."""
    font_paths = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc", 
        "/Library/Fonts/Arial.ttf",
    ]
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except:
            continue
    return ImageFont.load_default()

def draw_translation(draw: ImageDraw.Draw, box: Tuple[int, int, int, int], text: str):
    """Draw translated text over the original Korean text."""
    x, y, w, h = box
    
    # Skip SFX, SKIP tags, or empty text
    if "[SFX]" in text or "[SKIP]" in text or text.strip() == "":
        return
    
    # Draw white background with padding for cleaner coverage
    padding = 5
    draw.rectangle([x-padding, y-padding, x+w+padding, y+h+padding], fill="white")
    
    # Calculate font size to fit
    font_size = min(28, max(10, int(h * 0.7)))
    font = get_font(font_size)
    
    # Word wrap
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        test = ' '.join(current_line + [word])
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] <= w - 6:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
    if current_line:
        lines.append(' '.join(current_line))
    
    # Shrink font if too many lines
    while len(lines) * font_size * 1.1 > h and font_size > 8:
        font_size -= 2
        font = get_font(font_size)
        lines = []
        current_line = []
        for word in words:
            test = ' '.join(current_line + [word])
            bbox = draw.textbbox((0, 0), test, font=font)
            if bbox[2] - bbox[0] <= w - 6:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        if current_line:
            lines.append(' '.join(current_line))
    
    # Draw centered text
    total_h = len(lines) * font_size * 1.1
    current_y = y + (h - total_h) / 2
    
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_w = bbox[2] - bbox[0]
        line_x = x + (w - line_w) / 2
        draw.text((line_x, current_y), line, fill="black", font=font)
        current_y += font_size * 1.1

def process_image(model, image_path: str) -> str:
    """Process a single image: detect text, translate, overlay."""
    print(f"[*] Processing {os.path.basename(image_path)}...")
    
    # Step 1: Detect text with EasyOCR
    regions = detect_text_regions(image_path)
    
    if not regions:
        out_path = os.path.join(OUTPUT_DIR, os.path.basename(image_path))
        shutil.copy(image_path, out_path)
        print(f"    -> No text found, copied original")
        return out_path
    
    print(f"    -> Detected {len(regions)} text regions")
    
    # Step 2: Batch translate all texts
    korean_texts = [r['text'] for r in regions]
    english_texts = batch_translate(model, korean_texts)
    
    # Step 3: Draw translations on image
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    
    translated_count = 0
    for region, translation in zip(regions, english_texts):
        if "[SFX]" not in translation:
            draw_translation(draw, region['box'], translation)
            translated_count += 1
    
    # Save
    out_path = os.path.join(OUTPUT_DIR, os.path.basename(image_path))
    img.save(out_path, quality=95)
    print(f"    -> Translated {translated_count} bubbles, saved: {out_path}")
    return out_path

def serve_results():
    """Start local web server for iPad viewing."""
    import http.server
    import socketserver
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except:
        IP = '127.0.0.1'
    finally:
        s.close()

    PORT = 8000
    os.chdir(OUTPUT_DIR)
    
    files = sorted([f for f in os.listdir('.') if f.lower().endswith(('.jpg', '.png', '.webp'))])
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Manhwa Reader</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {{ background: #1a1a1a; margin: 0; padding: 0; }}
        img {{ width: 100%; display: block; }}
    </style>
</head>
<body>
{''.join(f'<img src="{f}" loading="lazy">' for f in files)}
</body>
</html>"""
    
    with open("index.html", "w") as f:
        f.write(html)

    print(f"\n[+] Reader ready!")
    print(f"    Open on iPad: http://{IP}:{PORT}")
    print(f"    Press Ctrl+C to stop\n")

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), http.server.SimpleHTTPRequestHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[-] Server stopped.")

def main():
    parser = argparse.ArgumentParser(description="Manhwa Auto-Scanlator (EasyOCR Edition)")
    parser.add_argument("url", nargs="?", help="Chapter URL")
    parser.add_argument("--serve", action="store_true", help="Start reader server")
    args = parser.parse_args()

    url = args.url or input("Enter chapter URL: ").strip()
    if not url:
        print("No URL provided.")
        return

    setup_directories()
    
    # Initialize Gemini
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)
    
    # Pre-initialize OCR (downloads models if needed)
    get_ocr_reader()

    # Download images
    images = scrape_images(url)
    if not images:
        print("No images found.")
        return

    # Process each image
    print(f"\n[*] Translating {len(images)} pages...\n")
    
    for img_path in images:
        try:
            process_image(model, img_path)
        except Exception as e:
            print(f"    [!] Error: {e}")
            shutil.copy(img_path, os.path.join(OUTPUT_DIR, os.path.basename(img_path)))

    print(f"\n[DONE] Check '{OUTPUT_DIR}' folder.")
    
    if args.serve:
        serve_results()

if __name__ == "__main__":
    main()
