#!/usr/bin/env python3
"""
Download up to 200 images from Pexels search results, saving only images where
OpenCV detects a face.

Requirements:
    export PEXELS_API_KEY="your_api_key"

Usage:
    python3 face_image_downloader.py "https://www.pexels.com/search/face/"
    python3 face_image_downloader.py "https://www.pexels.com/search/face/" --output-dir pexels_faces
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from urllib.parse import parse_qs, quote_plus, urlparse
from urllib.request import Request, urlopen

import cv2
import numpy as np


API_BASE = "https://api.pexels.com/v1/search"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)
CASCADE_PATH = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
MAX_DOWNLOADS = 200
PER_PAGE = 80


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Pexels search images only when OpenCV detects a face."
    )
    parser.add_argument("url", help="Pexels search URL, for example https://www.pexels.com/search/face/")
    parser.add_argument(
        "--output-dir",
        default="downloaded_faces",
        help="Directory to save downloaded images",
    )
    return parser.parse_args()


def require_api_key() -> str:
    api_key = os.environ.get("PEXELS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing PEXELS_API_KEY. Create a Pexels API key and export it before running."
        )
    return api_key


def extract_query_from_url(url: str) -> str:
    parsed = urlparse(url)
    if "pexels.com" not in parsed.netloc.lower():
        raise ValueError("This script now supports Pexels search URLs only.")

    query_params = parse_qs(parsed.query)
    if "query" in query_params and query_params["query"]:
        return query_params["query"][0].strip()

    match = re.search(r"/search/([^/]+)/?", parsed.path)
    if not match:
        raise ValueError("Could not extract the search term from the Pexels URL.")

    return match.group(1).replace("-", " ").strip()


def fetch_json(url: str, api_key: str) -> dict:
    request = Request(
        url,
        headers={
            "Authorization": api_key,
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "application/json",
        },
    )
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_binary(url: str) -> bytes:
    request = Request(
        url,
        headers={
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            "Referer": "https://www.pexels.com/",
        },
    )
    with urlopen(request, timeout=30) as response:
        return response.read()


def load_detector() -> cv2.CascadeClassifier:
    detector = cv2.CascadeClassifier(str(CASCADE_PATH))
    if detector.empty():
        raise RuntimeError(f"Failed to load Haar cascade: {CASCADE_PATH}")
    return detector


def detect_face(image_bytes: bytes, detector: cv2.CascadeClassifier) -> bool:
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        return False

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
    )
    return len(faces) > 0


def safe_filename(index: int, image_url: str) -> str:
    parsed = urlparse(image_url)
    original_name = Path(parsed.path).name or f"image_{index}.jpg"
    original_name = re.sub(r"[^A-Za-z0-9._-]", "_", original_name)
    if "." not in original_name:
        original_name += ".jpg"
    return f"{index:03d}_{original_name}"


def collect_pexels_image_urls(search_query: str, api_key: str, limit: int) -> list[str]:
    image_urls: list[str] = []
    page = 1

    while len(image_urls) < limit:
        api_url = (
            f"{API_BASE}?query={quote_plus(search_query)}"
            f"&per_page={PER_PAGE}&page={page}"
        )
        payload = fetch_json(api_url, api_key)
        photos = payload.get("photos", [])
        if not photos:
            break

        for photo in photos:
            src = photo.get("src", {})
            image_url = src.get("large2x") or src.get("large") or src.get("original")
            if image_url:
                image_urls.append(image_url)
                if len(image_urls) >= limit:
                    break

        page += 1

    return image_urls


def download_images(search_url: str, output_dir: Path) -> int:
    api_key = require_api_key()
    search_query = extract_query_from_url(search_url)
    image_urls = collect_pexels_image_urls(search_query, api_key, MAX_DOWNLOADS)
    detector = load_detector()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not image_urls:
        print("No images returned by the Pexels API.")
        return 0

    downloaded = 0
    for image_url in image_urls:
        try:
            image_bytes = fetch_binary(image_url)
            if not detect_face(image_bytes, detector):
                print(f"Skipped (no face detected): {image_url}")
                continue

            filename = safe_filename(downloaded + 1, image_url)
            destination = output_dir / filename
            destination.write_bytes(image_bytes)
            downloaded += 1
            print(f"Downloaded: {destination}")

            if downloaded >= MAX_DOWNLOADS:
                print(f"Reached download limit: {MAX_DOWNLOADS}")
                break
        except Exception as exc:
            print(f"Failed: {image_url} -> {exc}", file=sys.stderr)

    return downloaded


def main() -> int:
    args = parse_args()
    try:
        count = download_images(args.url, Path(args.output_dir))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Finished. Downloaded {count} image(s) into: {Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
