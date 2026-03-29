#!/usr/bin/env python3
"""
Download images from a web page only if OpenCV detects a face.

Usage:
    python face_image_downloader.py "https://example.com/page"
    python face_image_downloader.py "https://example.com/page" --output-dir downloads

Use this only on websites you are allowed to scrape and in ways that comply
with the site's terms and robots policy.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import List, Tuple
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

import cv2
import numpy as np


DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    )
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
CASCADE_PATH = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"


class ImageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.images: List[Tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, str]]) -> None:
        attr_map = dict(attrs)

        if tag.lower() == "a":
            href = attr_map.get("href")
            if href:
                description = " ".join(
                    filter(
                        None,
                        [
                            attr_map.get("title", ""),
                            attr_map.get("class", ""),
                            attr_map.get("aria-label", ""),
                        ],
                    )
                )
                self.images.append((href.strip(), description.lower()))
            return

        if tag.lower() != "img":
            return

        src = (
            attr_map.get("src")
            or attr_map.get("data-src")
            or attr_map.get("data-lazy-src")
            or attr_map.get("data-original")
        )
        if not src:
            return

        description = " ".join(
            filter(
                None,
                [
                    attr_map.get("alt", ""),
                    attr_map.get("title", ""),
                    attr_map.get("class", ""),
                ],
            )
        )
        self.images.append((src.strip(), description.lower()))


def fetch_text(url: str) -> str:
    request = Request(url, headers=DEFAULT_HEADERS)
    with urlopen(request, timeout=20) as response:
        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
            raise ValueError(f"URL does not look like an HTML page: {content_type}")
        return response.read().decode("utf-8", errors="ignore")


def fetch_binary(url: str) -> bytes:
    request = Request(url, headers=DEFAULT_HEADERS)
    with urlopen(request, timeout=30) as response:
        return response.read()


def normalize_image_url(page_url: str, raw_src: str) -> str | None:
    if raw_src.startswith("data:"):
        return None
    if raw_src.startswith("//"):
        return f"{urlparse(page_url).scheme}:{raw_src}"
    return urljoin(page_url, raw_src)


def looks_like_image(url: str) -> bool:
    path = urlparse(url).path.lower()
    if any(path.endswith(ext) for ext in IMAGE_EXTENSIONS):
        return True
    return "images.pexels.com" in urlparse(url).netloc.lower()


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
    original_name = os.path.basename(parsed.path) or f"image_{index}.jpg"
    original_name = re.sub(r"[^A-Za-z0-9._-]", "_", original_name)
    if "." not in original_name:
        original_name += ".jpg"
    return f"{index:03d}_{original_name}"


def collect_image_urls(page_url: str) -> List[str]:
    html = fetch_text(page_url)
    parser = ImageParser()
    parser.feed(html)

    image_urls: List[str] = []
    seen = set()

    for raw_src, _description in parser.images:
        full_url = normalize_image_url(page_url, raw_src)
        if not full_url:
            continue
        if not looks_like_image(full_url):
            continue
        if full_url in seen:
            continue
        seen.add(full_url)
        image_urls.append(full_url)

    return image_urls


def load_detector() -> cv2.CascadeClassifier:
    detector = cv2.CascadeClassifier(str(CASCADE_PATH))
    if detector.empty():
        raise RuntimeError(f"Failed to load Haar cascade: {CASCADE_PATH}")
    return detector


def download_images(page_url: str, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_urls = collect_image_urls(page_url)
    detector = load_detector()

    if not image_urls:
        print("No matching images found.")
        return 0

    downloaded = 0
    for index, image_url in enumerate(image_urls, start=1):
        filename = safe_filename(index, image_url)
        destination = output_dir / filename
        try:
            image_bytes = fetch_binary(image_url)
            if not detect_face(image_bytes, detector):
                print(f"Skipped (no face detected): {image_url}")
                continue
            destination.write_bytes(image_bytes)
            downloaded += 1
            print(f"Downloaded: {destination}")
        except Exception as exc:
            print(f"Failed: {image_url} -> {exc}", file=sys.stderr)

    return downloaded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download images from a web page only when OpenCV detects a face."
    )
    parser.add_argument("url", help="Web page URL to scan for images")
    parser.add_argument(
        "--output-dir",
        default="downloaded_faces",
        help="Directory to save downloaded images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)

    try:
        count = download_images(args.url, output_dir)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Finished. Downloaded {count} image(s) into: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
