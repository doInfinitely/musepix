"""
Sheet music web scraper + GPT-4o bounding-box annotator.

Scrapes real sheet music images from IMSLP, Musescore, and Google Images,
then uses GPT-4o vision to detect and localise musical elements with
bounding boxes.

Usage:
    # Full pipeline (scrape + annotate):
    python scrape_sheet_music.py --sources imslp musescore google \
        --max_per_source 50 --visualise

    # Scrape only (no GPT-4o):
    python scrape_sheet_music.py --skip_annotation --sources imslp

    # Annotate existing images:
    python scrape_sheet_music.py --annotate_only data/real_sheet_music/images
"""

import os
import io
import json
import time
import base64
import argparse
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from urllib import robotparser

import requests
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

ELEMENT_TYPES = [
    "staff_system", "staff_line", "barline",
    "treble_clef", "bass_clef", "alto_clef", "tenor_clef",
    "time_signature", "key_signature",
    "notehead", "stem", "beam", "flag", "ledger_line",
    "whole_rest", "half_rest", "quarter_rest", "eighth_rest", "sixteenth_rest",
    "sharp", "flat", "natural",
    "dynamics", "tie", "slur", "dot", "fermata",
]

# Reuse palette from generate_sheet_music.py:507
PALETTE = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
    '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabebe',
]

USER_AGENT = (
    "MusepixBot/1.0 (research; sheet music OCR dataset collection; "
    "contact: musepix@example.com)"
)

# GPT-4o pricing (per 1k tokens, as of 2025)
GPT4O_INPUT_COST_PER_1K = 0.0025
GPT4O_OUTPUT_COST_PER_1K = 0.01


# ──────────────────────────────────────────────────────────────────────────────
# Base Scraper
# ──────────────────────────────────────────────────────────────────────────────

class SheetMusicScraper(ABC):
    """Base class for sheet music scrapers with rate limiting and retries."""

    def __init__(self, rate_limit: float = 2.0, max_retries: int = 3):
        self.rate_limit = rate_limit
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self._robot_parsers: dict = {}
        self._last_request_time = 0.0

    @property
    @abstractmethod
    def source_name(self) -> str:
        ...

    @abstractmethod
    def scrape(self, max_images: int, **kwargs) -> list[dict]:
        """Return list of {'image': PIL.Image, 'source_url': str, ...}."""
        ...

    def _rate_limit_wait(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_request_time = time.time()

    def _get(self, url: str, **kwargs) -> requests.Response:
        for attempt in range(self.max_retries):
            self._rate_limit_wait()
            try:
                resp = self.session.get(url, timeout=30, **kwargs)
                resp.raise_for_status()
                return resp
            except requests.RequestException as e:
                wait = 2 ** attempt
                log.warning(
                    "%s: request failed (attempt %d/%d): %s — retrying in %ds",
                    self.source_name, attempt + 1, self.max_retries, e, wait,
                )
                time.sleep(wait)
        raise requests.RequestException(
            f"{self.source_name}: all {self.max_retries} attempts failed for {url}"
        )

    def _check_robots(self, url: str) -> bool:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        if base not in self._robot_parsers:
            rp = robotparser.RobotFileParser()
            rp.set_url(f"{base}/robots.txt")
            try:
                rp.read()
            except Exception:
                log.warning("Could not read robots.txt for %s, allowing", base)
                return True
            self._robot_parsers[base] = rp
        return self._robot_parsers[base].can_fetch(USER_AGENT, url)


# ──────────────────────────────────────────────────────────────────────────────
# IMSLP Scraper
# ──────────────────────────────────────────────────────────────────────────────

class IMSLPScraper(SheetMusicScraper):
    """Scrapes sheet music PDFs from IMSLP via the MediaWiki API."""

    API_URL = "https://imslp.org/api.php"
    DEFAULT_COMPOSERS = ["Bach", "Mozart", "Beethoven", "Chopin"]

    def __init__(self, composers: list[str] | None = None,
                 max_pdf_pages: int = 3, **kwargs):
        super().__init__(rate_limit=3.0, **kwargs)
        self.composers = composers or self.DEFAULT_COMPOSERS
        self.max_pdf_pages = max_pdf_pages

    @property
    def source_name(self) -> str:
        return "imslp"

    def scrape(self, max_images: int, **kwargs) -> list[dict]:
        results = []
        per_composer = max(1, max_images // len(self.composers))

        for composer in self.composers:
            if len(results) >= max_images:
                break
            log.info("IMSLP: searching for %s...", composer)
            try:
                pages = self._search_composer(composer, limit=per_composer)
            except Exception as e:
                log.warning("IMSLP: search failed for %s: %s", composer, e)
                continue

            for page_title in pages:
                if len(results) >= max_images:
                    break
                try:
                    pdf_urls = self._get_pdf_urls(page_title)
                except Exception as e:
                    log.warning("IMSLP: could not get PDFs for %s: %s",
                                page_title, e)
                    continue

                for pdf_url in pdf_urls[:1]:  # take first PDF per work
                    if len(results) >= max_images:
                        break
                    try:
                        images = self._download_and_convert_pdf(pdf_url)
                        for img in images:
                            if len(results) >= max_images:
                                break
                            results.append({
                                "image": img,
                                "source_url": pdf_url,
                                "source": self.source_name,
                                "composer": composer,
                                "work": page_title,
                            })
                    except Exception as e:
                        log.warning("IMSLP: PDF conversion failed: %s", e)

        log.info("IMSLP: collected %d images", len(results))
        return results

    def _search_composer(self, composer: str, limit: int = 10) -> list[str]:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": f"Category:Compositions by {composer}",
            "srnamespace": "0",
            "srlimit": str(min(limit * 3, 50)),
            "format": "json",
        }
        if not self._check_robots(self.API_URL):
            log.warning("IMSLP: robots.txt disallows access")
            return []
        resp = self._get(self.API_URL, params=params)
        data = resp.json()
        results = data.get("query", {}).get("search", [])
        return [r["title"] for r in results[:limit]]

    def _get_pdf_urls(self, page_title: str) -> list[str]:
        params = {
            "action": "query",
            "titles": page_title,
            "prop": "images",
            "imlimit": "10",
            "format": "json",
        }
        resp = self._get(self.API_URL, params=params)
        data = resp.json()
        pages = data.get("query", {}).get("pages", {})
        pdf_urls = []
        for page in pages.values():
            for img_info in page.get("images", []):
                name = img_info.get("title", "")
                if name.lower().endswith(".pdf"):
                    file_url = self._get_file_url(name)
                    if file_url:
                        pdf_urls.append(file_url)
        return pdf_urls

    def _get_file_url(self, file_title: str) -> str | None:
        params = {
            "action": "query",
            "titles": file_title,
            "prop": "imageinfo",
            "iiprop": "url",
            "format": "json",
        }
        resp = self._get(self.API_URL, params=params)
        data = resp.json()
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            info_list = page.get("imageinfo", [])
            if info_list:
                return info_list[0].get("url")
        return None

    def _download_and_convert_pdf(self, pdf_url: str) -> list[Image.Image]:
        from pdf2image import convert_from_bytes

        log.info("IMSLP: downloading PDF %s", pdf_url[:80])
        resp = self._get(pdf_url)
        images = convert_from_bytes(
            resp.content,
            first_page=1,
            last_page=self.max_pdf_pages,
            dpi=200,
        )
        return [img.convert("RGB") for img in images]


# ──────────────────────────────────────────────────────────────────────────────
# Musescore Scraper
# ──────────────────────────────────────────────────────────────────────────────

class MusescoreScraper(SheetMusicScraper):
    """Scrapes sheet music preview thumbnails from Musescore."""

    SEARCH_URL = "https://musescore.com/sheetmusic"

    def __init__(self, **kwargs):
        super().__init__(rate_limit=2.0, **kwargs)

    @property
    def source_name(self) -> str:
        return "musescore"

    def scrape(self, max_images: int, **kwargs) -> list[dict]:
        from bs4 import BeautifulSoup

        results = []
        queries = ["piano sonata", "violin concerto", "symphony",
                    "nocturne", "etude", "fugue", "prelude"]

        for query in queries:
            if len(results) >= max_images:
                break

            url = f"{self.SEARCH_URL}?text={query.replace(' ', '+')}"
            if not self._check_robots(url):
                log.warning("Musescore: robots.txt disallows %s", url)
                continue

            log.info("Musescore: searching '%s'...", query)
            try:
                resp = self._get(url)
            except Exception as e:
                log.warning("Musescore: search failed for '%s': %s", query, e)
                continue

            soup = BeautifulSoup(resp.text, "html.parser")

            # Look for preview images in search results
            for img_tag in soup.find_all("img"):
                if len(results) >= max_images:
                    break

                src = img_tag.get("src", "")
                if not src or "score" not in src.lower():
                    continue

                # Skip tiny icons/avatars
                width = img_tag.get("width")
                if width and int(width) < 100:
                    continue

                try:
                    img_resp = self._get(src)
                    img = Image.open(io.BytesIO(img_resp.content)).convert("RGB")
                    if img.width >= 200 and img.height >= 200:
                        results.append({
                            "image": img,
                            "source_url": src,
                            "source": self.source_name,
                        })
                except Exception as e:
                    log.warning("Musescore: image download failed: %s", e)

        log.info("Musescore: collected %d images", len(results))
        return results


# ──────────────────────────────────────────────────────────────────────────────
# Google Image Scraper
# ──────────────────────────────────────────────────────────────────────────────

class GoogleImageScraper(SheetMusicScraper):
    """Uses Google Custom Search JSON API for sheet music images."""

    API_URL = "https://www.googleapis.com/customsearch/v1"

    def __init__(self, **kwargs):
        super().__init__(rate_limit=1.0, **kwargs)
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        self.cse_id = os.environ.get("GOOGLE_CSE_ID")

    @property
    def source_name(self) -> str:
        return "google"

    def scrape(self, max_images: int, **kwargs) -> list[dict]:
        if not self.api_key or not self.cse_id:
            log.warning(
                "Google: GOOGLE_API_KEY or GOOGLE_CSE_ID not set in .env — "
                "skipping Google Image search"
            )
            return []

        results = []
        queries = [
            "sheet music score classical",
            "piano sheet music notation",
            "orchestral score page",
            "violin sheet music",
        ]

        for query in queries:
            if len(results) >= max_images:
                break

            start_index = 1
            while len(results) < max_images and start_index <= 91:
                params = {
                    "key": self.api_key,
                    "cx": self.cse_id,
                    "q": query,
                    "searchType": "image",
                    "imgType": "photo",
                    "imgSize": "large",
                    "num": 10,
                    "start": start_index,
                }
                log.info("Google: searching '%s' (start=%d)...",
                         query, start_index)
                try:
                    resp = self._get(self.API_URL, params=params)
                    data = resp.json()
                except Exception as e:
                    log.warning("Google: API request failed: %s", e)
                    break

                items = data.get("items", [])
                if not items:
                    break

                for item in items:
                    if len(results) >= max_images:
                        break
                    img_url = item.get("link", "")
                    try:
                        img_resp = self._get(img_url)
                        img = Image.open(
                            io.BytesIO(img_resp.content)
                        ).convert("RGB")
                        if img.width >= 200 and img.height >= 200:
                            results.append({
                                "image": img,
                                "source_url": img_url,
                                "source": self.source_name,
                            })
                    except Exception as e:
                        log.warning("Google: image download failed: %s", e)

                start_index += 10

        log.info("Google: collected %d images", len(results))
        return results


# ──────────────────────────────────────────────────────────────────────────────
# GPT-4o Annotator
# ──────────────────────────────────────────────────────────────────────────────

class GPT4oAnnotator:
    """Sends sheet music images to GPT-4o for element detection."""

    SYSTEM_PROMPT = f"""\
You are a music notation expert. You analyse sheet music images and detect
musical elements with precise bounding boxes.

**Coordinate system**: origin at top-left, x increases rightward,
y increases downward. Coordinates are in pixels.

**Output format**: respond with ONLY a JSON array (no markdown fences,
no explanation). Each element:
{{"label": "<type>", "bbox": [x1, y1, x2, y2], "confidence": <0.0-1.0>}}

**Valid labels**: {', '.join(ELEMENT_TYPES)}

**Rules**:
- bbox = [x1, y1, x2, y2] where (x1,y1) is top-left, (x2,y2) is bottom-right
- All coordinates must be within image bounds
- Confidence: 0.0–1.0 (your certainty about this detection)
- Detect ALL visible instances of each element type
- Be thorough: every staff line, every notehead, every accidental
- Overlapping boxes are fine (a notehead on a staff line should have both)
- If you cannot identify any elements, return an empty array: []
"""

    def __init__(self, model: str = "gpt-4o", cost_limit: float = 50.0):
        import openai
        self.client = openai.OpenAI()
        self.model = model
        self.cost_limit = cost_limit
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0

    def annotate(self, img: Image.Image, image_name: str = "") -> list[dict]:
        import openai

        self._check_cost_limit()

        # Encode image to base64
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        user_prompt = (
            f"Image dimensions: {img.width}x{img.height} pixels.\n"
            f"Detect all musical elements and return the JSON array."
        )

        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0.1,
                    max_tokens=4096,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": user_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{b64}",
                                        "detail": "high",
                                    },
                                },
                            ],
                        },
                    ],
                )
                break
            except openai.RateLimitError as e:
                wait = 2 ** (attempt + 1)
                log.warning(
                    "GPT-4o rate limit hit (attempt %d/3) — waiting %ds: %s",
                    attempt + 1, wait, e,
                )
                time.sleep(wait)
        else:
            log.error("GPT-4o: all retry attempts exhausted for %s", image_name)
            return []

        # Track costs
        usage = response.usage
        if usage:
            self.total_input_tokens += usage.prompt_tokens
            self.total_output_tokens += usage.completion_tokens
            self.total_cost = (
                self.total_input_tokens / 1000 * GPT4O_INPUT_COST_PER_1K
                + self.total_output_tokens / 1000 * GPT4O_OUTPUT_COST_PER_1K
            )
            log.info(
                "GPT-4o cost: $%.4f (this call: %d+%d tokens, running: $%.2f)",
                (usage.prompt_tokens / 1000 * GPT4O_INPUT_COST_PER_1K
                 + usage.completion_tokens / 1000 * GPT4O_OUTPUT_COST_PER_1K),
                usage.prompt_tokens, usage.completion_tokens,
                self.total_cost,
            )

        # Parse response
        raw = response.choices[0].message.content.strip()
        elements = self._parse_response(raw, img.width, img.height)
        log.info(
            "GPT-4o: %d elements detected in %s",
            len(elements), image_name,
        )
        return elements

    def _check_cost_limit(self):
        if self.total_cost >= self.cost_limit:
            raise RuntimeError(
                f"GPT-4o cost limit exceeded: ${self.total_cost:.2f} >= "
                f"${self.cost_limit:.2f}. Use --cost_limit to increase."
            )

    def _parse_response(self, raw: str, width: int, height: int) -> list[dict]:
        # Strip markdown fences if present
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last fence lines
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        try:
            elements = json.loads(text)
        except json.JSONDecodeError as e:
            log.warning("GPT-4o: JSON parse failed: %s — raw: %s", e, raw[:200])
            return []

        if not isinstance(elements, list):
            log.warning("GPT-4o: expected list, got %s", type(elements).__name__)
            return []

        validated = []
        for elem in elements:
            if not isinstance(elem, dict):
                continue
            label = elem.get("label", "")
            bbox = elem.get("bbox", [])
            confidence = elem.get("confidence", 0.5)

            if label not in ELEMENT_TYPES:
                continue
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue

            try:
                x1, y1, x2, y2 = [float(v) for v in bbox]
            except (ValueError, TypeError):
                continue

            # Clamp to image bounds
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))

            # Ensure x1 < x2, y1 < y2
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1

            # Discard zero-area boxes
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue

            validated.append({
                "label": label,
                "bbox": [round(x1, 1), round(y1, 1),
                         round(x2, 1), round(y2, 1)],
                "confidence": round(float(confidence), 3),
            })

        return validated


# ──────────────────────────────────────────────────────────────────────────────
# Image preprocessing
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_image(img: Image.Image, max_dimension: int = 2048) -> Image.Image:
    """Resize image so longest side <= max_dimension, convert to RGB."""
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_dimension:
        scale = max_dimension / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    return img


# ──────────────────────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────────────────────

# Map element types to palette colours (cycle)
_LABEL_COLORS = {label: PALETTE[i % len(PALETTE)]
                 for i, label in enumerate(ELEMENT_TYPES)}


def draw_annotations(img: Image.Image, elements: list[dict]) -> Image.Image:
    """Draw labelled bounding boxes with confidence scores."""
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except (IOError, OSError):
        font = ImageFont.load_default()

    for elem in elements:
        label = elem["label"]
        bbox = elem["bbox"]
        conf = elem.get("confidence", 0.0)
        colour = _LABEL_COLORS.get(label, "#ffffff")

        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline=colour, width=2)

        text = f"{label} {conf:.0%}"
        text_bbox = draw.textbbox((x1, y1), text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        # Background for label text
        ty = max(0, y1 - text_h - 2)
        draw.rectangle([x1, ty, x1 + text_w + 4, ty + text_h + 2],
                        fill=colour)
        draw.text((x1 + 2, ty), text, fill="white", font=font)

    return overlay


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_scrape(sources: list[str], max_per_source: int,
               max_pdf_pages: int) -> list[dict]:
    """Run scrapers and return list of image dicts."""
    scrapers: dict[str, SheetMusicScraper] = {
        "imslp": IMSLPScraper(max_pdf_pages=max_pdf_pages),
        "musescore": MusescoreScraper(),
        "google": GoogleImageScraper(),
    }

    all_results = []
    for name in sources:
        scraper = scrapers.get(name)
        if not scraper:
            log.warning("Unknown source: %s — skipping", name)
            continue
        try:
            results = scraper.scrape(max_per_source)
            all_results.extend(results)
        except Exception as e:
            log.error("Scraper %s failed: %s", name, e)

    log.info("Total images scraped: %d", len(all_results))
    return all_results


def run_pipeline(args):
    """Main pipeline: scrape → preprocess → annotate → save."""
    out_dir = args.out_dir
    img_dir = os.path.join(out_dir, "images")
    vis_dir = os.path.join(out_dir, "vis")
    os.makedirs(img_dir, exist_ok=True)
    if args.visualise:
        os.makedirs(vis_dir, exist_ok=True)

    scrape_log = {
        "started": datetime.now().isoformat(),
        "args": vars(args),
        "images": [],
    }

    # ── Step 1: Get images ──
    if args.annotate_only:
        # Load existing images from directory
        image_items = _load_existing_images(args.annotate_only)
        # Copy to output dir if different
        if os.path.abspath(args.annotate_only) != os.path.abspath(img_dir):
            for item in image_items:
                dest = os.path.join(img_dir, item["filename"])
                item["image"].save(dest)
    else:
        scraped = run_scrape(args.sources, args.max_per_source,
                             args.max_pdf_pages)
        image_items = []
        for idx, result in enumerate(scraped):
            img = preprocess_image(result["image"], args.max_dimension)
            filename = f"scraped_{idx:05d}.png"
            filepath = os.path.join(img_dir, filename)
            img.save(filepath)

            item = {
                "filename": filename,
                "image": img,
                "source": result.get("source", "unknown"),
                "source_url": result.get("source_url", ""),
            }
            image_items.append(item)

            scrape_log["images"].append({
                "filename": filename,
                "source": item["source"],
                "source_url": item["source_url"],
                "width": img.width,
                "height": img.height,
            })

        log.info("Saved %d images to %s", len(image_items), img_dir)

    if args.skip_annotation:
        _save_scrape_log(scrape_log, out_dir)
        log.info("Scraping complete (annotation skipped). %d images saved.",
                 len(image_items))
        return

    # ── Step 2: Annotate with GPT-4o ──
    annotator = GPT4oAnnotator(model=args.openai_model,
                               cost_limit=args.cost_limit)
    annotations = []
    checkpoint_path = os.path.join(out_dir, "annotations_checkpoint.json")

    for idx, item in enumerate(image_items):
        img = item["image"]
        filename = item["filename"]

        log.info("Annotating [%d/%d]: %s", idx + 1, len(image_items), filename)

        try:
            elements = annotator.annotate(img, image_name=filename)
        except RuntimeError as e:
            log.error("Stopping annotation: %s", e)
            break

        ann = {
            "image": filename,
            "width": img.width,
            "height": img.height,
            "elements": elements,
            "source": item.get("source", "unknown"),
            "source_url": item.get("source_url", ""),
        }
        annotations.append(ann)

        # Visualization
        if args.visualise and elements:
            vis = draw_annotations(img, elements)
            vis.save(os.path.join(vis_dir, filename))

        # Checkpoint every 25 images
        if (idx + 1) % 25 == 0:
            with open(checkpoint_path, "w") as f:
                json.dump(annotations, f, indent=2)
            log.info("Checkpoint saved (%d annotations)", len(annotations))

    # ── Step 3: Save final outputs ──
    ann_path = os.path.join(out_dir, "annotations.json")
    with open(ann_path, "w") as f:
        json.dump(annotations, f, indent=2)
    log.info("Annotations saved: %s", ann_path)

    scrape_log["completed"] = datetime.now().isoformat()
    scrape_log["total_images"] = len(image_items)
    scrape_log["total_annotated"] = len(annotations)
    scrape_log["gpt4o_cost"] = {
        "input_tokens": annotator.total_input_tokens,
        "output_tokens": annotator.total_output_tokens,
        "total_usd": round(annotator.total_cost, 4),
    }
    _save_scrape_log(scrape_log, out_dir)

    # Clean up checkpoint if final save succeeded
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    log.info(
        "Done. %d images, %d annotated. GPT-4o cost: $%.2f",
        len(image_items), len(annotations), annotator.total_cost,
    )


def _load_existing_images(directory: str) -> list[dict]:
    """Load PNG images from a directory."""
    items = []
    for fname in sorted(os.listdir(directory)):
        if not fname.lower().endswith(".png"):
            continue
        fpath = os.path.join(directory, fname)
        try:
            img = Image.open(fpath).convert("RGB")
            items.append({
                "filename": fname,
                "image": img,
                "source": "existing",
                "source_url": "",
            })
        except Exception as e:
            log.warning("Could not load %s: %s", fpath, e)
    log.info("Loaded %d existing images from %s", len(items), directory)
    return items


def _save_scrape_log(scrape_log: dict, out_dir: str):
    log_path = os.path.join(out_dir, "scrape_log.json")
    with open(log_path, "w") as f:
        json.dump(scrape_log, f, indent=2)
    log.info("Scrape log saved: %s", log_path)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Scrape real sheet music and annotate with GPT-4o.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Full pipeline:
  python scrape_sheet_music.py --sources imslp musescore --max_per_source 50 --visualise

  # Scrape only (no GPT-4o):
  python scrape_sheet_music.py --skip_annotation --sources imslp

  # Annotate existing images:
  python scrape_sheet_music.py --annotate_only data/real_sheet_music/images --cost_limit 1.0
""",
    )

    parser.add_argument(
        "--sources", nargs="+", default=["imslp", "musescore", "google"],
        choices=["imslp", "musescore", "google"],
        help="Sources to scrape from (default: all three).",
    )
    parser.add_argument(
        "--max_per_source", type=int, default=50,
        help="Maximum images per source (default: 50).",
    )
    parser.add_argument(
        "--max_pdf_pages", type=int, default=3,
        help="Max pages to convert per PDF (IMSLP only, default: 3).",
    )
    parser.add_argument(
        "--max_dimension", type=int, default=2048,
        help="Resize images so longest side <= this value (default: 2048).",
    )
    parser.add_argument(
        "--openai_model", type=str, default="gpt-4o",
        help="OpenAI model for annotation (default: gpt-4o).",
    )
    parser.add_argument(
        "--cost_limit", type=float, default=50.0,
        help="Hard cost limit in USD for GPT-4o calls (default: $50).",
    )
    parser.add_argument(
        "--visualise", action="store_true",
        help="Save annotated visualisation images to vis/ subdirectory.",
    )
    parser.add_argument(
        "--out_dir", type=str, default="data/real_sheet_music",
        help="Output directory (default: data/real_sheet_music).",
    )
    parser.add_argument(
        "--skip_annotation", action="store_true",
        help="Scrape images without running GPT-4o annotation.",
    )
    parser.add_argument(
        "--annotate_only", type=str, default=None, metavar="DIR",
        help="Skip scraping; annotate existing images in DIR.",
    )

    args = parser.parse_args()

    if args.annotate_only and args.skip_annotation:
        parser.error("Cannot use both --annotate_only and --skip_annotation.")

    run_pipeline(args)


if __name__ == "__main__":
    main()
