from __future__ import annotations

import re, logging, asyncio, aiohttp, aiofiles, json
import pandas as pd

from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple
from pathlib import Path
from bs4 import BeautifulSoup

# ---------------------------
# Config / Logger
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("HBRScraper")

# ---------------------------
# Data model
# ---------------------------
@dataclass
class Listing:
    id: Optional[str] = None
    detail_url: Optional[str] = None
    title: Optional[str] = None
    location: Optional[str] = None
    timeline_hours: Optional[int] = None
    area_m2: Optional[int] = None
    bedrooms: Optional[int] = None
    bathrooms: Optional[int] = None
    floors: Optional[int] = None
    frontage: Optional[int] = None  # True or False
    price_million_vnd: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------
# Async Fetcher: Handles HTTP
# ---------------------------
class AsyncFetcher:
    def __init__(
        self,
        base_url: str,
        session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
        timeout: int = 20,
        retries: int = 3,
        backoff: float = 1.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.session = session
        self.semaphore = semaphore
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff

    async def fetch_text(self, url: Optional[str] = None) -> Optional[str]:
        final_url = url or self.base_url

        async with self.semaphore:
            for attempt in range(1, self.retries + 1):
                try:
                    async with self.session.get(final_url, timeout=self.timeout) as resp:
                        resp.raise_for_status()
                        return await resp.text()

                except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                    logger.warning(
                        "Fetch failed (%d/%d): %s",
                        attempt,
                        self.retries,
                        final_url,
                    )
                    await asyncio.sleep(self.backoff * attempt)

        logger.error("Giving up on %s", final_url)
        return None

    async def fetch_soup(self, url: Optional[str] = None) -> Optional[BeautifulSoup]:
        text = await self.fetch_text(url)
        if not text:
            return None
        return BeautifulSoup(text, "lxml")

# ---------------------------
# Checkpoint system (ADD)
# ---------------------------
class Checkpoint:
    def __init__(self, path: Path):
        self.path = path
        self.seen_ids = set()
        self._lock = asyncio.Lock()

    async def load(self):
        if not self.path.exists():
            return
        async with aiofiles.open(self.path, "r", encoding="utf-8") as f:
            async for line in f:
                try:
                    data = json.loads(line)
                    if data.get("id"):
                        self.seen_ids.add(data["id"])
                except json.JSONDecodeError:
                    continue

        logger.info("Loaded %d listings from checkpoint", len(self.seen_ids))

    async def append(self, listing: Listing):
        async with self._lock:
            async with aiofiles.open(self.path, "a", encoding="utf-8") as f:
                await f.write(json.dumps(listing.to_dict(), ensure_ascii=False) + "\n")

# ---------------------------
# Parser: Pure functions to parse nodes
# ---------------------------
class Parser:

    TIME_MULTIPLIERS = {
        "giờ": 1,
        "ngày": 24,
        "tuần": 168,
        "tháng": 720,
        "năm": 8760,
    }

    @staticmethod
    def get_total_pages(soup: BeautifulSoup) -> Optional[int]:
        if soup is None:
            return None
        pagination = soup.find("ul", class_="pagination")
        if not pagination:
            logger.info("Pagination not found.")
            return None
        links = pagination.find_all("a")
        page_numbers = []
        for link in links:
            text = link.get_text(strip=True)
            if text.isdigit():
                page_numbers.append(int(text))
        if page_numbers:
            return max(page_numbers)
        logger.info("No valid page numbers found in pagination.")
        return None

    @staticmethod
    def parse_id(node: BeautifulSoup) -> Optional[str]:
        heart_div = node.find("div", class_="heart")
        if not heart_div:
            return None
        span = heart_div.find("span", class_="btn-save js-btn-save add-like")
        if not span:
            return None
        return span.get("post_id")

    @staticmethod
    def parse_title(node: BeautifulSoup) -> Optional[str]:
        tag = node.find("h3", class_="title")
        if not tag:
            return None
        for icon in tag.find_all("i"):
            icon.decompose()
        return tag.get_text(strip=True) or None

    @staticmethod
    def parse_location(node: BeautifulSoup) -> Optional[str]:
        tag = node.find("div", class_="description")
        if not tag:
            return None
        raw_text = tag.get_text(separator=" ", strip=True)
        location = re.sub(r"\s+,", ",", raw_text)
        return location or None

    @staticmethod
    def parse_timeline_hours(node: BeautifulSoup) -> Optional[int]:
        tag = node.find("div", class_="time")
        if not tag:
            return None

        timetext = tag.get_text(strip=True).lower()

        m = re.search(r"(\d+)\s*(giờ|ngày|tuần|tháng|năm)", timetext)
        if m:
            value = int(m.group(1))
            unit = m.group(2)
            return value * Parser.TIME_MULTIPLIERS[unit]

        logger.debug("Unrecognized time format: %s", timetext)
        return None

    @staticmethod
    def _get_description_items(node: BeautifulSoup) -> List[str]:
        blocks = node.find_all("div", class_="description-item")
        return [b.get_text(strip=True) for b in blocks if b]

    @staticmethod
    def parse_area(node: BeautifulSoup) -> Optional[int]:
        items = Parser._get_description_items(node)
        area_regex = r"(\d+)\s*m²"
        for text in items:
            m = re.search(area_regex, text)
            if m:
                return int(m.group(1))
        return None

    @staticmethod
    def parse_bedrooms(node: BeautifulSoup) -> Optional[int]:
        items = Parser._get_description_items(node)
        bed_regex = r"(\d+)\s*Phòng\s*ngủ"
        for text in items:
            m = re.search(bed_regex, text)
            if m:
                return int(m.group(1))
        return None

    @staticmethod
    def parse_bathrooms(node: BeautifulSoup) -> Optional[int]:
        items = Parser._get_description_items(node)
        bath_regex = r"(\d+)\s*WC"
        for text in items:
            m = re.search(bath_regex, text)
            if m:
                return int(m.group(1))
        return None

    @staticmethod
    def parse_floor(node: BeautifulSoup) -> Optional[int]:
        title_tag = node.find("h3", class_="title")
        if not title_tag:
            return None
        title_text = title_tag.get_text(strip=True)

        # Case 1: "X tầng" or "X lầu" includes ground floor
        direct_floor_regex = r"(\d+)\s*(?:tầng|lầu)"
        m = re.search(direct_floor_regex, title_text, flags=re.IGNORECASE)
        if m:
            return int(m.group(1)) + 1

        # Case 2: "X lầu Y trệt"
        combo_floor_regex = r"(\d+)\s*lầu\s*(\d+)\s*trệt"
        m = re.search(combo_floor_regex, title_text, flags=re.IGNORECASE)
        if m:
            return int(m.group(1)) + int(m.group(2))

        # Case 3: "tầng X" or "lầu X"
        reversed_floor_regex = r"(?:tầng|lầu)\s*(\d+)"
        m = re.search(reversed_floor_regex, title_text, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))

        # Case 4: "X lầu Y lửng" ignore mezzanine
        mezzanine_regex = r"(\d+)\s*lầu\s*(\d+)\s*lửng"
        m = re.search(mezzanine_regex, title_text, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))

        return None

    @staticmethod
    def parse_frontage(node: BeautifulSoup) -> bool:
        title_tag = node.find("h3", class_="title")
        if not title_tag:
            return False
        title_text = title_tag.get_text(strip=True)
        if re.search(r"mặt\s+(tiền|phố|đường)", title_text, flags=re.IGNORECASE):
            return True
        #If not found, return false as usual
        return False

    @staticmethod
    def parse_price(node: BeautifulSoup) -> Optional[float]:
        price_struct = node.find("div", class_="price")
        if not price_struct:
            return None

        price_text = price_struct.get_text(strip=True).lower()

        # Patterns
        mil_thou_regex = r"(\d+)\s*triệu\s*(\d+)\s*(nghìn|ngàn)"
        bil_mil_regex = r"(\d+)\s*tỷ\s*(\d+)\s*triệu"
        bil_half_regex = r"(\d+)\s*tỷ\s*rưỡi"
        mil_half_regex = r"(\d+)\s*triệu\s*rưỡi"
        bil_only_regex = r"(\d+)\s*tỷ"
        mil_only_regex = r"(\d+)\s*triệu"
        fallback_number = r"(\d+)"

        # Match patterns in order of specificity
        m = re.match(mil_thou_regex, price_text)
        if m:
            return round(int(m.group(1)) + int(m.group(2)) / 1000.0, 3)

        m = re.match(bil_mil_regex, price_text)
        if m:
            return int(m.group(1)) * 1000.0 + int(m.group(2))

        m = re.match(bil_half_regex, price_text)
        if m:
            return int(m.group(1)) * 1000.0 + 500.0  # Half a billion VND

        m = re.match(mil_half_regex, price_text)
        if m:
            return int(m.group(1)) + 0.5  # Half a million

        m = re.match(bil_only_regex, price_text)
        if m:
            return int(m.group(1)) * 1000.0

        m = re.match(mil_only_regex, price_text)
        if m:
            return float(m.group(1))

        m = re.match(fallback_number, price_text)
        if m:
            return float(m.group(1))

        logger.debug("Unrecognized price format: %s", price_text)
        return None

    @staticmethod
    def parse_listing(block: BeautifulSoup) -> Listing:
        """
        Given a block (usually the <a class='card-cm'> node), return a Listing dataclass.
        """
        listing = Listing()
        listing.id = Parser.parse_id(block)
        listing.detail_url = block["href"]
        listing.title = Parser.parse_title(block)
        listing.location = Parser.parse_location(block)
        listing.timeline_hours = Parser.parse_timeline_hours(block)
        listing.area_m2 = Parser.parse_area(block)
        listing.bedrooms = Parser.parse_bedrooms(block)
        listing.bathrooms = Parser.parse_bathrooms(block)
        listing.floors = Parser.parse_floor(block)
        listing.frontage = Parser.parse_frontage(block)
        listing.price_million_vnd = Parser.parse_price(block)
        return listing

class DetailParser:
    FLOOR_REGEXES = [
        r"\b(\d+)\s*(?:tầng|lầu)\b",
        r"\b(\d+)\s*[t]\b",           
        r"\b(\d+)\s*[l]\b",          
    ]

    @staticmethod
    def parse_floors_from_description(soup: BeautifulSoup) -> Optional[int]:
        content = soup.select_one("div.col-md-12.text")
        if not content:
            return None

        text = content.get_text(" ", strip=True).lower()

        for pattern in DetailParser.FLOOR_REGEXES:
            m = re.search(pattern, text)
            if m:
                return int(m.group(1))

        return None

# ---------------------------
# High-level Scraper (Memory-safe version)
# ---------------------------
class AsyncHBRScraper:
    def __init__(
        self,
        base_url: str,
        max_connections: int = 12,
        checkpoint_path: Path = Path("checkpoint.jsonl"),
    ):
        self.base_url = base_url.rstrip("/")
        self.semaphore = asyncio.Semaphore(max_connections)
        self.checkpoint = Checkpoint(checkpoint_path)
        self.total_pages: int = 0

    async def enrich_listing_with_detail(fetcher: AsyncFetcher, listing: Listing):
        if listing.floors is not None or not listing.detail_url:
            return listing

        soup = await fetcher.fetch_soup(listing.detail_url)
        if soup:
            listing.floors = DetailParser.parse_floors_from_description(soup)
            del soup  # free memory
        return listing

    async def scrape(self, max_pages: Optional[int] = None):
        headers = {"User-Agent": "Mozilla/5.0 (HBRScraper/async)"}
        connector = aiohttp.TCPConnector(limit=30)

        await self.checkpoint.load()

        async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
            fetcher = AsyncFetcher(self.base_url, session, self.semaphore)

            # Determine total pages
            soup = await fetcher.fetch_soup()
            if not soup:
                logger.error("Failed to fetch base page")
                return

            total_pages = Parser.get_total_pages(soup) or 1
            if max_pages:
                total_pages = min(total_pages, max_pages)
            self.total_pages = total_pages
            logger.info("Total pages: %d", total_pages)
            del soup

            # --- Queue for writing listings safely ---
            write_queue: asyncio.Queue = asyncio.Queue()

            async def writer():
                async with aiofiles.open(self.checkpoint.path, "a", encoding="utf-8") as f:
                    while True:
                        listing = await write_queue.get()
                        if listing is None:  # sentinel to stop
                            break
                        await f.write(json.dumps(listing.to_dict(), ensure_ascii=False) + "\n")
                        self.checkpoint.seen_ids.add(listing.id)
                        write_queue.task_done()

            writer_task = asyncio.create_task(writer())

            # Schedule page processing concurrently
            page_tasks = [
                self.process_page(fetcher, page, write_queue)
                for page in range(1, total_pages + 1)
            ]
            await asyncio.gather(*page_tasks)

            # Stop writer
            await write_queue.put(None)
            await writer_task

    async def process_page(self, fetcher: AsyncFetcher, page: int, write_queue: asyncio.Queue):
        logger.info("Processing page %d / %d", page, self.total_pages)

        page_url = f"{self.base_url}/p{page}"
        soup = await fetcher.fetch_soup(page_url)
        if not soup:
            logger.warning("Page %d: failed to fetch", page)
            return

        container = soup.find("div", class_="gap-24 d-flex flex-column card-container")
        if not container:
            logger.warning("Page %d: container not found", page)
            del soup
            return

        blocks = container.find_all("a", class_="card-cm", href=True)
        total_on_page = len(blocks)
        logger.info("Page %d: %d listings found", page, total_on_page)

        # Process each listing immediately
        detail_tasks = []
        for block in blocks:
            listing = Parser.parse_listing(block)
            if listing.detail_url and listing.detail_url.startswith("/"):
                listing.detail_url = self.base_url + listing.detail_url

            if listing.id and listing.id in self.checkpoint.seen_ids:
                continue

            # Schedule enrichment if needed
            if listing.floors is None and listing.detail_url:
                detail_tasks.append(AsyncHBRScraper.enrich_listing_with_detail(fetcher, listing))
            else:
                # Wrap in completed future
                detail_tasks.append(asyncio.sleep(0, result=listing))

        # Process detail tasks as they complete (memory-efficient)
        for task in asyncio.as_completed(detail_tasks):
            listing = await task
            if listing and listing.id:
                await write_queue.put(listing)
            del listing  # free memory immediately

        # Cleanup
        del blocks
        del container
        del soup
        import gc
        gc.collect()

        logger.info("Page %d done → total: %d", page, total_on_page)


# ---------------------------
# Implementation
# ---------------------------
if __name__ == "__main__":
    target_url = "https://batdongsan.vn/cho-thue-nha"

    async def main():
        scraper = AsyncHBRScraper(
            target_url,
            max_connections=12,
            checkpoint_path=Path("checkpoint.jsonl"),
        )

        await scraper.scrape()

        # Optionally read the JSONL to create DataFrame
        rows = []
        async with aiofiles.open("checkpoint.jsonl", "r", encoding="utf-8") as f:
            async for line in f:
                rows.append(json.loads(line))
        df = pd.DataFrame(rows)
        df.to_csv("house_buying_async.csv", index=False, encoding="utf-8-sig")

    asyncio.run(main())


