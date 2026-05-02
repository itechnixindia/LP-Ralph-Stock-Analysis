"""
News Fetcher — Retrieves recent headlines for a given stock ticker.

Uses Yahoo Finance RSS (free, no API key) and Google News RSS as fallback.
Results are cached to avoid redundant fetches within the same run.
"""

import hashlib
import json
import logging
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, List
from urllib.request import Request, urlopen
from urllib.error import URLError

logger = logging.getLogger(__name__)

CACHE_DIR = "cache/"
USER_AGENT = "Mozilla/5.0 (compatible; QuantRALPH/1.0)"
MAX_HEADLINES = 50
REQUEST_TIMEOUT = 10


def fetch_headlines(
    ticker: str,
    max_headlines: int = MAX_HEADLINES,
) -> List[Dict]:
    """
    Fetch news headlines for a ticker. Returns list of dicts:
    [{"title": str, "published": str, "source": str}, ...]
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_key = _cache_key(ticker)
    cache_file = os.path.join(CACHE_DIR, f"news_{cache_key}.json")

    # Use cache if less than 6 hours old
    if os.path.exists(cache_file):
        age_hours = (
            datetime.now().timestamp() - os.path.getmtime(cache_file)
        ) / 3600
        if age_hours < 6:
            with open(cache_file, "r") as f:
                return json.load(f)

    headlines = []

    # Source 1: Yahoo Finance RSS
    yahoo_headlines = _fetch_yahoo_rss(ticker)
    headlines.extend(yahoo_headlines)

    # Source 2: Google News RSS (company name search)
    company_name = _ticker_to_company(ticker)
    if company_name:
        google_headlines = _fetch_google_news_rss(company_name)
        headlines.extend(google_headlines)

    # Deduplicate by title similarity
    headlines = _deduplicate(headlines)
    headlines = headlines[:max_headlines]

    # Cache results
    with open(cache_file, "w") as f:
        json.dump(headlines, f, indent=2)

    logger.info(f"Fetched {len(headlines)} headlines for {ticker}")
    return headlines


def _fetch_yahoo_rss(ticker: str) -> List[Dict]:
    """Fetch from Yahoo Finance RSS feed."""
    clean_ticker = ticker.replace(".NS", "").replace(".BO", "")
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    return _parse_rss(url, source="yahoo_finance")


def _fetch_google_news_rss(query: str) -> List[Dict]:
    """Fetch from Google News RSS feed."""
    encoded = query.replace(" ", "+")
    url = f"https://news.google.com/rss/search?q={encoded}+stock&hl=en&gl=US&ceid=US:en"
    return _parse_rss(url, source="google_news")


def _parse_rss(url: str, source: str) -> List[Dict]:
    """Generic RSS parser."""
    try:
        req = Request(url, headers={"User-Agent": USER_AGENT})
        response = urlopen(req, timeout=REQUEST_TIMEOUT)
        xml_data = response.read()
        root = ET.fromstring(xml_data)

        items = root.findall(".//item")
        headlines = []
        for item in items:
            title = item.findtext("title", "").strip()
            pub_date = item.findtext("pubDate", "").strip()
            link = item.findtext("link", "").strip()

            if title:
                headlines.append({
                    "title": _clean_headline(title),
                    "published": pub_date,
                    "source": source,
                    "link": link,
                })

        return headlines

    except (URLError, ET.ParseError, Exception) as e:
        logger.debug(f"RSS fetch failed ({source}): {e}")
        return []


def _clean_headline(title: str) -> str:
    """Remove HTML tags and excessive whitespace."""
    title = re.sub(r"<[^>]+>", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def _deduplicate(headlines: List[Dict]) -> List[Dict]:
    """Remove near-duplicate headlines by normalized title."""
    seen = set()
    unique = []
    for h in headlines:
        normalized = h["title"].lower().strip()
        # Simple dedup: first 50 chars
        key = normalized[:50]
        if key not in seen:
            seen.add(key)
            unique.append(h)
    return unique


def _ticker_to_company(ticker: str) -> str:
    """Map common tickers to company names for search."""
    known = {
        "RELIANCE.NS": "Reliance Industries",
        "TCS.NS": "Tata Consultancy Services",
        "INFY.NS": "Infosys",
        "HDFCBANK.NS": "HDFC Bank",
        "ICICIBANK.NS": "ICICI Bank",
        "WIPRO.NS": "Wipro",
        "BHARTIARTL.NS": "Bharti Airtel",
        "ITC.NS": "ITC Limited",
        "SBIN.NS": "State Bank of India",
        "LT.NS": "Larsen Toubro",
        "AAPL": "Apple",
        "MSFT": "Microsoft",
        "GOOGL": "Alphabet Google",
        "AMZN": "Amazon",
        "NVDA": "NVIDIA",
        "TSLA": "Tesla",
        "META": "Meta Platforms",
        "JPM": "JPMorgan Chase",
    }
    clean = ticker.replace(".BO", ".NS")
    return known.get(clean, ticker.replace(".NS", "").replace(".BO", ""))


def _cache_key(ticker: str) -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    raw = f"{ticker}_{today}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    headlines = fetch_headlines("RELIANCE.NS")
    for i, h in enumerate(headlines[:5], 1):
        print(f"  {i}. [{h['source']}] {h['title']}")
    print(f"\nTotal headlines: {len(headlines)}")
    print("news_fetcher.py: OK")
