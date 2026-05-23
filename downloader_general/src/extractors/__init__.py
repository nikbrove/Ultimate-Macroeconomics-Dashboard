"""Concrete downloader implementations for each external data source."""

from .github_download import NewsDownloader
from .world_bank_download import WorldBankDownloader
from .yahoo_download import YahooDownloader

__all__ = ["NewsDownloader", "WorldBankDownloader", "YahooDownloader"]
