import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from typing import Tuple


class WebsiteLoader:
    """
    Responsible for:
    - Validating website URL
    - Fetching HTML content
    - Extracting raw textual data
    """

    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }

    def _is_valid_url(self, url: str) -> bool:
        """
        Checks whether the URL is syntactically valid.
        """
        try:
            parsed = urlparse(url)
            return all([parsed.scheme, parsed.netloc])
        except Exception:
            return False

    def fetch(self, url: str) -> Tuple[str, str]:
        """
        Fetches website content and returns:
        - page_title
        - raw_text (unprocessed)

        Raises descriptive exceptions for failures.
        """

        if not self._is_valid_url(url):
            raise ValueError("Invalid URL format.")

        try:
            response = requests.get(
                url,
                headers=self.headers,
                timeout=self.timeout
            )
        except requests.RequestException as e:
            raise ConnectionError(
                f"Failed to connect to the website: {e}"
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"Website returned status code {response.status_code}"
            )

        soup = BeautifulSoup(response.text, "html.parser")

        # Page title
        page_title = soup.title.string.strip() if soup.title else "Untitled Page"

        # Remove script and style tags early
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        # Extract visible text
        raw_text = soup.get_text(separator=" ")

        # Normalize spaces
        raw_text = " ".join(raw_text.split())

        if not raw_text or len(raw_text) < 50:
            raise ValueError(
                "The website does not contain sufficient textual content."
            )

        return page_title, raw_text
