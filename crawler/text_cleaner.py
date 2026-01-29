import re
from typing import List


class TextCleaner:
    """
    Cleans and normalizes raw website text by:
    - Removing boilerplate patterns (headers, footers, nav-like content)
    - Removing advertisements and cookie notices
    - Deduplicating lines
    - Normalizing whitespace
    """

    def __init__(self):
        # Common noise patterns found across websites
        self.noise_patterns = [
            r"cookie policy",
            r"accept cookies",
            r"privacy policy",
            r"terms of service",
            r"all rights reserved",
            r"subscribe",
            r"sign up",
            r"login",
            r"register",
            r"copyright \d{4}",
        ]

    def _remove_noise_lines(self, lines: List[str]) -> List[str]:
        """
        Removes lines matching common website boilerplate patterns.
        """
        cleaned_lines = []
        for line in lines:
            lowered = line.lower()
            if any(re.search(pattern, lowered) for pattern in self.noise_patterns):
                continue
            if len(line.strip()) < 30:
                continue
            cleaned_lines.append(line)
        return cleaned_lines

    def _deduplicate(self, lines: List[str]) -> List[str]:
        """
        Removes duplicate lines while preserving order.
        """
        seen = set()
        unique_lines = []
        for line in lines:
            normalized = line.strip().lower()
            if normalized not in seen:
                seen.add(normalized)
                unique_lines.append(line)
        return unique_lines

    def clean(self, raw_text: str) -> str:
        """
        Main cleaning pipeline.
        Input: raw extracted text
        Output: cleaned and normalized text
        """
        if not raw_text or not raw_text.strip():
            raise ValueError("Empty text received for cleaning.")

        # Split into logical lines
        lines = re.split(r"[.\n]", raw_text)

        # Normalize whitespace
        lines = [re.sub(r"\s+", " ", line).strip() for line in lines]

        # Remove noise and boilerplate
        lines = self._remove_noise_lines(lines)

        # Deduplicate content
        lines = self._deduplicate(lines)

        cleaned_text = ". ".join(lines)

        if len(cleaned_text) < 200:
            raise ValueError(
                "Cleaned text is too short. Website content may not be meaningful."
            )

        return cleaned_text
