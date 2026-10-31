"""Logging configuration for the application."""

import logging


def configure_logging(level: str) -> None:
    """Configure process logging once, using the configured level."""

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
