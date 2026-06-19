"""Uvicorn entrypoint: ``scouter`` console script / ``python -m scouter``."""

from __future__ import annotations

import logging

from .api import create_app
from .config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# Importable ASGI app for `uvicorn scouter.server:app`.
app = create_app()


def main() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "scouter.server:app",
        host=settings.host,
        port=settings.port,
        workers=1,  # single worker: the backend owns one model/session
        log_level="info",
    )


if __name__ == "__main__":
    main()
