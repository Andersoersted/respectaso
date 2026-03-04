"""
Optional in-process scheduler fallback.

Production should use the external cron management command:
  python manage.py refresh_tracked_keywords --trigger cron
"""

from __future__ import annotations

import logging
import threading
import time

from django.conf import settings

from .models import RefreshRun, SearchResult
from .refresh_service import get_pairs_to_refresh, run_refresh

logger = logging.getLogger(__name__)

_scheduler_started = False
_scheduler_lock = threading.Lock()


def get_status():
    """Return latest persisted refresh status payload."""
    return RefreshRun.latest_status_payload()


def _scheduler_loop():
    # Let app boot finish first.
    time.sleep(30)
    while True:
        try:
            if get_pairs_to_refresh(force=False):
                run_refresh(
                    trigger=RefreshRun.TRIGGER_THREAD,
                    source=SearchResult.SOURCE_DAILY_REFRESH,
                    force=False,
                )
        except Exception as exc:  # pragma: no cover - defensive production path
            logger.error("Thread scheduler error: %s", exc)
        # Hourly check cadence.
        time.sleep(3600)


def start_scheduler():
    """Start the thread fallback scheduler when enabled."""
    global _scheduler_started
    if settings.AUTO_REFRESH_MODE != "thread":
        logger.info("Auto-refresh thread scheduler disabled (AUTO_REFRESH_MODE=%s).", settings.AUTO_REFRESH_MODE)
        return
    with _scheduler_lock:
        if _scheduler_started:
            return
        _scheduler_started = True
    thread = threading.Thread(target=_scheduler_loop, daemon=True, name="aso-auto-refresh")
    thread.start()
    logger.info("Auto-refresh thread scheduler started.")
