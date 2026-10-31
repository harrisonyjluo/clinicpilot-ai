"""Deterministic evaluation of clinic operating hours.

The seed dataset has no per-doctor schedule. A doctor's bookable hours are
therefore bounded by the operating hours of the clinic where they practice.
Parsing is catalog-data handling, not natural-language understanding.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from functools import lru_cache

_DAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_DAY_INDEX = {
    name.casefold(): index for index, name in enumerate(_DAY_NAMES)
}


@dataclass(frozen=True)
class WeeklySchedule:
    """Open windows per weekday (0 = Monday)."""

    windows_by_weekday: dict[int, tuple[tuple[time, time], ...]]

    def windows_for(self, day: date) -> tuple[tuple[time, time], ...]:
        return self.windows_by_weekday.get(day.weekday(), ())

    def window_containing(
        self,
        day: date,
        at: time,
    ) -> tuple[time, time] | None:
        for start, end in self.windows_for(day):
            if start <= at < end:
                return (start, end)
        return None

    def covers(self, day: date, at: time) -> bool:
        return self.window_containing(day, at) is not None


def _parse_day_range(value: str) -> list[int]:
    value = value.strip()
    if "-" in value:
        start_name, end_name = (
            part.strip().casefold() for part in value.split("-", 1)
        )
        start = _DAY_INDEX[start_name]
        end = _DAY_INDEX[end_name]
        if start <= end:
            return list(range(start, end + 1))
        return list(range(start, 7)) + list(range(end + 1))
    return [_DAY_INDEX[value.casefold()]]


def _parse_clock(value: str) -> time:
    value = " ".join(value.split())
    for pattern in ("%I:%M %p", "%I %p", "%H:%M"):
        try:
            return datetime.strptime(value, pattern).time()
        except ValueError:
            continue
    raise ValueError(f"Unrecognized clinic time: {value!r}")


@lru_cache(maxsize=32)
def parse_operating_hours(value: str) -> WeeklySchedule:
    """Parse a clinic `operating_hours` CSV value into open windows."""

    windows: dict[int, list[tuple[time, time]]] = {}
    for segment in value.split(","):
        segment = segment.strip()
        if not segment:
            continue
        try:
            day_part, hours_part = segment.split(":", 1)
        except ValueError as exc:
            raise ValueError(
                f"Missing day/time separator in clinic hours: {segment!r}"
            ) from exc

        days = _parse_day_range(day_part)
        hours = hours_part.strip()
        if hours.casefold() == "closed":
            for day in days:
                windows[day] = []
            continue

        try:
            start_text, end_text = hours.split("-", 1)
        except ValueError as exc:
            raise ValueError(
                f"Unrecognized clinic hours: {segment!r}"
            ) from exc
        window = (_parse_clock(start_text), _parse_clock(end_text))
        for day in days:
            windows.setdefault(day, []).append(window)

    return WeeklySchedule(
        {day: tuple(ranges) for day, ranges in windows.items()}
    )
