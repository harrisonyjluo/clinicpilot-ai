"""Parsing and evaluation of clinic operating hours as doctor availability."""

from datetime import date, time

from app.agent.availability import parse_operating_hours


def test_weekday_range_is_open_during_business_hours():
    schedule = parse_operating_hours(
        "Mon-Fri: 8:00 AM - 6:00 PM, Sat: 9:00 AM - 1:00 PM, Sun: Closed"
    )

    monday = date(2026, 11, 2)
    assert schedule.covers(monday, time(8, 0)) is True
    assert schedule.covers(monday, time(17, 59)) is True
    assert schedule.covers(monday, time(18, 0)) is False
    assert schedule.covers(monday, time(7, 59)) is False


def test_saturday_window_and_sunday_closure_are_parsed():
    schedule = parse_operating_hours(
        "Mon-Fri: 8:00 AM - 6:00 PM, Sat: 9:00 AM - 1:00 PM, Sun: Closed"
    )

    saturday = date(2026, 11, 7)
    sunday = date(2026, 11, 8)
    assert schedule.covers(saturday, time(9, 0)) is True
    assert schedule.covers(saturday, time(12, 59)) is True
    assert schedule.covers(saturday, time(13, 0)) is False
    assert schedule.windows_for(sunday) == ()


def test_weekend_range_can_be_closed_together():
    schedule = parse_operating_hours(
        "Mon-Fri: 9:00 AM - 5:00 PM, Sat-Sun: Closed"
    )

    assert schedule.windows_for(date(2026, 11, 2)) == (
        (time(9, 0), time(17, 0)),
    )
    assert schedule.windows_for(date(2026, 11, 7)) == ()
    assert schedule.windows_for(date(2026, 11, 8)) == ()


def test_window_containing_returns_the_closing_time_for_a_start_time():
    schedule = parse_operating_hours(
        "Mon-Fri: 9:00 AM - 5:00 PM, Sat-Sun: Closed"
    )

    window = schedule.window_containing(date(2026, 11, 3), time(16, 45))
    assert window == (time(9, 0), time(17, 0))
    assert schedule.window_containing(date(2026, 11, 3), time(7, 0)) is None


def test_24_hour_notation_is_accepted():
    schedule = parse_operating_hours("Mon-Fri: 09:00 - 17:00")

    assert schedule.covers(date(2026, 11, 3), time(9, 0)) is True
    assert schedule.covers(date(2026, 11, 3), time(17, 0)) is False
