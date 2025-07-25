"""Smart date parsing utilities for flexible date input."""

import re
from datetime import date, datetime


def parse_flexible_date(date_string: str, is_end_date: bool = False) -> datetime:
    """
    Parse flexible date formats and return appropriate datetime.

    Supported formats:
    - YYYY-MM-DD (exact date)
    - YYYY-MM (month - start/end of month based on is_end_date)
    - YYYY (year - start/end of year based on is_end_date)

    Args:
        date_string: Date string to parse
        is_end_date: If True, use end of period (last day of month/year)
                    If False, use start of period (first day of month/year)

    Returns:
        datetime object representing the parsed date

    Raises:
        ValueError: If date format is invalid
    """
    date_string = date_string.strip()

    # YYYY-MM-DD format (exact date)
    if re.match(r"^\d{4}-\d{2}-\d{2}$", date_string):
        try:
            return datetime.strptime(date_string, "%Y-%m-%d")
        except ValueError:
            msg = f"Invalid date: {date_string}"
            raise ValueError(msg) from None

    # YYYY-MM format (month)
    elif re.match(r"^\d{4}-\d{2}$", date_string):
        try:
            year, month = map(int, date_string.split("-"))
            if is_end_date:
                # Last day of the month
                next_month_start = datetime(year + 1, 1, 1) if month == 12 else datetime(year, month + 1, 1)
                # Subtract one day to get last day of current month
                from datetime import timedelta

                return next_month_start - timedelta(days=1)
            # First day of the month
            return datetime(year, month, 1)
        except ValueError:
            msg = f"Invalid year-month: {date_string}"
            raise ValueError(msg) from None

    # YYYY format (year)
    elif re.match(r"^\d{4}$", date_string):
        try:
            year = int(date_string)
            if is_end_date:
                # Last day of the year (December 31)
                return datetime(year, 12, 31)
            # First day of the year (January 1)
            return datetime(year, 1, 1)
        except ValueError:
            msg = f"Invalid year: {date_string}"
            raise ValueError(msg) from None

    else:
        msg = f"Invalid date format: {date_string}. Supported formats: YYYY-MM-DD, YYYY-MM, YYYY"
        raise ValueError(
            msg,
        )


def get_default_end_date() -> str:
    """Get today's date as default end date in YYYY-MM-DD format."""
    return date.today().strftime("%Y-%m-%d")


def validate_date_range(start_date: datetime, end_date: datetime) -> None:
    """
    Validate that start_date is before end_date.

    Args:
        start_date: Start datetime
        end_date: End datetime

    Raises:
        ValueError: If start_date is after end_date
    """
    if start_date > end_date:
        msg = (
            f"Start date ({start_date.strftime('%Y-%m-%d')}) must be before end date ({end_date.strftime('%Y-%m-%d')})"
        )
        raise ValueError(
            msg,
        )


def parse_date_range(
    start_date_str: str,
    end_date_str: str | None = None,
) -> tuple[datetime, datetime]:
    """
    Parse flexible start and end dates with smart defaults.

    Args:
        start_date_str: Start date string (YYYY-MM-DD, YYYY-MM, or YYYY)
        end_date_str: End date string (optional, defaults to today)

    Returns:
        Tuple of (start_datetime, end_datetime)

    Raises:
        ValueError: If dates are invalid or start > end
    """
    # Parse start date (use beginning of period)
    start_date = parse_flexible_date(start_date_str, is_end_date=False)

    # Parse end date (use end of period, or today if not provided)
    if end_date_str is None:
        end_date_str = get_default_end_date()
        end_date = parse_flexible_date(end_date_str, is_end_date=False)  # Today is exact
    else:
        end_date = parse_flexible_date(end_date_str, is_end_date=True)

    # Validate range
    validate_date_range(start_date, end_date)

    return start_date, end_date


def format_date_range_description(start_date: datetime, end_date: datetime) -> str:
    """
    Create a human-readable description of the date range.

    Args:
        start_date: Start datetime
        end_date: End datetime

    Returns:
        Formatted string describing the date range
    """
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    # Check for common patterns
    if start_date.year == end_date.year:
        if start_date.month == end_date.month:
            # Same month
            if start_date.day == 1 and end_date.day == _last_day_of_month(end_date):
                return f"{start_date.strftime('%B %Y')}"
            return f"{start_str} to {end_str}"
        # Same year, different months
        if start_date.month == 1 and start_date.day == 1 and end_date.month == 12 and end_date.day == 31:
            return f"{start_date.year}"
        return f"{start_str} to {end_str}"
    # Different years
    return f"{start_str} to {end_str}"


def _last_day_of_month(dt: datetime) -> int:
    """Get the last day number of the month for a given datetime."""
    next_month = datetime(dt.year + 1, 1, 1) if dt.month == 12 else datetime(dt.year, dt.month + 1, 1)

    from datetime import timedelta

    last_day = next_month - timedelta(days=1)
    return last_day.day


# Examples and test cases
def _test_date_parsing() -> None:
    """Test function to demonstrate date parsing functionality."""
    test_cases = [
        ("2024", None, "2024-01-01 to 2024-12-31"),
        ("2024-07", None, "2024-07-01 to 2024-07-31"),
        ("2024-07-15", None, "2024-07-15 to today"),
        ("2024-01", "2024-06", "2024-01-01 to 2024-06-30"),
        ("2023", "2024", "2023-01-01 to 2024-12-31"),
        ("2024-07-01", "2024-07-31", "2024-07-01 to 2024-07-31"),
    ]

    for start, end, _description in test_cases:
        try:
            start_dt, end_dt = parse_date_range(start, end)
            result = format_date_range_description(start_dt, end_dt)
            print(
                f"Input: start='{start}', end='{end}' -> "
                f"{start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')} ({result})",
            )
        except (ValueError, TypeError) as e:
            print(f"Input: start='{start}', end='{end}' -> ERROR: {e}")


if __name__ == "__main__":
    _test_date_parsing()
