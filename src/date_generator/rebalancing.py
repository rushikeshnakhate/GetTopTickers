from calendar import monthrange
from datetime import datetime
from typing import List, Tuple, Union


class Rebalancing:
    @staticmethod
    def generate_daily(year: int, month: int) -> List[Tuple[datetime.date, datetime.date]]:
        days_in_month = monthrange(year, month)[1]
        return [(datetime(year, month, day).date(), datetime(year, month, day).date()) for day in
                range(1, days_in_month + 1)]

    @staticmethod
    def generate_half_monthly(year: int, month: int) -> List[Tuple[datetime.date, datetime.date]]:
        days_in_month = monthrange(year, month)[1]
        mid_month = 15
        return [(datetime(year, month, 1).date(), datetime(year, month, mid_month).date()),
                (datetime(year, month, mid_month + 1).date(), datetime(year, month, days_in_month).date())]

    @staticmethod
    def generate_monthly(year: int, month: int) -> List[Tuple[datetime.date, datetime.date]]:
        days_in_month = monthrange(year, month)[1]
        return [(datetime(year, month, 1).date(), datetime(year, month, days_in_month).date())]

    @staticmethod
    def generate_yearly(year: int) -> List[Tuple[datetime.date, datetime.date]]:
        return [(datetime(year, 1, 1).date(), datetime(year, 12, 31).date())]

    @staticmethod
    def generate_custom_days(year: int, month: int, period_days: int) -> List[Tuple[datetime.date, datetime.date]]:
        days_in_month = monthrange(year, month)[1]
        date_ranges = []
        start_day = 1
        while start_day <= days_in_month:
            end_day = min(start_day + period_days - 1, days_in_month)
            date_ranges.append((datetime(year, month, start_day).date(), datetime(year, month, end_day).date()))
            start_day = end_day + 1
        return date_ranges

    @staticmethod
    def generate_custom_months(year: int, start_month: int, period_months: int) -> List[
        Tuple[datetime.date, datetime.date]]:
        date_ranges = []

        # Start from the given start_month and iterate in steps of period_months
        month = start_month
        while month <= 12:
            # Calculate the start and end month based on period_months
            end_month = min(month + period_months - 1, 12)  # Ensure we don't go past December

            # Get the last day of the end month
            start_date = datetime(year, month, 1).date()
            end_date = datetime(year, end_month, monthrange(year, end_month)[1]).date()

            date_ranges.append((start_date, end_date))

            # Move to the next period_months interval
            month = end_month + 1

        return date_ranges

    @staticmethod
    def generate_rebalancing_dates(year: int, month: int, rebalancing_days: Union[int, None],
                                   rebalancing_months: Union[int, None]) -> List[Tuple[datetime.date, datetime.date]]:
        if rebalancing_days:
            return Rebalancing.generate_custom_days(year, month, rebalancing_days)
        elif rebalancing_months:
            return Rebalancing.generate_custom_months(year, month, rebalancing_months)
        return Rebalancing.generate_monthly(year, month)
