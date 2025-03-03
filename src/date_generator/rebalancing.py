from calendar import monthrange
from datetime import datetime
from typing import List, Tuple, Union

from src.date_generator.rebalancing_period import RebalancingPeriod


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
    def generate_custom(year: int, month: int, period_days: int) -> List[Tuple[datetime.date, datetime.date]]:
        days_in_month = monthrange(year, month)[1]
        date_ranges = []
        start_day = 1
        while start_day <= days_in_month:
            end_day = min(start_day + period_days - 1, days_in_month)
            date_ranges.append((datetime(year, month, start_day).date(), datetime(year, month, end_day).date()))
            start_day = end_day + 1
        return date_ranges

    @staticmethod
    def generate_rebalancing_dates(year: int, month: int, rebalancing: Union[RebalancingPeriod, int]) -> List[
        Tuple[datetime.date, datetime.date]]:
        if rebalancing == RebalancingPeriod.DAILY:
            return Rebalancing.generate_daily(year, month)
        elif rebalancing == RebalancingPeriod.HALF_MONTHLY:
            return Rebalancing.generate_half_monthly(year, month)
        elif rebalancing == RebalancingPeriod.MONTHLY:
            return Rebalancing.generate_monthly(year, month)
        elif rebalancing == RebalancingPeriod.YEARLY:
            return Rebalancing.generate_yearly(year)
        elif isinstance(rebalancing, int) and 1 < rebalancing < 12:
            return Rebalancing.generate_custom(year, month, rebalancing)
        return []
