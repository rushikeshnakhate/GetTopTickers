from datetime import datetime
from typing import List, Tuple, Union

from src.date_generator.rebalancing import Rebalancing
from src.date_generator.rebalancing_period import RebalancingPeriod


class DateRangeGenerator:
    def __init__(self, years: Union[int, List[int]], months: Union[int, List[int], None] = None,
                 rebalancing: Union[RebalancingPeriod, int] = RebalancingPeriod.MONTHLY):
        self.years = [years] if isinstance(years, int) else years
        self.months = ([months] if isinstance(months, int) else months) if months else list(range(1, 13))
        self.rebalancing = rebalancing

    def get_date_range(self) -> List[Tuple[datetime.date, datetime.date]]:
        date_ranges = []

        for year in self.years:
            for month in self.months:
                date_ranges.extend(Rebalancing.generate_rebalancing_dates(year, month, self.rebalancing))

        return date_ranges
