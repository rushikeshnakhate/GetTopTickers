from datetime import datetime
from typing import Union, List, Tuple

from src.date_generator.rebalancing import Rebalancing


class DateRangeGenerator:
    def __init__(self, years: Union[int, List[int]], months: Union[int, List[int], None] = None,
                 rebalancing_days: Union[int, None] = None, rebalancing_months: Union[int, None] = 1):
        """
        - If `rebalancing_days` is provided, it takes precedence.
        - If `rebalancing_days` is not provided, `rebalancing_months` will be used.
        - Default rebalancing is **monthly** (i.e., `rebalancing_months=1`).
        """
        self.years = [years] if isinstance(years, int) else years
        self.months = ([months] if isinstance(months, int) else months) if months else list(range(1, 13))
        self.rebalancing_days = rebalancing_days
        self.rebalancing_months = rebalancing_months if rebalancing_days is None else None  # Ensure days take precedence

    def get_date_range(self) -> List[Tuple[datetime.date, datetime.date]]:
        date_ranges = set()  # Using a set to eliminate duplicates

        for year in self.years:
            for month in self.months:
                ranges = Rebalancing.generate_rebalancing_dates(year, month, self.rebalancing_days,
                                                                self.rebalancing_months)
                date_ranges.update(ranges)  # Ensures uniqueness

        return sorted(date_ranges)  # Sorting for consistency
