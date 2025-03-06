# Date Range Generator Library for Rebalancing Periods

This Python project provides a tool for generating date ranges based on various rebalancing periods. The rebalancing periods can be daily, half-monthly, monthly, yearly, or custom, allowing flexibility for various financial, trading, or portfolio management scenarios.

## Purpose

The tool helps generate date ranges based on different rebalancing periods, where "rebalancing" refers to the frequency at which portfolio allocations, data, or financial instruments are adjusted or updated. For example:
- **Daily rebalancing**: Updates the portfolio every day.
- **Half-monthly rebalancing**: Divides a month into two periods (1-15 and 16-end of the month).
- **Monthly rebalancing**: Adjusts the portfolio once a month.
- **Custom rebalancing**: Allows you to define custom periods (e.g., every 5 days).

## Features

- **Rebalancing by Days**: Generates date ranges based on a specified number of days.
- **Rebalancing by Months**: Generates date ranges based on a specified number of months.
- **Flexible Priority Handling**: If both rebalancing_days and rebalancing_months are provided, rebalancing_days takes precedence


### Example 1: Daily Rebalancing for year 2025
```
rom datetime import datetime
from your_module import DateRangeGenerator

generator = DateRangeGenerator(years=2025, rebalancing_months=3, rebalancing_days=1)
date_ranges = generator.get_date_range()

for start_date, end_date in date_ranges:
    print(f"Start: {start_date}, End: {end_date}")
    
Start: 2025-03-01, End: 2025-03-01
Start: 2025-03-02, End: 2025-03-02
Start: 2025-03-03, End: 2025-03-03
...
```



### Example 2: Half-Monthly Rebalancing 2025
```
generator = DateRangeGenerator(years=2025, rebalancing_months=2, rebalancing_days=15)
date_ranges = generator.get_date_range()

for start_date, end_date in date_ranges:
    print(f"Start: {start_date}, End: {end_date}")
Start: 2025-02-01, End: 2025-02-15
Start: 2025-02-16, End: 2025-02-28
```

### Example 3: Monthly Rebalancing  2025
```
generator = DateRangeGenerator(years=2025, months=5, rebalancing_months=1)
date_ranges = generator.get_date_range()

for start_date, end_date in date_ranges:
    print(f"Start: {start_date}, End: {end_date}")
Output Example:
Start: 2025-01-01, End: 2025-01-31
Start: 2025-02-01, End: 2025-02-31
```


### Example 4: Custom Rebalancing Period (Every 5 Days) 2025
```
generator = DateRangeGenerator(years=2025, rebalancing_months=4, rebalancing_days=5)
date_ranges = generator.get_date_range()

for start_date, end_date in date_ranges:
    print(f"Start: {start_date}, End: {end_date}")
Output Example:

Start: 2025-01-01, End: 2025-01-05
Start: 2025-01-06, End: 2025-01-10
Start: 2025-01-11, End: 2025-01-15
Start: 2025-01-16, End: 2025-01-20
Start: 2025-01-21, End: 2025-01-25
Start: 2025-01-26, End: 2025-01-30
```