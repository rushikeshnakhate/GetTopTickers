# Date Range Generator Library for Rebalancing Periods

This Python project provides a tool for generating date ranges based on various rebalancing periods. The rebalancing periods can be daily, half-monthly, monthly, yearly, or custom, allowing flexibility for various financial, trading, or portfolio management scenarios.

## Purpose

The tool helps generate date ranges based on different rebalancing periods, where "rebalancing" refers to the frequency at which portfolio allocations, data, or financial instruments are adjusted or updated. For example:
- **Daily rebalancing**: Updates the portfolio every day.
- **Half-monthly rebalancing**: Divides a month into two periods (1-15 and 16-end of the month).
- **Monthly rebalancing**: Adjusts the portfolio once a month.
- **Yearly rebalancing**: Updates the portfolio once per year.
- **Custom rebalancing**: Allows you to define custom periods (e.g., every 5 days).

## Features

- **Daily Rebalancing**: Generates date ranges for every single day in a month.
- **Half-Monthly Rebalancing**: Generates two periods for the month (1st-15th, 16-end of the month).
- **Monthly Rebalancing**: Generates a single range for the whole month.
- **Yearly Rebalancing**: Generates a range for the entire year.
- **Custom Rebalancing**: Allows the user to define custom rebalancing periods, such as every 5, 10, or 15 days.



### Example 1: Daily Rebalancing for March 2025
To generate daily rebalancing periods for March 2025, use the following code:
```
from datetime import datetime
from your_module import DateRangeGenerator, RebalancingPeriod

generator = DateRangeGenerator(years=2025, months=3, rebalancing=RebalancingPeriod.DAILY)
date_ranges = generator.get_date_range()

for start_date, end_date in date_ranges:
    print(f"Start: {start_date}, End: {end_date}")
Output Example:
Start: 2025-03-01, End: 2025-03-01
Start: 2025-03-02, End: 2025-03-02
Start: 2025-03-03, End: 2025-03-03
Start: 2025-03-04, End: 2025-03-04
Start: 2025-03-05, End: 2025-03-05
```





### Example 2: Half-Monthly Rebalancing for February 2025
To generate half-monthly rebalancing periods for February 2025, where the month is split into two periods (1st-15th, 16-end):
```
generator = DateRangeGenerator(years=2025, months=2, rebalancing=RebalancingPeriod.HALF_MONTHLY)
date_ranges = generator.get_date_range()

for start_date, end_date in date_ranges:
    print(f"Start: {start_date}, End: {end_date}")
Output Example:
Start: 2025-02-01, End: 2025-02-15
Start: 2025-02-16, End: 2025-02-28
```




### Example 3: Monthly Rebalancing for May 2025
To generate monthly rebalancing periods for May 2025:
```
generator = DateRangeGenerator(years=2025, months=5, rebalancing=RebalancingPeriod.MONTHLY)
date_ranges = generator.get_date_range()

for start_date, end_date in date_ranges:
    print(f"Start: {start_date}, End: {end_date}")
Output Example:
Start: 2025-05-01, End: 2025-05-31

```



### Example 4: Custom Rebalancing Period (Every 5 Days) for April 2025
To generate custom rebalancing periods every 5 days for April 2025:
```
generator = DateRangeGenerator(years=2025, months=4, rebalancing=5)
date_ranges = generator.get_date_range()

for start_date, end_date in date_ranges:
    print(f"Start: {start_date}, End: {end_date}")
Output Example:

Start: 2025-04-01, End: 2025-04-05
Start: 2025-04-06, End: 2025-04-10
Start: 2025-04-11, End: 2025-04-15
Start: 2025-04-16, End: 2025-04-20
Start: 2025-04-21, End: 2025-04-25
Start: 2025-04-26, End: 2025-04-30
```