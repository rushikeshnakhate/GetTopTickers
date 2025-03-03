# Date Range Generator for Rebalancing Periods

This project provides a Python tool designed to generate date ranges for different rebalancing periods. The rebalancing periods can be daily, half-monthly, monthly, yearly, or custom, allowing users to choose the period that best fits their needs.

## Purpose

The main purpose of this tool is to help generate date ranges based on different rebalancing periods. A "rebalancing period" refers to the frequency at which a portfolio or data is adjusted or updated. For example, a daily rebalancing period means updating the portfolio every day, while a yearly rebalancing period means updating once a year.

## Features

This tool offers several rebalancing period options:

- **Daily Rebalancing**: Generates a date range for every single day in a month.
- **Half-Monthly Rebalancing**: Divides a month into two periods — the first half of the month and the second half.
- **Monthly Rebalancing**: Provides a date range for the entire month.
- **Yearly Rebalancing**: Generates a date range for the whole year.
- **Custom Rebalancing**: Allows you to set a custom rebalancing period, such as every 5 days, 10 days, etc., within a month.

## How It Works

The tool consists of two main components:

### 1. **RebalancingPeriod Enum**

This is a set of predefined options to define how often rebalancing should happen:

- **DAILY**: Rebalancing occurs every day.
- **HALF_MONTHLY**: Rebalancing occurs twice a month, splitting the month into two halves.
- **MONTHLY**: Rebalancing occurs once per month.
- **YEARLY**: Rebalancing occurs once per year.

### 2. **Rebalancing Class**

This class contains methods that generate the specific date ranges based on the chosen rebalancing period.

- **Daily Rebalancing**: The method generates a date range for every day in a month.
- **Half-Monthly Rebalancing**: The method splits a month into two periods — one from the 1st to the 15th, and the other from the 16th to the last day of the month.
- **Monthly Rebalancing**: This method generates one range for the entire month.
- **Yearly Rebalancing**: This generates a range for the entire year (from January 1st to December 31st).
- **Custom Rebalancing**: This method generates date ranges based on a custom period in days, allowing for more flexibility.

### 3. **DateRangeGenerator Class**

This class makes it easy to generate date ranges over multiple years and months. It allows users to specify the rebalancing period and the specific years and months for which they want the date ranges.

### 4. **How to Use**

The tool works by creating an instance of the `DateRangeGenerator` class. You can specify the following:

- **Years**: The year(s) for which you want to generate date ranges.
- **Months**: The month(s) for which you want to generate date ranges.
- **Rebalancing Period**: The rebalancing period, which can be daily, half-monthly, monthly, yearly, or custom.

Once you create an instance of `DateRangeGenerator`, you can call the `get_date_range()` method to get a list of date ranges.

### Example Usage

- **Daily Rebalancing**: 
  If you want daily rebalancing for the month of March 2025, you can create an instance of the `DateRangeGenerator` class like this:
  ```python
  generator = DateRangeGenerator(years=2025, months=3, rebalancing=RebalancingPeriod.DAILY)
  date_ranges = generator.get_date_range()
  print(date_ranges)
