# Financial Data Providers Library

## Summary

This module provides a flexible and scalable way to fetch financial data from multiple sources with built-in caching for
performance optimization. The factory pattern enables easy integration of new data sources, ensuring adaptability to
different data requirements.

## Features

- **Multiple Data Sources**: Supports various financial data providers, making it easy to switch between them.
- **Caching Mechanism**: Reduces redundant API calls and improves response times by storing frequently accessed data.
- **Factory Pattern Implementation**: Enables seamless addition of new data providers without modifying the core logic.
- **Extensibility**: Easily extendable by implementing a new provider class and registering it with the factory.

## Usage

1. **Initialize the Provider Factory**
   ```python
   from provider_factory import ProviderFactory

   factory = ProviderFactory()
   provider = factory.get_provider("yahoo")
   ```

2. **Fetch Financial Data**
   ```python
   balance_sheet = provider.get_balance_sheet("AAPL")
   print(balance_sheet)
   ```

3. **Add a New Provider**
    - Create a new provider class implementing the required methods.
    - Register the new provider in `provider_factory.py`.
    - The new provider will automatically integrate into the workflow.

## Benefits

- **Performance Optimization**: The caching layer improves data retrieval efficiency.
- **Modular Design**: Easily maintainable and extendable architecture.
- **Scalability**: Can accommodate multiple financial data sources with minimal changes.

This module is ideal for financial analysts, developers, and researchers looking to streamline data retrieval processes
efficiently.

