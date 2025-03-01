from .yahoo_provider import YahooProvider
from .jugad_provider import JugadProvider
from .custom_provider import CustomProvider
from .base_provider import BaseProvider
from .yahoo_provider_balance_sheet import YahooProviderBalanceSheet
from ..utils.constants import Providers


class ProviderFactory:
    @staticmethod
    def get_provider(provider_name: str, **kwargs) -> BaseProvider:
        """
        Factory method to get the appropriate provider instance.
        :param provider_name: Name of the provider (e.g., "yahoo", "jugad", "custom").
        :param kwargs: Additional arguments for the provider (e.g., file_path for CustomProvider).
        :return: Instance of the requested provider.
        """
        if provider_name == Providers.YAHOO:
            return YahooProvider()
        elif provider_name == Providers.YAHOO_FOR_BALANCE_SHEET:
            return YahooProviderBalanceSheet()
        elif provider_name == Providers.JUGAD:
            return JugadProvider()
        elif provider_name == Providers.CUSTOM:
            if "file_path" not in kwargs:
                raise ValueError("file_path is required for CustomProvider.")
            return CustomProvider(file_path=kwargs["file_path"])
        else:
            raise ValueError(
                f"Incorrect provider: {provider_name}. Available values are: {[e.name for e in Providers]}")
