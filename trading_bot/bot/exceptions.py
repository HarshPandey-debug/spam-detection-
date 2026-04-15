"""Custom exceptions for the trading bot."""


class TradingBotError(Exception):
    """Base exception for trading bot."""


class ValidationError(TradingBotError):
    """Raised when CLI inputs are invalid."""


class BinanceAPIError(TradingBotError):
    """Raised when Binance API returns an error payload."""

    def __init__(self, code: int | None, message: str, payload: dict | None = None) -> None:
        self.code = code
        self.payload = payload or {}
        super().__init__(f"Binance API error {code}: {message}" if code is not None else message)


class NetworkError(TradingBotError):
    """Raised when network/transport issue happens."""
