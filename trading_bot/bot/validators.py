"""Validation utilities for CLI order inputs."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

from .exceptions import ValidationError

VALID_SIDES = {"BUY", "SELL"}
VALID_ORDER_TYPES = {"MARKET", "LIMIT"}


def normalize_symbol(symbol: str) -> str:
    normalized = symbol.strip().upper()
    if not normalized or not normalized.isalnum():
        raise ValidationError("symbol must be alphanumeric, e.g. BTCUSDT")
    if len(normalized) < 6:
        raise ValidationError("symbol looks too short; expected values like BTCUSDT")
    return normalized


def normalize_side(side: str) -> str:
    normalized = side.strip().upper()
    if normalized not in VALID_SIDES:
        raise ValidationError(f"side must be one of: {', '.join(sorted(VALID_SIDES))}")
    return normalized


def normalize_order_type(order_type: str) -> str:
    normalized = order_type.strip().upper()
    if normalized not in VALID_ORDER_TYPES:
        raise ValidationError(
            f"order_type must be one of: {', '.join(sorted(VALID_ORDER_TYPES))}"
        )
    return normalized


def normalize_positive_decimal(value: Any, field_name: str) -> str:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValidationError(f"{field_name} must be numeric") from exc

    if parsed <= 0:
        raise ValidationError(f"{field_name} must be greater than 0")

    return format(parsed.normalize(), "f")


def validate_inputs(
    symbol: str,
    side: str,
    order_type: str,
    quantity: Any,
    price: Any | None,
) -> dict[str, str]:
    normalized_symbol = normalize_symbol(symbol)
    normalized_side = normalize_side(side)
    normalized_type = normalize_order_type(order_type)
    normalized_quantity = normalize_positive_decimal(quantity, "quantity")

    normalized_price = None
    if normalized_type == "LIMIT":
        if price is None:
            raise ValidationError("price is required when order_type=LIMIT")
        normalized_price = normalize_positive_decimal(price, "price")

    return {
        "symbol": normalized_symbol,
        "side": normalized_side,
        "order_type": normalized_type,
        "quantity": normalized_quantity,
        "price": normalized_price,
    }
