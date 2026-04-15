"""CLI entrypoint for placing Binance Futures testnet orders."""

from __future__ import annotations

import argparse
import logging
import os
import sys

from bot.client import BinanceFuturesClient
from bot.exceptions import BinanceAPIError, NetworkError, ValidationError
from bot.logging_config import configure_logging
from bot.orders import OrderRequest, OrderService
from bot.validators import validate_inputs

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Binance Futures Testnet order bot")
    parser.add_argument("--symbol", required=True, help="Trading pair symbol, e.g. BTCUSDT")
    parser.add_argument("--side", required=True, choices=["BUY", "SELL", "buy", "sell"])
    parser.add_argument(
        "--order-type",
        required=True,
        choices=["MARKET", "LIMIT", "market", "limit"],
        help="Order type",
    )
    parser.add_argument("--quantity", required=True, help="Order quantity")
    parser.add_argument("--price", help="Limit order price (required for LIMIT orders)")
    parser.add_argument("--log-file", default="logs/trading_bot.log", help="Log file path")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and simulate order without sending to Binance",
    )
    return parser


def print_summary(order: OrderRequest) -> None:
    print("\n=== ORDER REQUEST SUMMARY ===")
    print(f"Symbol     : {order.symbol}")
    print(f"Side       : {order.side}")
    print(f"Order Type : {order.order_type}")
    print(f"Quantity   : {order.quantity}")
    if order.order_type == "LIMIT":
        print(f"Price      : {order.price}")


def print_response(response: dict) -> None:
    print("\n=== ORDER RESPONSE ===")
    print(f"orderId     : {response.get('orderId', 'N/A')}")
    print(f"status      : {response.get('status', 'N/A')}")
    print(f"executedQty : {response.get('executedQty', 'N/A')}")
    print(f"avgPrice    : {response.get('avgPrice', 'N/A')}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    configure_logging(args.log_file)

    try:
        validated = validate_inputs(
            symbol=args.symbol,
            side=args.side,
            order_type=args.order_type,
            quantity=args.quantity,
            price=args.price,
        )

        order = OrderRequest(
            symbol=validated["symbol"],
            side=validated["side"],
            order_type=validated["order_type"],
            quantity=validated["quantity"],
            price=validated["price"],
        )

        print_summary(order)

        api_key = os.getenv("BINANCE_API_KEY", "")
        api_secret = os.getenv("BINANCE_API_SECRET", "")

        client = None
        if not args.dry_run:
            if not api_key or not api_secret:
                raise ValidationError(
                    "BINANCE_API_KEY and BINANCE_API_SECRET must be set unless --dry-run is used"
                )
            client = BinanceFuturesClient(api_key=api_key, api_secret=api_secret)

        service = OrderService(client=client, dry_run=args.dry_run)
        response = service.submit_order(order)
        print_response(response)

        print("\n✅ SUCCESS: order processed.")
        return 0

    except ValidationError as exc:
        logger.error("validation_error error=%s", exc)
        print(f"\n❌ VALIDATION ERROR: {exc}")
    except BinanceAPIError as exc:
        logger.error("api_error error=%s payload=%s", exc, getattr(exc, "payload", {}))
        print(f"\n❌ BINANCE API ERROR: {exc}")
    except NetworkError as exc:
        logger.error("network_error error=%s", exc)
        print(f"\n❌ NETWORK ERROR: {exc}")
    except Exception as exc:  # broad safeguard for CLI UX
        logger.exception("unexpected_error")
        print(f"\n❌ UNEXPECTED ERROR: {exc}")

    return 1


if __name__ == "__main__":
    sys.exit(main())
