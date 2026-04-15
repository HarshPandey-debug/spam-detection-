"""Order service layer that composes validated parameters and calls the API client."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .client import BinanceFuturesClient

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class OrderRequest:
    symbol: str
    side: str
    order_type: str
    quantity: str
    price: str | None = None

    def to_api_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            "symbol": self.symbol,
            "side": self.side,
            "type": self.order_type,
            "quantity": self.quantity,
        }

        if self.order_type == "LIMIT":
            params["price"] = self.price
            params["timeInForce"] = "GTC"

        return params


class OrderService:
    def __init__(self, client: BinanceFuturesClient | None, dry_run: bool = False) -> None:
        self.client = client
        self.dry_run = dry_run

    def submit_order(self, order_request: OrderRequest) -> dict[str, Any]:
        api_params = order_request.to_api_params()
        logger.info("order_submit request=%s", api_params)

        if self.dry_run:
            dry_response = {
                "orderId": 999999,
                "status": "NEW" if order_request.order_type == "LIMIT" else "FILLED",
                "executedQty": "0" if order_request.order_type == "LIMIT" else order_request.quantity,
                "avgPrice": order_request.price or "0",
                "symbol": order_request.symbol,
                "side": order_request.side,
                "type": order_request.order_type,
                "clientNote": "dry-run mode, no request sent",
            }
            logger.info("order_submit_dry_run response=%s", dry_response)
            return dry_response

        if self.client is None:
            raise ValueError("client must be provided when dry_run=False")

        response = self.client.place_order(api_params)
        logger.info("order_submit_success response=%s", response)
        return response
