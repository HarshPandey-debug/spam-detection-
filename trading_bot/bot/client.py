"""HTTP client for Binance USDT-M Futures testnet."""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
from typing import Any
from urllib.parse import urlencode

import requests

from .exceptions import BinanceAPIError, NetworkError

logger = logging.getLogger(__name__)


class BinanceFuturesClient:
    """Minimal client wrapper around Binance Futures REST endpoints."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str = "https://testnet.binancefuture.com",
        timeout: int = 15,
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def _sign(self, query_string: str) -> str:
        return hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        signed: bool = False,
    ) -> dict[str, Any]:
        params = params.copy() if params else {}
        headers = {}

        if signed:
            params["timestamp"] = int(time.time() * 1000)
            query = urlencode(params, doseq=True)
            params["signature"] = self._sign(query)
            headers["X-MBX-APIKEY"] = self.api_key

        url = f"{self.base_url}{path}"

        logger.info(
            "api_request method=%s path=%s signed=%s params=%s",
            method,
            path,
            signed,
            {k: v for k, v in params.items() if k != "signature"},
        )

        try:
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                headers=headers,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            logger.exception("network_error method=%s path=%s", method, path)
            raise NetworkError(f"Network error while calling Binance: {exc}") from exc

        text_body = response.text
        logger.info(
            "api_response status=%s path=%s body=%s",
            response.status_code,
            path,
            text_body,
        )

        payload: dict[str, Any]
        try:
            payload = response.json()
        except ValueError:
            if response.ok:
                return {"raw": text_body}
            raise BinanceAPIError(response.status_code, f"Non-JSON error response: {text_body}")

        if not response.ok:
            raise BinanceAPIError(payload.get("code"), payload.get("msg", "Unknown API error"), payload)

        return payload

    def place_order(self, params: dict[str, Any]) -> dict[str, Any]:
        """Place a futures order."""
        return self._request("POST", "/fapi/v1/order", params=params, signed=True)
