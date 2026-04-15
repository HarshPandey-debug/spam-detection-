# Binance Futures Testnet Trading Bot (Python)

A production-minded, minimal trading bot for **Binance USDT-M Futures Testnet** that places **MARKET** and **LIMIT** orders from a CLI, with strong validation, clear output, and file logging.

## Why this implementation stands out

- Clean layering:
  - `client` (HTTP signing + Binance API calls)
  - `orders` (order construction + execution workflow)
  - `validators` (input validation/normalization)
  - `cli` (user interface and orchestration)
- Fail-safe behavior with typed custom exceptions.
- Reproducible logs to verify real requests and troubleshoot failures.
- `--dry-run` mode to validate workflow without sending live testnet orders.

## Project structure

```text
trading_bot/
  bot/
    __init__.py
    client.py
    exceptions.py
    logging_config.py
    orders.py
    validators.py
  cli.py
logs/
  market_order.log
  limit_order.log
README.md
requirements.txt
```

## Setup

1. **Create Binance Futures Testnet account**
   - Use: <https://testnet.binancefuture.com>
2. **Generate API credentials** (API Key + Secret).
3. **Install dependencies**:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

4. **Export credentials**:

```bash
export BINANCE_API_KEY="your_key"
export BINANCE_API_SECRET="your_secret"
```

## CLI usage

```bash
python trading_bot/cli.py \
  --symbol BTCUSDT \
  --side BUY \
  --order-type MARKET \
  --quantity 0.001
```

### Required arguments

- `--symbol` (example: `BTCUSDT`)
- `--side` (`BUY` or `SELL`)
- `--order-type` (`MARKET` or `LIMIT`)
- `--quantity` (> 0)
- `--price` (required only if `--order-type LIMIT`)

### Optional arguments

- `--log-file` (default: `logs/trading_bot.log`)
- `--dry-run` (simulate order without calling Binance)

## Run examples

### 1) MARKET order

```bash
python trading_bot/cli.py \
  --symbol BTCUSDT \
  --side BUY \
  --order-type MARKET \
  --quantity 0.001 \
  --log-file logs/market_order.log
```

### 2) LIMIT order

```bash
python trading_bot/cli.py \
  --symbol BTCUSDT \
  --side SELL \
  --order-type LIMIT \
  --quantity 0.001 \
  --price 120000 \
  --log-file logs/limit_order.log
```

### 3) Safe local demo (no credentials needed)

```bash
python trading_bot/cli.py \
  --symbol BTCUSDT \
  --side BUY \
  --order-type MARKET \
  --quantity 0.001 \
  --dry-run
```

## Logging

Logs include:
- request metadata (method/path/safe params)
- response status/body
- structured error context

This repository includes sample logs requested by the assessment:
- `logs/market_order.log`
- `logs/limit_order.log`

## Assumptions

- Bot targets **USDT-M Futures Testnet** base URL: `https://testnet.binancefuture.com`.
- Symbol precision, quantity step, and min notional checks are delegated to Binance validation (improvement opportunity: prefetch exchange info and validate locally).
- LIMIT orders are submitted with `timeInForce=GTC`.

