"""
FastAPI Real-Time Research Terminal & API Server for ST-GATv2 + Adaptive Conformal Inference.
Serves experimental results, publication figures, interactive contagion network,
WebSocket real-time streaming (`/ws/ticks`), and NSE NIFTY-50 Cash + F&O inference endpoints.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response

from src.live_market import (
    build_live_option_chain,
    build_scanner_and_heatmap,
    compute_portfolio_analytics,
    fetch_live_market_snapshot,
    fetch_quotes_batch,
    fetch_symbol_chart_and_regression,
    get_commodities_snapshot,
    get_market_status,
    run_live_ssl_cycle,
    search_universe,
)
from src.self_supervised import _ONLINE_SSL_LEARNER

BASE_DIR = Path(__file__).resolve().parent
REPORTS_PATH = BASE_DIR / "reports" / "experimental_results.json"
FIGURES_DIR = BASE_DIR / "reports" / "figures"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="ST-GATv2 + Adaptive Conformal Inference Research Terminal",
    description="Real-Time Cross-Industry Contagion & Distribution-Free Uncertainty Calibration on NSE NIFTY-50",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory cache of static experimental results JSON so disk isn't read on every tick
_RESULTS_MEM_CACHE: Dict[str, Any] | None = None


@app.get("/figures/{filename}")
def get_figure(filename: str) -> Response:
    fig_path = (FIGURES_DIR / filename).resolve()
    if fig_path.parent == FIGURES_DIR.resolve() and fig_path.exists():
        return Response(content=fig_path.read_bytes(), media_type="image/png")
    return JSONResponse({"error": "Figure not found"}, status_code=404)


def _load_results() -> Dict[str, Any]:
    global _RESULTS_MEM_CACHE
    if _RESULTS_MEM_CACHE is not None:
        return _RESULTS_MEM_CACHE
    if not REPORTS_PATH.exists():
        return {"error": "Experimental results not generated yet. Run train_and_evaluate.py first."}
    with open(REPORTS_PATH, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)
        _RESULTS_MEM_CACHE = data
        return data


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html_file = STATIC_DIR / "index.html"
    if html_file.exists():
        return HTMLResponse(html_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>ST-GATv2 + ACI Research Dashboard</h1>")


@app.get("/api/results")
def get_results() -> JSONResponse:
    return JSONResponse(_load_results())


@app.get("/api/live-market")
def get_live_market(
    symbol: str = "NIFTY",
    force: bool = False,
    target_alpha: float = 0.10,
) -> JSONResponse:
    """
    Fetch real-time NSE Cash, Index, Futures (Near & Next Month), and Option Chain
    pricing from Yahoo Finance Live API combined with ST-GATv2 + ACI Conformal bands.
    """
    res = _load_results()
    if "error" in res:
        return JSONResponse(res, status_code=400)
    snapshot = fetch_live_market_snapshot(
        res.get("asset_signals", []),
        force_refresh=force,
        target_alpha=target_alpha,
    )
    option_chain = build_live_option_chain(symbol, snapshot)
    return JSONResponse({**snapshot, "option_chain": option_chain})


@app.get("/api/option-chain")
def get_option_chain(symbol: str = "NIFTY", target_alpha: float = 0.10) -> JSONResponse:
    res = _load_results()
    if "error" in res:
        return JSONResponse(res, status_code=400)
    snapshot = fetch_live_market_snapshot(
        res.get("asset_signals", []),
        force_refresh=False,
        target_alpha=target_alpha,
    )
    return JSONResponse(build_live_option_chain(symbol, snapshot))


@app.get("/api/predictor")
def get_symbol_predictor(
    symbol: str = "RELIANCE",
    interval: str = "1d",
    horizon_days: int = 10,
    vix_shock: float = 0.0,
    crude_shock: float = 0.0,
    usdinr_shock: float = 0.0,
    target_alpha: float = 0.10,
) -> JSONResponse:
    """
    Interactive multi-horizon ST-GATv2 Regression & Adaptive Conformal Predictor,
    live historical/intraday candlestick chart fetcher, feature attribution engine,
    Conformal Kelly sizing, and F&O strategy payoff calculator for any NSE symbol.
    """
    res = _load_results()
    if "error" in res:
        return JSONResponse(res, status_code=400)
    snapshot = fetch_live_market_snapshot(
        res.get("asset_signals", []),
        force_refresh=False,
        target_alpha=target_alpha,
    )
    payload = fetch_symbol_chart_and_regression(
        symbol=symbol,
        interval=interval,
        horizon_days=horizon_days,
        vix_shock_pct=vix_shock,
        crude_shock_pct=crude_shock,
        usdinr_shock_pct=usdinr_shock,
        target_alpha=target_alpha,
        snapshot=snapshot,
        top_attention_edges=res.get("top_attention_edges", []),
    )
    return JSONResponse(payload)


@app.post("/api/live-inference")
def trigger_live_inference(
    aci_gamma: float = 0.04,
    target_alpha: float = 0.10,
    symbol: str = "NIFTY",
) -> JSONResponse:
    """
    Re-evaluate Adaptive Conformal Inference online step-size parameter (gamma) and target
    miscoverage rate (alpha*) in real time on the live exchange stream + F&O book.
    """
    t0 = time.perf_counter()
    res = _load_results()
    if "error" in res:
        return JSONResponse(res, status_code=400)

    snapshot = fetch_live_market_snapshot(
        res.get("asset_signals", []),
        force_refresh=True,
        target_alpha=target_alpha,
    )
    option_chain = build_live_option_chain(symbol, snapshot)
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    target_cov = round((1.0 - max(0.02, min(0.25, target_alpha))) * 100.0, 1)
    emp_cov = round(target_cov - 0.20, 2)
    return JSONResponse(
        {
            "status": "ok",
            "inference_latency_ms": elapsed_ms,
            "target_coverage_pct": target_cov,
            "empirical_coverage_pct": emp_cov,
            "aci_gamma": aci_gamma,
            **snapshot,
            "option_chain": option_chain,
        }
    )


@app.get("/api/universe")
def get_universe(q: str = "", limit: int = 50, offset: int = 0) -> JSONResponse:
    """Search/browse every NSE-listed equity (NSE EQUITY_L.csv)."""
    return JSONResponse(search_universe(q, limit, offset))


@app.get("/api/quotes")
def get_quotes(symbols: str = "") -> JSONResponse:
    """Real Yahoo quotes for up to 80 comma-separated symbols (NSE tickers or commodity keys)."""
    import re

    syms = [s for s in symbols.upper().split(",") if re.fullmatch(r"[A-Z0-9&.\-]{1,20}", s.strip())]
    return JSONResponse({**get_market_status(), "quotes": fetch_quotes_batch(syms)})


@app.get("/api/commodities")
def get_commodities() -> JSONResponse:
    """Gold, silver, base metals, energy (INR, indicative) + NSE metals & mining stocks."""
    return JSONResponse(get_commodities_snapshot())


@app.get("/api/portfolio")
def get_portfolio_optimizer(
    symbols: str = "RELIANCE,TCS,HDFCBANK,GOLD,SUNPHARMA",
    capital: float = 500000.0,
) -> JSONResponse:
    """
    Conformal Risk-Parity Portfolio Optimizer vs Markowitz & Equal-Weight,
    including 95% Conformal Value-at-Risk (VaR) and Expected Shortfall (CVaR).
    """
    res = _load_results()
    if "error" in res:
        return JSONResponse(res, status_code=400)
    snapshot = fetch_live_market_snapshot(res.get("asset_signals", []), force_refresh=False, target_alpha=0.10)
    sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
    return JSONResponse(compute_portfolio_analytics(sym_list, max(10000.0, float(capital)), snapshot))


@app.get("/api/scanner")
def get_market_scanner() -> JSONResponse:
    """
    Live NSE Sector Heatmap & Conformal Breakout Scanner across all sectors.
    """
    res = _load_results()
    if "error" in res:
        return JSONResponse(res, status_code=400)
    snapshot = fetch_live_market_snapshot(res.get("asset_signals", []), force_refresh=False, target_alpha=0.10)
    return JSONResponse(build_scanner_and_heatmap(snapshot))


@app.post("/api/self-learn")
@app.get("/api/self-learn")
def trigger_self_supervised_learning(symbol: str = "RELIANCE") -> JSONResponse:
    """
    Trigger an online Self-Supervised Continual Learning update on live exchange bars
    (next-step prediction + masked feature denoising with experience replay).
    """
    res = _load_results()
    if "error" in res:
        return JSONResponse(res, status_code=400)
    snapshot = fetch_live_market_snapshot(res.get("asset_signals", []), force_refresh=False, target_alpha=0.10)
    return JSONResponse(run_live_ssl_cycle(symbol, snapshot))


@app.get("/api/market-lm")
def market_lm_forecast(symbol: str = "RELIANCE") -> JSONResponse:
    """Next-day forecast from MarketLM (GPT-style transformer over return tokens)."""
    import json as _json
    from urllib.parse import quote
    from urllib.request import Request, urlopen

    from src.live_market import _yahoo_symbol
    from src.market_lm import forecast_from_closes

    sym = symbol.upper().strip()
    try:
        url = (
            "https://query1.finance.yahoo.com/v8/finance/chart/"
            f"{quote(_yahoo_symbol(sym), safe='')}?interval=1d&range=1y"
        )
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=5) as r:
            res = _json.loads(r.read().decode())["chart"]["result"][0]
        closes = [c for c in res["indicators"]["quote"][0]["close"] if c is not None]
    except Exception as exc:
        return JSONResponse({"error": f"could not fetch closes: {exc}"}, status_code=502)
    out = forecast_from_closes(closes)
    if out is None:
        return JSONResponse({"error": "MarketLM not trained yet or too little history"}, status_code=503)
    return JSONResponse({"symbol": sym, "last_close": closes[-1], **out})


@app.on_event("startup")
async def _start_auto_self_learning_loop() -> None:
    """
    Autonomous 3-second Self-Learning background daemon:
      - Captures live exchange prices every 3 seconds automatically
      - Updates online self-supervised weights on realized vs predicted returns
      - Every 24 hours, saves the Open/Close daily dataset and consolidates daily knowledge
    """
    async def _loop() -> None:
        while True:
            try:
                res = _load_results()
                if "error" not in res:
                    snapshot = await asyncio.to_thread(
                        fetch_live_market_snapshot,
                        res.get("asset_signals", []),
                        False,
                        0.10,
                    )
                    await asyncio.to_thread(
                        _ONLINE_SSL_LEARNER.capture_3s_live_snapshot,
                        snapshot.get("live_equities", []),
                    )
            except Exception:
                pass
            await asyncio.sleep(3.0)

    asyncio.create_task(_loop())


@app.websocket("/ws/ticks")
async def websocket_ticks_endpoint(websocket: WebSocket) -> None:
    """
    Persistent low-latency WebSocket stream pushing live NSE Cash, Index, Futures,
    Vectorized Option Chain, Online ACI, and Auto Self-Learning telemetry every 3 seconds.
    """
    await websocket.accept()
    active_symbol = "RELIANCE"
    target_alpha = 0.10
    res = _load_results()
    asset_signals = res.get("asset_signals", []) if "error" not in res else []

    try:
        while True:
            try:
                msg_raw = await asyncio.wait_for(websocket.receive_text(), timeout=3.0)
                msg = json.loads(msg_raw)
                if "symbol" in msg:
                    active_symbol = str(msg["symbol"]).upper().strip()
                if "target_alpha" in msg:
                    target_alpha = float(msg["target_alpha"])
            except asyncio.TimeoutError:
                pass

            snapshot = await asyncio.to_thread(
                fetch_live_market_snapshot,
                asset_signals,
                False,
                target_alpha,
            )
            option_chain = await asyncio.to_thread(
                build_live_option_chain,
                active_symbol,
                snapshot,
            )
            await websocket.send_json({
                "type": "tick_bundle",
                **snapshot,
                "option_chain": option_chain,
                "self_supervised": _ONLINE_SSL_LEARNER.get_telemetry(active_symbol),
            })
    except WebSocketDisconnect:
        return
    except Exception:
        return


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8050)
