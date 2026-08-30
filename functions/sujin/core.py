"""sujin_portfolio d80e55d의 고정비중 월별 백테스트를 순수 계산으로 이식."""

import math

import numpy as np
import pandas as pd

WEIGHTS = {
    "QQQM": 0.15,
    "SPY": 0.10,
    "SPYM": 0.10,
    "JEPQ": 0.10,
    "BRK-B": 0.15,
    "IEF": 0.15,
    "TLT": 0.10,
    "GLD": 0.10,
    "PDBC": 0.05,
}
TICKERS = list(WEIGHTS)
VERSION = "sujin-fixed-spy10-spym10-pdbc-v1"


def backtest(monthly, irx, start=None, end=None, initial=10000.0):
    if not math.isfinite(initial) or not 0 < initial <= 1e12:
        raise ValueError("초기 금액은 0 초과, 1조 달러 이하로 입력하세요.")
    if not set(TICKERS).issubset(monthly.columns):
        raise ValueError("수진 백테스트 필수 종목이 누락되었습니다.")
    data = monthly[TICKERS].sort_index()
    complete = data.dropna()
    if len(complete) < 2:
        raise ValueError("모든 종목이 있는 완료 월말이 최소 2개 필요합니다.")
    first, last = complete.index[0], complete.index[-1]
    # 상장 전 기간은 제외하되, 중간·마지막 월 누락은 종목 재배분이나 0%로 감추지 않는다.
    data = data.loc[first:]
    if not np.isfinite(data.to_numpy()).all() or (data <= 0).any().any():
        raise ValueError("필수 월말 가격에 누락값이 있습니다. 일부 자산만으로 계산하지 않습니다.")
    if not data.index.equals(pd.date_range(first, data.index[-1], freq="ME")):
        raise ValueError("백테스트 중간에 누락된 월이 있습니다.")
    begin, finish = pd.Timestamp(start) if start else first, pd.Timestamp(end) if end else last
    if pd.isna(begin) or pd.isna(finish) or begin > finish:
        raise ValueError("백테스트 기간을 확인하세요.")
    selected = data.loc[max(begin, first) : min(finish, last)]
    if len(selected) < 2:
        raise ValueError("선택한 기간에는 최소 2개의 월말이 필요합니다.")
    # 원본처럼 SPY 10% + SPYM 10%, PDBC 자체를 사용한다. HAA DBC 프록시는 적용하지 않는다.
    returns = selected.pct_change(fill_method=None).iloc[1:].mul(pd.Series(WEIGHTS)).sum(axis=1)
    curve = pd.concat([pd.Series([initial], index=selected.index[:1]), initial * (1 + returns).cumprod()])
    drawdown = curve / curve.cummax() - 1
    years = (curve.index[-1] - curve.index[0]).days / 365.25
    cagr = (curve.iloc[-1] / initial) ** (1 / years) - 1
    volatility = float(returns.std() * np.sqrt(12)) if len(returns) > 1 else 0.0
    rates = irx.loc[curve.index[0] : curve.index[-1]].dropna()
    rf = float(rates.mean() / 100) if len(rates) and np.isfinite(rates).all() else None
    yearly = (1 + returns).groupby(returns.index.year).prod() - 1
    return {
        "version": VERSION,
        "basis": "SPY 10% + SPYM 10%",
        "commodityProxy": "없음 (PDBC)",
        "availableStart": first.date().isoformat(),
        "availableEnd": last.date().isoformat(),
        "actualStart": curve.index[0].date().isoformat(),
        "actualEnd": curve.index[-1].date().isoformat(),
        "metrics": {
            "totalReturn": float(curve.iloc[-1] / initial - 1),
            "cagr": float(cagr),
            "volatility": volatility,
            "mdd": float(drawdown.min()),
            "riskFree": rf,
            "sharpe": float((cagr - rf) / volatility) if rf is not None and volatility > 0 else None,
        },
        "curve": [
            {"date": d.date().isoformat(), "value": float(v), "drawdown": float(drawdown[d])}
            for d, v in curve.items()
        ],
        "monthly": [{"month": d.strftime("%Y-%m"), "return": float(v)} for d, v in returns.items()],
        "yearly": [{"year": int(y), "return": float(v)} for y, v in yearly.items()],
        "history": [
            {
                "signalMonth": previous.strftime("%Y-%m"),
                "returnMonth": current.strftime("%Y-%m"),
                "weights": WEIGHTS.copy(),
                "return": float(returns[current]),
            }
            for previous, current in zip(selected.index[:-1], selected.index[1:])
        ],
        "warnings": []
        if rf is not None
        else ["선택 기간의 ^IRX가 없어 무위험수익률·샤프를 표시하지 않습니다."],
    }
