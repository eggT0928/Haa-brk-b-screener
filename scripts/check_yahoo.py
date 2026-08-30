"""배포·Firestore 쓰기 없이 Yahoo 실데이터로 계산과 캐시 크기를 점검한다."""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "functions"))

from haa.calendar import completed_month  # noqa: E402
from haa.core import backtest, signal  # noqa: E402
from haa.market import YahooMarket  # noqa: E402
from haa.store import pack  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--as-of", help="과거 기준시각으로 검증할 때만 지정 (시간대 포함 ISO 8601)")
args = parser.parse_args()
now = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp.now(tz="UTC")
adjusted, raw = YahooMarket().daily(now, full=True)
month, session, close = completed_month(now)
confirmed = signal(adjusted.loc[:session], month, "confirmed", close.isoformat())
result = backtest(adjusted, raw["^IRX"], session)
print(
    {
        "조회시각": now.isoformat(),
        "확정월": confirmed["month"],
        "자산선택": confirmed["selected"],
        "백테스트기간": [result["actualStart"], result["actualEnd"]],
        "압축캐시바이트": len(pack(adjusted, raw)),
        "수익률개월수": len(result["monthly"]),
        "IRX평균": result["metrics"]["riskFree"],
    }
)
