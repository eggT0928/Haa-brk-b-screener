import { TICKERS, type Market, type Profile, type Signal } from './types';

export function defaultProfile(): Profile {
  return { holdings: Object.fromEntries(TICKERS.map(t => [t, 0])), cash: 10000, sp500: 'SPY', backtestStart: '', backtestEnd: '' };
}

export function validateProfile(profile: Profile) {
  if (!['SPY', 'SPYM'].includes(profile.sp500)) throw new Error('실전 ETF를 선택하세요.');
  if (!Number.isFinite(profile.cash) || profile.cash < 0 || profile.cash > 1e12) throw new Error('현금 금액을 확인하세요.');
  if (Object.keys(profile.holdings).some(t => !(TICKERS as readonly string[]).includes(t))) throw new Error('지원하지 않는 보유종목입니다.');
  for (const ticker of TICKERS) {
    const n = profile.holdings[ticker];
    if (!Number.isFinite(n) || n < 0 || n > 1e9) throw new Error(`${ticker} 보유수량을 확인하세요.`);
  }
}

export function buildPlan(profile: Profile, signal: Signal, market: Market) {
  validateProfile(profile);
  const weights = Object.fromEntries(Object.entries(signal.weights).map(([t, w]) => [t === 'SPY' ? profile.sp500 : t, w]));
  const tickers = TICKERS.filter(t => profile.holdings[t] > 0 || (weights[t] ?? 0) > 0);
  for (const ticker of tickers) {
    if (!Number.isFinite(market.prices[ticker]) || market.prices[ticker] <= 0) {
      // SPYM 가격이 없을 때 SPY 가격을 대신 쓰면 수량이 잘못되므로 중단한다.
      throw new Error(`${ticker} 가격이 없어 계산할 수 없습니다. 다음 갱신을 기다려 주세요.`);
    }
  }
  const equity = profile.cash + tickers.reduce((sum, t) => sum + profile.holdings[t] * market.prices[t], 0);
  const lines = tickers.map(ticker => {
    const price = market.prices[ticker], held = profile.holdings[ticker], weight = weights[ticker] ?? 0;
    const target = Math.floor(equity * weight / price);
    const trade = Math.round((target - held) * 1e8) / 1e8;
    return { ticker, held, price, weight, target, trade, amount: trade * price };
  });
  return { equity, lines, remainingCash: equity - lines.reduce((sum, l) => sum + l.target * l.price, 0) };
}

export const percent = (n: number | null | undefined) => n == null ? '—' : `${(n * 100).toFixed(2)}%`;
export const money = (n: number) => new Intl.NumberFormat('ko-KR', { style: 'currency', currency: 'USD', maximumFractionDigits: 2 }).format(n);
export const timeLabel = (s?: string) => s ? new Date(s).toLocaleString('ko-KR', { timeZone: 'Asia/Seoul', hour12: false }) + ' KST' : '아직 없음';
