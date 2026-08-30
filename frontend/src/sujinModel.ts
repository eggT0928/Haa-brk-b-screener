import type { Market, PlanLine } from './types';

export const SUJIN_WEIGHTS: Record<string, number> = {
  QQQM: .15, 'SPY+SPYM': .20, JEPQ: .10, 'BRK-B': .15, IEF: .15, TLT: .10, GLD: .10, PDBC: .05,
};
export const SUJIN_TICKERS = ['QQQM','SPY','SPYM','JEPQ','BRK-B','IEF','TLT','GLD','PDBC'] as const;
export type SujinProfile = {
  strategy: 'sujin-fixed-v1'; holdings: Record<string, number>;
  totalMode: 'manual' | 'holdings'; totalBalance: number; cash: number;
  quantityMode: 'fractional' | 'whole'; backtestStart: string; backtestEnd: string;
};
export type SujinRecord = {
  id: string; createdAt?: { seconds: number }; month: string; strategy: 'sujin-fixed-v1';
  kind: '계산안'; equity: number; remainingCash: number; priceUpdatedAt: string;
  oldestPriceAt: string; quantityMode: SujinProfile['quantityMode']; lines: Pick<PlanLine,'ticker'|'held'|'price'|'target'|'trade'>[];
};
export function defaultSujinProfile(): SujinProfile {
  return { strategy:'sujin-fixed-v1', holdings:Object.fromEntries(SUJIN_TICKERS.map(t=>[t,0])),
    totalMode:'manual', totalBalance:0, cash:0, quantityMode:'fractional', backtestStart:'', backtestEnd:'' };
}
export function validateSujinProfile(p: SujinProfile) {
  if (p.strategy !== 'sujin-fixed-v1' || !['manual','holdings'].includes(p.totalMode)
      || !['fractional','whole'].includes(p.quantityMode)) throw new Error('수진 포트폴리오 설정을 확인하세요.');
  for (const value of [p.totalBalance,p.cash]) if (!Number.isFinite(value) || value < 0 || value > 1e12) throw new Error('금액을 확인하세요.');
  if (Object.keys(p.holdings).length !== SUJIN_TICKERS.length || Object.keys(p.holdings).some(t=>!SUJIN_TICKERS.includes(t as typeof SUJIN_TICKERS[number]))) throw new Error('보유종목을 확인하세요.');
  for (const t of SUJIN_TICKERS) if (!Number.isFinite(p.holdings[t]) || p.holdings[t] < 0 || p.holdings[t] > 1e9) throw new Error(`${t} 보유수량을 확인하세요.`);
  for (const date of [p.backtestStart,p.backtestEnd]) if (typeof date !== 'string' || (date !== '' && !/^\d{4}-\d{2}-\d{2}$/.test(date))) throw new Error('기간 형식을 확인하세요.');
}
export function buildSujinPlan(profile: SujinProfile, market: Market) {
  validateSujinProfile(profile);
  for (const t of SUJIN_TICKERS) {
    if (!Number.isFinite(market.prices[t]) || market.prices[t] <= 0 || market.prices[t] > 1e12 || !Number.isFinite(Date.parse(market.priceTimes[t]))) throw new Error(`${t}의 유효한 가격·시각이 필요합니다. 시세를 갱신하세요.`);
  }
  const holdingsValue = SUJIN_TICKERS.reduce((sum,t)=>sum+profile.holdings[t]*market.prices[t],0);
  const equity = profile.totalMode === 'manual' ? profile.totalBalance : holdingsValue + profile.cash;
  if (equity <= 0 || equity > 1e12) throw new Error('총자산을 0 초과, 1조 달러 이하로 입력하세요.');
  if (profile.totalMode === 'manual' && holdingsValue > equity + .01) throw new Error('입력한 총자산이 보유주식 평가액보다 작습니다. 예수금을 포함한 총액을 확인하세요.');
  const spyValue = profile.holdings.SPY * market.prices.SPY;
  if (spyValue > equity * .2 + .000001) throw new Error('기존 SPY만으로 목표 비중 20%를 넘었습니다. SPY 유지 조건으로는 목표를 맞출 수 없어 매매 계산을 중단합니다. SPY를 임의로 매도하거나 SPYM을 초과 매도하지 않습니다.');
  const units = profile.quantityMode === 'whole' ? 1 : 100;
  const lines: PlanLine[] = SUJIN_TICKERS.map(ticker=>{
    const price = market.prices[ticker], held = profile.holdings[ticker];
    const amount = ticker === 'SPY' ? spyValue : ticker === 'SPYM' ? Math.max(0,equity * .2 - spyValue) : equity * SUJIN_WEIGHTS[ticker];
    // SPY는 그대로 유지한다. 신규 목표는 주문 단위에 맞춰 내림하여 예수금을 초과하지 않는다.
    const target = ticker === 'SPY' ? held : Math.floor(amount / price * units) / units;
    if (!Number.isFinite(target) || target > 1e9) throw new Error(`${ticker} 목표수량이 허용 범위를 넘었습니다. 총자산과 가격을 확인하세요.`);
    const trade = Math.round((target - held) * 1e8) / 1e8;
    return {ticker,held,price,weight:amount/equity,target,trade,amount:trade*price};
  });
  const groups = Object.entries(SUJIN_WEIGHTS).map(([ticker,weight])=>{
    const value = ticker === 'SPY+SPYM' ? spyValue + profile.holdings.SPYM*market.prices.SPYM : profile.holdings[ticker]*market.prices[ticker];
    return {ticker,weight,current:value/equity,deviation:value/equity-weight};
  });
  return {equity,holdingsValue,remainingCash:Math.max(0,equity-lines.reduce((sum,l)=>sum+l.target*l.price,0)),lines,groups,
    oldestPriceAt:SUJIN_TICKERS.map(t=>market.priceTimes[t]).sort((a,b)=>Date.parse(a)-Date.parse(b))[0]};
}

export function sujinCsv(lines: PlanLine[]) {
  const rows = [['종목','목표 비중','현재 가격 USD','보유수량','목표수량','매매수량','매매금액 USD'],
    ...lines.map(l=>[l.ticker,(l.weight*100).toFixed(4),l.price,l.held,l.target,l.trade,l.amount.toFixed(2)])];
  return '\uFEFF'+rows.map(row=>row.join(',')).join('\r\n');
}
