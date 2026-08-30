export const TICKERS = ['SPY','SPYM','VEA','VWO','IWM','BIL','IEF','TLT','TIP','PDBC','VNQ','BRK-B'] as const;
export type Sp500 = 'SPY' | 'SPYM';
export type Signal = {
  kind: 'confirmed' | 'preview'; month: string; asOf: string; updatedAt: string;
  regime: string; selected: string[]; weights: Record<string, number>;
  scores: Record<string, number>; returns: Record<string, Record<string, number>>;
  oldestPriceAt?: string; sessionDate?: string;
};
export type Profile = { holdings: Record<string, number>; cash: number; sp500: Sp500; backtestStart: string; backtestEnd: string };
export type Market = { prices: Record<string, number>; priceTimes: Record<string, string>; updatedAt: string };
export type UpdateStatus = { ok: boolean; message: string; lastAttemptAt: string };
export type PlanLine = { ticker: string; held: number; price: number; weight: number; target: number; trade: number; amount: number };
export type Rebalance = { id: string; signalMonth: string; sp500: Sp500; equity: number; kind: string; lines: PlanLine[] };
export type Backtest = {
  availableStart: string; availableEnd: string; actualStart: string; actualEnd: string; dataUpdatedAt: string;
  metrics: { totalReturn: number; cagr: number; volatility: number; mdd: number; riskFree: number | null; sharpe: number | null };
  curve: { date: string; value: number; drawdown: number }[];
  monthly: { month: string; return: number }[]; yearly: { year: number; return: number }[];
  history: { signalMonth: string; returnMonth: string; weights: Record<string, number>; return: number }[];
  warnings: string[];
};
