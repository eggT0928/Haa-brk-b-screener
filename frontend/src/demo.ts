import type { Market, Signal } from './types';
// 데모는 화면 검토용 가상 값이다. API 및 Firestore와 연결하지 않는다.
const scores = { SPY: .14, VEA: .11, VWO: .05, IWM: .09, BIL: .02, IEF: .03, TLT: -.03, TIP: .015, PDBC: .07, VNQ: .01, 'BRK-B': .12 };
export const demoConfirmed: Signal = {
  kind: 'confirmed', month: '2026-07', asOf: '2026-07-31T20:00:00Z', updatedAt: '2026-07-31T21:30:00Z',
  regime: '공격', selected: ['SPY','VEA','IWM','PDBC'], weights: { SPY: .2, VEA: .2, IWM: .2, PDBC: .2, 'BRK-B': .2 }, scores,
  returns: Object.fromEntries([1,3,6,12].map(n => [String(n), Object.fromEntries(Object.entries(scores).map(([t,s]) => [t,s * n / 5.5]))])),
};
export const demoPreview: Signal = { ...demoConfirmed, kind: 'preview', month: '2026-08', regime: '방어',
  selected: ['IEF'], weights: { IEF: .8, 'BRK-B': .2 }, scores: { ...scores, TIP: -.003 },
  returns: Object.fromEntries(Object.entries(demoConfirmed.returns).map(([n,r]) => [n, {...r,TIP:-.003}])),
  asOf: '2026-08-28T19:45:00Z', updatedAt: '2026-08-28T19:45:00Z', oldestPriceAt: '2026-08-28T19:44:00Z',
};
export const demoMarket: Market = {
  prices: { SPY: 600, SPYM: 70, VEA: 55, VWO: 45, IWM: 210, BIL: 92, IEF: 95, TLT: 88, TIP: 108, PDBC: 14, VNQ: 90, 'BRK-B': 480 },
  priceTimes: Object.fromEntries(Object.keys(scores).concat('SPYM').map(t => [t, '2026-08-28T19:44:00Z'])),
  updatedAt: '2026-08-28T19:45:00Z',
};
