import { describe, expect, it } from 'vitest';
import { demoConfirmed, demoMarket } from './demo';
import { buildPlan, defaultProfile, validateProfile } from './portfolio';

describe('실전 수량과 신호 분리', () => {
  it('SPYM을 선택해도 SPY 신호는 바꾸지 않고 SPYM 가격으로 계산한다', () => {
    const p = defaultProfile(); p.sp500 = 'SPYM';
    const plan = buildPlan(p, demoConfirmed, demoMarket);
    expect(plan.lines.find(l => l.ticker === 'SPYM')?.target).toBe(Math.floor(2000/70));
    expect(demoConfirmed.weights.SPY).toBe(.2);
    expect(plan.lines.some(l => l.ticker === 'SPY')).toBe(false);
  });
  it('SPYM 가격 누락 시 SPY 가격을 대신 사용하지 않는다', () => {
    const p = defaultProfile(); p.sp500 = 'SPYM';
    expect(() => buildPlan(p, demoConfirmed, {...demoMarket,prices:{...demoMarket.prices,SPYM:NaN}})).toThrow('SPYM');
  });
  it('기존 SPY 보유분은 매도로, 새 SPYM 목표는 매수로 표시한다', () => {
    const p = defaultProfile(); p.sp500 = 'SPYM'; p.holdings.SPY = 10; p.holdings.SPYM = 1;
    const plan = buildPlan(p, demoConfirmed, demoMarket);
    expect(plan.equity).toBe(16070);
    expect(plan.lines.find(l => l.ticker === 'SPY')?.trade).toBe(-10);
    expect(plan.lines.find(l => l.ticker === 'SPYM')?.trade).toBe(44);
    expect(plan.remainingCash).toBeGreaterThanOrEqual(0);
  });
  it('미선택 보유종목도 평가금액에 포함하고 청산한다', () => {
    const p = defaultProfile(); p.holdings.VNQ = 5;
    const plan = buildPlan(p, demoConfirmed, demoMarket);
    expect(plan.equity).toBe(10450);
    expect(plan.lines.find(l=>l.ticker==='VNQ')?.trade).toBe(-5);
  });
  it.each([-1,NaN,Infinity,1e10])('잘못된 보유수량 %s을 거부한다', n => {
    const p = defaultProfile(); p.holdings.SPY = n;
    expect(() => validateProfile(p)).toThrow();
  });
  it('정수 매수와 소수 보유분 매도를 지원한다', () => {
    const p = defaultProfile(); p.holdings.TIP = .5;
    const plan = buildPlan(p, demoConfirmed, demoMarket);
    expect(plan.lines.find(l=>l.ticker==='TIP')?.trade).toBe(-.5);
    expect(plan.lines.every(l => Number.isInteger(l.target))).toBe(true);
  });
});
