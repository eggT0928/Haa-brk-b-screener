import { describe, expect, it } from 'vitest';
import { demoConfirmed, demoMarket } from './demo';
import { buildPlan, defaultProfile, quantityLabel, validateProfile } from './portfolio';

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
  it('기존 저장 문서의 수량 단위 누락은 정수 목표로 해석한다',()=>{
    const legacy=defaultProfile();delete legacy.quantityMode;
    expect(()=>validateProfile(legacy)).not.toThrow();
    expect(buildPlan(legacy,demoConfirmed,demoMarket).lines.every(l=>Number.isInteger(l.target))).toBe(true);
    expect(()=>validateProfile({...legacy,quantityMode:'unsupported' as never})).toThrow('수량 단위');
  });
  it('소수점 목표는 여섯 자리로 내림하고 남은 현금이 음수가 되지 않는다',()=>{
    const p={...defaultProfile(),quantityMode:'fractional' as const};
    const fractional=buildPlan(p,demoConfirmed,demoMarket);
    expect(fractional.lines.find(l=>l.ticker==='SPY')?.target).toBe(3.333333);
    for(const line of fractional.lines)expect(line.target*line.price).toBeLessThanOrEqual(fractional.equity*line.weight+1e-9);
    expect(fractional.remainingCash).toBeGreaterThanOrEqual(0);
    expect(fractional.remainingCash).toBeLessThan(buildPlan({...p,quantityMode:'whole'},demoConfirmed,demoMarket).remainingCash);
    expect(demoConfirmed.weights.SPY).toBe(.2);
  });
  it('소수점 모드에서도 SPYM 원시가격을 적용하고 단위 변경은 보유수량을 바꾸지 않는다',()=>{
    const p={...defaultProfile(),quantityMode:'fractional' as const};p.sp500='SPYM';p.holdings.SPY=.12345678;
    const original={...p.holdings};
    const fractional=buildPlan(p,demoConfirmed,demoMarket);
    expect(fractional.lines.find(l=>l.ticker==='SPY')?.trade).toBe(-.12345678);
    expect(fractional.lines.find(l=>l.ticker==='SPYM')?.target).toBe(Math.floor(fractional.equity*.2/70*1e6)/1e6);
    const whole=buildPlan({...p,quantityMode:'whole'},demoConfirmed,demoMarket);
    expect(whole.lines.every(l=>Number.isInteger(l.target))).toBe(true);
    expect(p.holdings).toEqual(original);
    expect(quantityLabel(-.12345678)).toBe('-0.12345678');
    expect(quantityLabel(.000001)).toBe('0.000001');
  });
});
