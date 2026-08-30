import { describe, expect, it } from 'vitest';
import { buildSujinPlan, defaultSujinProfile, SUJIN_TICKERS, sujinCsv, validateSujinProfile } from './sujinModel';
import type { Market } from './types';
const market: Market = {prices:Object.fromEntries(SUJIN_TICKERS.map(t=>[t,100])),
  priceTimes:Object.fromEntries(SUJIN_TICKERS.map(t=>[t,'2026-08-28T20:00:00Z'])),updatedAt:'2026-08-28T21:30:00Z'};
const profile = ()=>({...defaultSujinProfile(),totalBalance:10000});
describe('수진 고정비중 수량 계산',()=>{
  it('원본 목표 비중과 SPY 유지·SPYM 조정 규칙을 보존한다',()=>{
    const p=profile();p.holdings.SPY=5;p.holdings.SPYM=4;
    const plan=buildSujinPlan(p,market);
    expect(plan.lines.find(l=>l.ticker==='SPY')).toMatchObject({held:5,target:5,trade:0});
    expect(plan.lines.find(l=>l.ticker==='SPYM')).toMatchObject({target:15,trade:11});
    expect(plan.lines.find(l=>l.ticker==='QQQM')?.target).toBe(15);
    expect(plan.lines.find(l=>l.ticker==='PDBC')?.target).toBe(5);
  });
  it('SPY 초과보유 시 음수 SPYM 목표나 초과매도를 생성하지 않는다',()=>{
    const p=profile();p.holdings.SPY=30;p.holdings.SPYM=5;
    expect(()=>buildSujinPlan(p,market)).toThrow('SPY만으로');
  });
  it('SPY가 정확히 20%이면 SPYM 목표는 0이다',()=>{
    const p=profile();p.holdings.SPY=20;p.holdings.SPYM=5;
    expect(buildSujinPlan(p,market).lines.find(l=>l.ticker==='SPYM')).toMatchObject({target:0,trade:-5});
  });
  it('보유주식과 현금 합산 모드에서도 동일한 금액으로 계산한다',()=>{
    const p=profile();p.totalMode='holdings';p.cash=9500;p.holdings.SPY=5;
    expect(buildSujinPlan(p,market).equity).toBe(10000);
  });
  it('정수·소수점 단위 내림 후 현금이 음수가 되지 않는다',()=>{
    for(const quantityMode of ['whole','fractional'] as const) for(let i=1;i<=50;i++) {
      const p={...profile(),quantityMode,totalBalance:10000+i*7.13};p.holdings.SPY=1.2345;
      const m={...market,prices:Object.fromEntries(SUJIN_TICKERS.map((t,j)=>[t,37.21+i+j]))};
      const result=buildSujinPlan(p,m);
      expect(result.remainingCash).toBeGreaterThanOrEqual(0);
      expect(result.lines.reduce((s,l)=>s+l.price*l.target,0)).toBeLessThanOrEqual(result.equity+.000001);
      expect(result.lines.find(l=>l.ticker==='SPY')?.target).toBe(1.2345);
      for(const line of result.lines) expect(line.trade).toBeGreaterThanOrEqual(-line.held);
    }
  });
  it('누락 시세를 다른 종목이나 0원으로 대신하지 않는다',()=>{
    expect(()=>buildSujinPlan(profile(),{...market,prices:{...market.prices,SPYM:NaN}})).toThrow('SPYM');
    expect(()=>buildSujinPlan(profile(),{...market,priceTimes:{...market.priceTimes,GLD:''}})).toThrow('GLD');
  });
  it('잘못된 금액·보유종목·수량을 거부한다',()=>{
    expect(()=>validateSujinProfile({...profile(),cash:-1})).toThrow();
    expect(()=>validateSujinProfile({...profile(),holdings:{...profile().holdings,QQQM:NaN}})).toThrow();
    expect(()=>validateSujinProfile({...profile(),holdings:{...profile().holdings,FAKE:1}})).toThrow();
    const p=profile();p.holdings.JEPQ=101;
    expect(()=>buildSujinPlan(p,market)).toThrow('총자산이');
  });
  it('비중 차이는 예수금을 포함한 총자산 기준이다',()=>{
    const p=profile();p.holdings.JEPQ=5;
    expect(buildSujinPlan(p,market).groups.find(g=>g.ticker==='JEPQ')).toMatchObject({current:.05,weight:.1,deviation:-.05});
  });
  it('가격과 계산 목표수량이 저장 허용 범위를 벗어나면 막는다',()=>{
    expect(()=>buildSujinPlan(profile(),{...market,prices:{...market.prices,QQQM:1e13}})).toThrow('QQQM');
    expect(()=>buildSujinPlan(profile(),{...market,prices:{...market.prices,QQQM:1e-10}})).toThrow('목표수량');
  });
  it('CSV는 한국어 헤더·BOM과 실제 계산 수량을 포함한다',()=>{
    const result=buildSujinPlan(profile(),market);
    const csv=sujinCsv(result.lines);
    expect(csv.startsWith('\uFEFF종목')).toBe(true);expect(csv).toContain('SPYM,20.0000,100,0,20,20,2000.00');
  });
});
