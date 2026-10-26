import { describe, expect, it } from 'vitest';
import { buildPensionPlan, defaultPensionProfile, PENSION_ASSETS, PENSION_CODES, pensionCsv, validatePensionProfile, type PensionMarket } from './pensionModel';
const market:PensionMarket={currency:'KRW',quoteMode:'close',marketDate:'2026-10-26',updatedAt:'2026-10-26T07:00:00Z',prices:Object.fromEntries(PENSION_CODES.map(c=>[c,10000])),priceTimes:Object.fromEntries(PENSION_CODES.map(c=>[c,'2026-10-26T06:30:00Z']))};
const profile=()=>({...defaultPensionProfile(),totalBalance:1000000,principal:800000});
describe('원본 연금 포트폴리오 계산',()=>{
  it('원본 9종목·비중과 백만원 배분을 보존한다',()=>{
    expect(PENSION_ASSETS.map(a=>a.weight)).toEqual([.24,.08,.08,.08,.19,.14,.07,.07,.05]);
    const p=profile(),plan=buildPensionPlan(p,market);
    expect(plan.lines.map(l=>l.target)).toEqual([24,8,8,8,19,14,7,7,5]);
    expect(plan.remainingCash).toBe(0);expect(plan.profit).toBe(200000);expect(plan.profitRate).toBe(.25);
  });
  it('빈 목표는 자동 계산하며 직접 입력한 0은 전량 매도이다',()=>{
    const p=profile();p.holdings['379800']=3;p.targetOverrides['379800']=0;
    let plan=buildPensionPlan(p,market);expect(plan.lines[0].trade).toBe(-3);expect(plan.remainingCash).toBe(240000);
    p.targetOverrides['379800']=null;plan=buildPensionPlan(p,market);expect(plan.lines[0].target).toBe(24);expect(plan.lines[0].trade).toBe(21);
  });
  it('가격이나 총자산 변경 후에도 수동 최종 목표를 보존한다',()=>{
    const p=profile();p.targetOverrides['379800']=1;p.totalBalance=2000000;
    expect(buildPensionPlan(p,market).lines[0].target).toBe(1);expect(buildPensionPlan(p,market).lines[1].target).toBe(16);
  });
  it('원금 미입력과 예수금을 구분한다',()=>{
    const p=profile();p.principal=0;p.holdings['379800']=2;
    const plan=buildPensionPlan(p,market);expect(plan.profitRate).toBeNull();expect(plan.profit).toBeNull();expect(plan.holdingsValue).toBe(20000);
  });
  it('수동 목표 초과 지출과 음수 예수금 입력을 차단한다',()=>{
    const p=profile();p.targetOverrides['379800']=100;
    expect(()=>buildPensionPlan(p,market)).toThrow('총자산을 초과');
    p.targetOverrides['379800']=null;p.holdings['379800']=101;
    expect(()=>buildPensionPlan(p,market)).toThrow('보유 ETF 평가액보다');
  });
  it.each([0,-1,NaN,Infinity])('누락·잘못된 가격 %s을 전량 매도로 대입하지 않는다',n=>{
    expect(()=>buildPensionPlan(profile(),{...market,prices:{...market.prices,'411060':n}})).toThrow('411060');
  });
  it('KRW 외 통화와 가격 시각 누락을 차단한다',()=>{
    expect(()=>buildPensionPlan(profile(),{...market,currency:'USD'} as unknown as PensionMarket)).toThrow('원화');
    expect(()=>buildPensionPlan(profile(),{...market,priceTimes:{...market.priceTimes,'379800':''}})).toThrow('379800');
  });
  it('비중 합계와 KRX 정수 수량을 검증한다',()=>{
    const p=profile();p.weights['379800']=.25;expect(()=>validatePensionProfile(p)).toThrow('100%');
    p.weights['379800']=.24;p.holdings['379800']=.5;expect(()=>validatePensionProfile(p)).toThrow('정수');
    p.holdings['379800']=0;p.targetOverrides['379800']=.1;expect(()=>validatePensionProfile(p)).toThrow('정수');
  });
  it('CSV는 원화·최종 목표·매매 수량을 별도 열로 내보낸다',()=>{
    const p=profile(),csv=pensionCsv(buildPensionPlan(p,market),p);expect(csv.startsWith('\uFEFF')).toBe(true);expect(csv).toContain('최종목표보유수량');expect(csv).toContain('KRW');expect(csv.split('\r\n')).toHaveLength(10);
  });
});
