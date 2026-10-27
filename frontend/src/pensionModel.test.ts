import { describe, expect, it } from 'vitest';
import { buildPensionPlan, defaultPensionProfile, PENSION_ASSETS, PENSION_CODES, pensionCsv, validatePensionProfile, type PensionMarket } from './pensionModel';
const market:PensionMarket={currency:'KRW',quoteMode:'close',marketDate:'2026-10-26',updatedAt:'2026-10-26T07:00:00Z',prices:Object.fromEntries(PENSION_CODES.map(c=>[c,10000])),priceTimes:Object.fromEntries(PENSION_CODES.map(c=>[c,'2026-10-26T06:30:00Z']))};
const profile=()=>({...defaultPensionProfile(),totalBalance:1000000,principal:800000});
describe('원본 연금 포트폴리오 계산',()=>{
  it('원본 12종목·비중과 백만원 배분을 보존한다',()=>{
    expect(PENSION_ASSETS.map(a=>a.weight)).toEqual([.12,.12,.08,.08,.08,0,0,.07,.07,.07,.07,.04]);
    const p=profile(),plan=buildPensionPlan(p,market);
    expect(plan.lines.map(l=>l.target)).toEqual([12,12,8,8,8,7,12,7,7,7,7,4]);
    expect(plan.remainingCash).toBe(10000);expect(plan.profit).toBe(200000);expect(plan.profitRate).toBe(.25);
  });
  it('빈 목표는 자동 계산하며 직접 입력한 0은 전량 매도이다',()=>{
    const p=profile();p.holdings['379800']=3;p.targetOverrides['379800']=0;
    let plan=buildPensionPlan(p,market);expect(plan.lines[0].trade).toBe(-3);expect(plan.remainingCash).toBe(130000);
    p.targetOverrides['379800']=null;plan=buildPensionPlan(p,market);expect(plan.lines[0].target).toBe(12);expect(plan.lines[0].trade).toBe(9);
  });
  it('가격이나 총자산 변경 후에도 수동 최종 목표를 보존한다',()=>{
    const p=profile();p.targetOverrides['379800']=1;p.totalBalance=2000000;
    expect(buildPensionPlan(p,market).lines[0].target).toBe(1);expect(buildPensionPlan(p,market).lines[1].target).toBe(24);
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
    p.weights['379800']=.12;p.holdings['379800']=.5;expect(()=>validatePensionProfile(p)).toThrow('정수');
    p.holdings['379800']=0;p.targetOverrides['379800']=.1;expect(()=>validatePensionProfile(p)).toThrow('정수');
  });
  it('CSV는 원화·최종 목표·매매 수량을 별도 열로 내보낸다',()=>{
    const p=profile(),csv=pensionCsv(buildPensionPlan(p,market),p);expect(csv.startsWith('\uFEFF')).toBe(true);expect(csv).toContain('최종목표보유수량');expect(csv).toContain('KRW');expect(csv.split('\r\n')).toHaveLength(13);
  });
});

it('금 합산 목표는 고정하되 현재 평가액에 따라 두 ETF를 배분한다',()=>{
 const p=profile();p.holdings['0072R0']=3;p.holdings['411060']=7;
 const plan=buildPensionPlan(p,market);
 expect(plan.weights['0072R0']).toBeCloseTo(.06);expect(plan.weights['411060']).toBeCloseTo(.14);
 expect(plan.weights['0072R0']+plan.weights['411060']).toBeCloseTo(.20);
 const changed=buildPensionPlan(p,{...market,prices:{...market.prices,'0072R0':20000}});
 expect(changed.weights['0072R0']).toBeCloseTo(.2*6/13);
 expect(changed.weights['411060']).toBeCloseTo(.2*7/13);
 expect(changed.weights['0072R0']+changed.weights['411060']).toBeCloseTo(.20);
});
it('금 한 종목만 보유하면 합산 목표를 그 종목에 배분한다',()=>{
 const p=profile();p.holdings['411060']=5;
 const plan=buildPensionPlan(p,market);
 expect(plan.weights['0072R0']).toBe(0);expect(plan.weights['411060']).toBe(.2);
});
it('합산 금 목표를 저장하며 개별 고정 목표는 허용하지 않는다',()=>{
 const p=profile();p.goldTargetWeight=.25;p.weights['379800']=.07;
 expect(()=>validatePensionProfile(p)).not.toThrow();
 p.weights['411060']=.01;expect(()=>validatePensionProfile(p)).toThrow('합산 목표');
});
