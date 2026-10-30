import type { Market } from './types';
import { percentagePoint } from './portfolio';

// magic-pension-rebalancer-streamlit @9fe50d9, final shared-chat configuration.
export const PENSION_PATH='pension-v2';
export const GOLD_CODES=['0072R0','411060'];
export const PENSION_ASSETS = [
  {code:'379800',name:'KODEX 미국S&P500',group:'선진국 · 미국',weight:0.12},
  {code:'379810',name:'KODEX 미국나스닥100',group:'선진국 · 미국',weight:0.12},
  {code:'294400',name:'KIWOOM 200TR',group:'신흥국 · 한국',weight:0.08},
  {code:'283580',name:'KODEX 차이나CSI300',group:'신흥국 · 중국',weight:0.08},
  {code:'453810',name:'KODEX 인도Nifty50',group:'신흥국 · 인도',weight:0.08},
  {code:'0072R0',name:'TIGER KRX금현물',group:'금',weight:0},
  {code:'411060',name:'ACE KRX금현물',group:'금',weight:0},
  {code:'148070',name:'KIWOOM 국고채10년',group:'한국 국채',weight:0.07},
  {code:'385560',name:'RISE KIS국고채30년Enhanced',group:'한국 국채',weight:0.07},
  {code:'0085P0',name:'ACE 미국10년국채액티브',group:'미국 국채',weight:0.07},
  {code:'476760',name:'ACE 미국30년국채액티브',group:'미국 국채',weight:0.07},
  {code:'469830',name:'SOL 초단기채권액티브',group:'현금성 자산',weight:0.04},
] as const;
export const PENSION_CODES=PENSION_ASSETS.map(a=>a.code);
export type PensionProfile={strategy:'magic-pension-v2';currency:'KRW';totalBalance:number;principal:number;cashBalance?:number;goldTargetWeight:number;holdings:Record<string,number>;weights:Record<string,number>;targetOverrides:Record<string,number|null>};
export type PensionMarket=Market & {source?:string;currency:'KRW';quoteMode:'intraday'|'close';marketDate:string};
export type PensionLine={ticker:string;held:number;price:number;target:number;trade:number};
// 저장 이력은 PENSION_CODES 순서 [held,price,target] × 12의 숫자 배열. 화면에서 행으로 복원한다.
export type PensionRecord={id:string;month:string;strategy:'magic-pension-v2';currency:'KRW';kind:'계산안';equity:number;principal:number;remainingCash:number;weights:Record<string,number>;lines:Pick<PensionLine,'held'|'price'|'target'>[];priceUpdatedAt:string;oldestPriceAt:string};
export const won=(value:number)=>new Intl.NumberFormat('ko-KR',{style:'currency',currency:'KRW',maximumFractionDigits:0}).format(value);
export function defaultPensionProfile():PensionProfile{return{strategy:'magic-pension-v2',currency:'KRW',totalBalance:0,principal:0,goldTargetWeight:.20,
  holdings:Object.fromEntries(PENSION_CODES.map(c=>[c,0])),weights:Object.fromEntries(PENSION_ASSETS.map(a=>[a.code,a.weight])),targetOverrides:Object.fromEntries(PENSION_CODES.map(c=>[c,null]))};}
function fixedMap(value:unknown){return !!value&&typeof value==='object'&&!Array.isArray(value)&&Object.keys(value).length===12&&Object.keys(value).every(k=>PENSION_CODES.includes(k as typeof PENSION_CODES[number]));}
export function validatePensionProfile(p:PensionProfile){
  if(!p||p.strategy!=='magic-pension-v2'||p.currency!=='KRW')throw new Error('연금 포트폴리오 설정을 확인하세요.');
  for(const n of [p.totalBalance,p.principal])if(!Number.isFinite(n)||n<0||n>1e12)throw new Error('연금 총자산과 원금을 확인하세요.');
  if(p.cashBalance!==undefined&&(!Number.isFinite(p.cashBalance)||p.cashBalance<0||p.cashBalance>1e12))throw new Error('연금 예수금을 확인하세요.');
  if(!fixedMap(p.holdings)||!fixedMap(p.weights)||!fixedMap(p.targetOverrides))throw new Error('연금 12개 종목의 입력을 확인하세요.');
  for(const c of PENSION_CODES){
    if(!Number.isInteger(p.holdings[c])||p.holdings[c]<0||p.holdings[c]>1e9)throw new Error(`${c} 보유수량은 0 이상의 정수로 입력하세요.`);
    if(!Number.isFinite(p.weights[c])||p.weights[c]<0||p.weights[c]>1)throw new Error(`${c} 목표 비중을 확인하세요.`);
    const n=p.targetOverrides[c];if(n!==null&&(!Number.isInteger(n)||n<0||n>1e9))throw new Error(`${c} 목표 보유수량은 정수로 입력하거나 비워 두세요.`);
  }
  if(!Number.isFinite(p.goldTargetWeight)||p.goldTargetWeight<0||p.goldTargetWeight>1)throw new Error('금 합산 목표 비중을 확인하세요.');
  if(GOLD_CODES.some(c=>p.weights[c]!==0))throw new Error('금은 개별 고정 비중 대신 합산 목표를 사용합니다.');
  if(Math.abs(Object.values(p.weights).reduce((s,w)=>s+w,0)+p.goldTargetWeight-1)>1e-8)throw new Error('연금 목표 비중의 합계를 100%로 맞춰 주세요.');
}
export function effectivePensionWeights(p:PensionProfile,m:PensionMarket|null) {
  const weights={...p.weights};
  const values=GOLD_CODES.map(c=>p.holdings[c]*(m?.prices[c]??0));
  const total=values.reduce((a,b)=>a+b,0);
  // 원본 규칙: 보유 금이 없을 때만 기존 시트의 초기 분배 비율을 사용한다.
  const fallback=[1730960/4499000,2768040/4499000];
  GOLD_CODES.forEach((c,i)=>weights[c]=p.goldTargetWeight*(total>0?values[i]/total:fallback[i]));
  return weights;
}
// 기존 수동 총자산은 보존하고, 예수금을 입력하면 현재 시세로 총자산을 계산한다.
export function pensionValuation(p:PensionProfile,m:PensionMarket){
  if(!m||m.currency!=='KRW')throw new Error('원화 시세가 필요합니다.');
  if(!Number.isFinite(p.totalBalance)||p.totalBalance<0||p.totalBalance>1e12||!fixedMap(p.holdings))throw new Error('연금 총자산과 보유수량을 확인하세요.');
  if(p.cashBalance!==undefined&&(!Number.isFinite(p.cashBalance)||p.cashBalance<0||p.cashBalance>1e12))throw new Error('연금 예수금을 확인하세요.');
  const rows=PENSION_CODES.map(code=>{
    const held=p.holdings[code],price=m.prices?.[code],stamp=m.priceTimes?.[code];
    if(!Number.isInteger(held)||held<0||held>1e9)throw new Error(`${code} 보유수량은 0 이상의 정수로 입력하세요.`);
    if(!Number.isFinite(price)||price<=0||price>1e12||!stamp||!Number.isFinite(Date.parse(stamp)))throw new Error(`${code}의 정상 가격·시각이 없어 계산할 수 없습니다. 시세를 갱신하세요.`);
    return {code,held,price,value:held*price};
  });
  const holdingsValue=rows.reduce((sum,row)=>sum+row.value,0);
  const explicitCash=p.cashBalance!==undefined;
  // 기존 입력이 시세 평가액보다 작으면 부족분을 부채나 손실로 간주하지 않는다.
  const assumedCash=!explicitCash&&holdingsValue>p.totalBalance;
  const cash=explicitCash?p.cashBalance!:Math.max(0,p.totalBalance-holdingsValue);
  const equity=explicitCash?holdingsValue+cash:Math.max(p.totalBalance,holdingsValue);
  if(!Number.isFinite(equity)||equity<=0||equity>1e12)throw new Error('총자산을 0 초과 1조 원 이하로 입력하세요.');
  return {rows,holdingsValue,equity,cash,explicitCash,assumedCash};
}
// 보유 비중은 매매안 입력의 유효성과 별도로 계산한다.
export function pensionAllocation(p:PensionProfile,m:PensionMarket|null){
  if(!m||!fixedMap(p.weights))return null;
  try{
    const valuation=pensionValuation(p,m);
    const rows=valuation.rows.map(row=>({...row,currentWeight:row.value/valuation.equity}));
    const goldValue=rows.filter(row=>GOLD_CODES.includes(row.code)).reduce((sum,row)=>sum+row.value,0);
    return {...valuation,rows,targetWeights:effectivePensionWeights(p,m),goldWeight:goldValue/valuation.equity,cashWeight:valuation.cash/valuation.equity};
  }catch{return null;}
}
export function pensionDifference(current:number,target:number){
  if(!Number.isFinite(current)||!Number.isFinite(target)||target<0||target>1)return '—';
  return percentagePoint(current-target);
}
export function buildPensionPlan(p:PensionProfile,m:PensionMarket){
  validatePensionProfile(p);
  const valuation=pensionValuation(p,m);
  const {equity}=valuation;
  const weights=effectivePensionWeights(p,m);
  const lines=valuation.rows.map(({code:ticker,held,price})=>{
    const calculated=equity*weights[ticker]/price;
    const target=p.targetOverrides[ticker]??Math.floor(calculated);
    if(!Number.isFinite(target)||target>1e9)throw new Error(`${ticker} 목표수량이 허용 범위를 넘었습니다.`);
    return{ticker,price,held,target,trade:target-held,calculated,allocation:equity*weights[ticker]};
  });
  const targetValue=lines.reduce((s,l)=>s+l.target*l.price,0);
  if(targetValue>equity+.0001)throw new Error('직접 지정한 목표 보유수량의 합계가 총자산을 초과합니다. 수량이나 총자산을 조정하세요.');
  return{lines,weights,...valuation,targetValue,remainingCash:Math.max(0,equity-targetValue),
    profit:p.principal>0?equity-p.principal:null,profitRate:p.principal>0?equity/p.principal-1:null,
    oldestPriceAt:lines.map(l=>m.priceTimes[l.ticker]).sort((a,b)=>Date.parse(a)-Date.parse(b))[0]};
}
export function pensionCsv(plan:ReturnType<typeof buildPensionPlan>,p:PensionProfile){
  const rows=[['종목코드','상품(원본 표기)','목표비중 %','현재가격 KRW','목표배분액 KRW','계산된수량','현재보유수량','최종목표보유수량','매매수량','최종목표금액 KRW'],
    ...plan.lines.map(l=>[l.ticker,PENSION_ASSETS.find(a=>a.code===l.ticker)!.name,(plan.weights[l.ticker]*100).toFixed(2),l.price,l.allocation,l.calculated,l.held,l.target,l.trade,l.target*l.price])];
  return '\uFEFF'+rows.map(r=>r.map(v=>`"${String(v).replaceAll('"','""')}"`).join(',')).join('\r\n');
}
