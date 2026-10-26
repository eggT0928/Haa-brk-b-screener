import type { Market } from './types';

// magic_pension_portfolio.py의 실제 PORTFOLIO_CONFIG 기준. 상품명은 원본 표기이다.
export const PENSION_ASSETS = [
  {code:'379800',name:'KODEX 미국 S&P500TR',group:'선진국',weight:.24},
  {code:'294400',name:'KOSEF 200TR',group:'신흥국 · 한국',weight:.08},
  {code:'283580',name:'KODEX 차이나CSI300',group:'신흥국 · 중국',weight:.08},
  {code:'453810',name:'KODEX 인도 NIFTY50',group:'신흥국 · 인도',weight:.08},
  {code:'411060',name:'ACE KRX금현물',group:'금',weight:.19},
  {code:'385560',name:'RISE KIS 국고채30년 Enhanced',group:'한국 국채',weight:.14},
  {code:'308620',name:'KODEX 미국채 10년선물',group:'미국 국채',weight:.07},
  {code:'453850',name:'ACE 미국30년 국채액티브(H)',group:'미국 국채',weight:.07},
  {code:'449170',name:'TIGER KOFR금리액티브(합성)',group:'현금성 자산',weight:.05},
] as const;
export const PENSION_CODES=PENSION_ASSETS.map(a=>a.code);
export type PensionProfile={strategy:'magic-pension-v1';currency:'KRW';totalBalance:number;principal:number;holdings:Record<string,number>;weights:Record<string,number>;targetOverrides:Record<string,number|null>};
export type PensionMarket=Market & {currency:'KRW';quoteMode:'intraday'|'close';marketDate:string};
export type PensionLine={ticker:string;held:number;price:number;target:number;trade:number};
// 이력은 PENSION_CODES 고정순서로 3개 숫자만 저장한다. 매매량은 목표-보유로 복원한다.
export type PensionRecord={id:string;month:string;strategy:'magic-pension-v1';currency:'KRW';kind:'계산안';equity:number;principal:number;remainingCash:number;weights:Record<string,number>;lines:Pick<PensionLine,'held'|'price'|'target'>[];priceUpdatedAt:string;oldestPriceAt:string};
export const won=(value:number)=>new Intl.NumberFormat('ko-KR',{style:'currency',currency:'KRW',maximumFractionDigits:0}).format(value);
export function defaultPensionProfile():PensionProfile{return{strategy:'magic-pension-v1',currency:'KRW',totalBalance:0,principal:0,
  holdings:Object.fromEntries(PENSION_CODES.map(c=>[c,0])),weights:Object.fromEntries(PENSION_ASSETS.map(a=>[a.code,a.weight])),targetOverrides:Object.fromEntries(PENSION_CODES.map(c=>[c,null]))};}
function fixedMap(value:unknown){return !!value&&typeof value==='object'&&!Array.isArray(value)&&Object.keys(value).length===9&&Object.keys(value).every(k=>PENSION_CODES.includes(k as typeof PENSION_CODES[number]));}
export function validatePensionProfile(p:PensionProfile){
  if(!p||p.strategy!=='magic-pension-v1'||p.currency!=='KRW')throw new Error('연금 포트폴리오 설정을 확인하세요.');
  for(const n of [p.totalBalance,p.principal])if(!Number.isFinite(n)||n<0||n>1e12)throw new Error('연금 총자산과 원금을 확인하세요.');
  if(!fixedMap(p.holdings)||!fixedMap(p.weights)||!fixedMap(p.targetOverrides))throw new Error('연금 9개 종목의 입력을 확인하세요.');
  for(const c of PENSION_CODES){
    if(!Number.isInteger(p.holdings[c])||p.holdings[c]<0||p.holdings[c]>1e9)throw new Error(`${c} 보유수량은 0 이상의 정수로 입력하세요.`);
    if(!Number.isFinite(p.weights[c])||p.weights[c]<0||p.weights[c]>1)throw new Error(`${c} 목표 비중을 확인하세요.`);
    const n=p.targetOverrides[c];if(n!==null&&(!Number.isInteger(n)||n<0||n>1e9))throw new Error(`${c} 목표 보유수량은 정수로 입력하거나 비워 두세요.`);
  }
  if(Math.abs(Object.values(p.weights).reduce((s,w)=>s+w,0)-1)>1e-8)throw new Error('연금 목표 비중의 합계를 100%로 맞춰 주세요.');
}
export function buildPensionPlan(p:PensionProfile,m:PensionMarket){
  validatePensionProfile(p);
  if(p.totalBalance<=0)throw new Error('계산할 연금 총자산을 입력하세요.');
  if(m.currency!=='KRW')throw new Error('원화 시세가 필요합니다.');
  const lines=PENSION_CODES.map(ticker=>{
    const price=m.prices[ticker],stamp=m.priceTimes[ticker];
    if(!Number.isFinite(price)||price<=0||price>1e12||!stamp||!Number.isFinite(Date.parse(stamp)))throw new Error(`${ticker}의 정상 가격·시각이 없어 계산할 수 없습니다. 시세를 갱신하세요.`);
    const calculated=p.totalBalance*p.weights[ticker]/price;
    const target=p.targetOverrides[ticker]??Math.floor(calculated);
    if(!Number.isFinite(target)||target>1e9)throw new Error(`${ticker} 목표수량이 허용 범위를 넘었습니다.`);
    return{ticker,price,held:p.holdings[ticker],target,trade:target-p.holdings[ticker],calculated,allocation:p.totalBalance*p.weights[ticker]};
  });
  const targetValue=lines.reduce((s,l)=>s+l.target*l.price,0);
  if(targetValue>p.totalBalance+.0001)throw new Error('직접 지정한 목표 보유수량의 합계가 총자산을 초과합니다. 수량이나 총자산을 조정하세요.');
  const holdingsValue=lines.reduce((s,l)=>s+l.held*l.price,0);
  if(holdingsValue>p.totalBalance+.0001)throw new Error('입력 총자산이 보유 ETF 평가액보다 작습니다. 평가금액과 예수금을 포함한 총액을 확인하세요.');
  return{lines,equity:p.totalBalance,holdingsValue,targetValue,remainingCash:Math.max(0,p.totalBalance-targetValue),
    profit:p.principal>0?p.totalBalance-p.principal:null,profitRate:p.principal>0?p.totalBalance/p.principal-1:null,
    oldestPriceAt:lines.map(l=>m.priceTimes[l.ticker]).sort((a,b)=>Date.parse(a)-Date.parse(b))[0]};
}
export function pensionCsv(plan:ReturnType<typeof buildPensionPlan>,p:PensionProfile){
  const rows=[['종목코드','상품(원본 표기)','목표비중 %','현재가격 KRW','목표배분액 KRW','계산된수량','현재보유수량','최종목표보유수량','매매수량','최종목표금액 KRW'],
    ...plan.lines.map(l=>[l.ticker,PENSION_ASSETS.find(a=>a.code===l.ticker)!.name,(p.weights[l.ticker]*100).toFixed(2),l.price,l.allocation,l.calculated,l.held,l.target,l.trade,l.target*l.price])];
  return '\uFEFF'+rows.map(r=>r.map(v=>`"${String(v).replaceAll('"','""')}"`).join(',')).join('\r\n');
}
