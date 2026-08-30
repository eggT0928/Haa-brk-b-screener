import { useEffect, useRef, useState, type ReactNode } from 'react';
import type { User } from 'firebase/auth';
import { addDoc, collection, doc, limit, onSnapshot, orderBy, query, serverTimestamp, setDoc } from 'firebase/firestore';
import { apiHeaders, db, demo } from './firebase';
import { money, percent, timeLabel } from './portfolio';
import { buildSujinPlan, defaultSujinProfile, SUJIN_TICKERS, SUJIN_WEIGHTS, sujinCsv, validateSujinProfile, type SujinProfile, type SujinRecord } from './sujinModel';
import type { Backtest, Market, UpdateStatus } from './types';
import './sujin.css';

const explain = (error: unknown) => error instanceof Error ? error.message : '요청을 처리하지 못했습니다.';
const sampleMarket: Market = {prices:Object.fromEntries(SUJIN_TICKERS.map(t=>[t,t==='SPY'?600:100])),
  priceTimes:Object.fromEntries(SUJIN_TICKERS.map(t=>[t,'2026-08-28T20:00:00Z'])),updatedAt:'2026-08-30T05:42:00Z'};

export function SujinPortfolio({ user, now, renderBacktest }: {
  user: User | null; now: number; renderBacktest: (result: Backtest) => ReactNode;
}) {
  const [profile,setProfile] = useState<SujinProfile>(()=>({...defaultSujinProfile(),totalBalance:demo?10000:0}));
  const [market,setMarket] = useState<Market | null>(demo?sampleMarket:null);
  const [status,setStatus] = useState<UpdateStatus | null>(null);
  const [tab,setTab] = useState('overview');
  const [records,setRecords] = useState<SujinRecord[]>([]);
  const [result,setResult] = useState<Backtest | null>(null);
  const [initial,setInitial] = useState(10000);
  const [busy,setBusy] = useState('');
  const [error,setError] = useState('');
  const [message,setMessage] = useState('');
  const [acknowledged,setAcknowledged] = useState(false);
  const [profileReady,setProfileReady] = useState(demo);
  const dirty = useRef(false);
  const alive = useRef(true);
  const activeRequest = useRef<AbortController | null>(null);
  useEffect(()=>{
    alive.current = true;
    if (demo || !db || !user) return ()=>{ alive.current=false; activeRequest.current?.abort(); };
    const fail = (e: unknown) => setError(`수진 데이터를 불러오지 못했습니다: ${explain(e)}`);
    const stop = [
      onSnapshot(doc(db,'users',user.uid,'portfolios','sujin'),s=>{
        if (!dirty.current) {
          const next = s.exists()?s.data() as SujinProfile:defaultSujinProfile();
          try { validateSujinProfile(next); setProfile(next); setProfileReady(true); }
          catch(e) { setProfileReady(false); fail(e); }
        }
      },fail),
      onSnapshot(doc(db,'market','sujin'),s=>{setMarket(s.exists()?s.data() as Market:null);setAcknowledged(false);},fail),
      onSnapshot(doc(db,'status','sujinQuotes'),s=>setStatus(s.exists()?s.data() as UpdateStatus:null),fail),
      onSnapshot(query(collection(db,'users',user.uid,'portfolios','sujin','rebalances'),orderBy('createdAt','desc'),limit(24)),
        s=>setRecords(s.docs.map(d=>({...d.data(),id:d.id} as SujinRecord))),fail),
    ];
    return ()=>{alive.current=false;activeRequest.current?.abort();stop.forEach(fn=>fn());};
  },[user?.uid]);
  const edit = (next: Partial<SujinProfile>) => {dirty.current=true;setProfile(p=>({...p,...next}));setMessage('');};
  let plan: ReturnType<typeof buildSujinPlan> | null = null, planError='';
  if (market && profileReady) {try {plan=buildSujinPlan(profile,market);} catch(e) {planError=explain(e);}}
  const stale = !market || status?.ok === false || SUJIN_TICKERS.some(t=>!market.priceTimes[t]
    || !Number.isFinite(Date.parse(market.priceTimes[t])) || now-Date.parse(market.priceTimes[t])>30*60000
    || Date.parse(market.priceTimes[t])-now>60000);
  async function request(route: 'refresh' | 'backtest', data: object) {
    const controller = new AbortController();
    activeRequest.current = controller;
    const timer = setTimeout(()=>controller.abort(),60000);
    try {
      const res = await fetch(`/api/sujin/${route}`,{method:'POST',headers:{...await apiHeaders(),'Content-Type':'application/json'},
        body:JSON.stringify(data),signal:controller.signal});
      if (!res.headers.get('content-type')?.includes('application/json')) throw new Error('수진 서버에 연결할 수 없습니다. 배포 상태를 확인하세요.');
      const payload = await res.json();
      if (!res.ok) throw new Error(payload.error ?? '요청에 실패했습니다.');
      return payload;
    } catch(e) {
      if (controller.signal.aborted) throw new Error('조회 시간이 길어 중단했습니다. 직전 성공값을 유지하며 잠시 후 다시 시도할 수 있습니다.');
      throw e;
    } finally {clearTimeout(timer);if(activeRequest.current===controller) activeRequest.current=null;}
  }
  async function refreshPrices() {
    if (demo || busy) return;
    setBusy('refresh');setError('');setMessage('');
    try {
      const payload = await request('refresh',{});
      if (alive.current) {setMarket(payload.market);setAcknowledged(false);setMessage(payload.message);}
    } catch(e) {if(alive.current) setError(explain(e));}
    finally {if(alive.current) setBusy('');}
  }
  async function saveProfile() {
    if (!db || !user || demo || !profileReady || busy) return;
    setBusy('save');setError('');setMessage('');
    try {
      validateSujinProfile(profile);
      await setDoc(doc(db,'users',user.uid,'portfolios','sujin'),{...profile,updatedAt:serverTimestamp()});
      if(alive.current) {dirty.current=false;setMessage('수진 보유수량과 설정을 저장했습니다. 현근 데이터는 변경하지 않았습니다.');}
    } catch(e) {if(alive.current) setError(explain(e));}
    finally {if(alive.current) setBusy('');}
  }
  async function savePlan() {
    if (!db || !user || demo || !plan || !market || busy || (stale&&!acknowledged)) return;
    setBusy('record');setError('');setMessage('');
    try {
      const month = new Intl.DateTimeFormat('sv-SE',{timeZone:'America/New_York',year:'numeric',month:'2-digit'}).format(new Date(now));
      await addDoc(collection(db,'users',user.uid,'portfolios','sujin','rebalances'),{
        createdAt:serverTimestamp(),month,strategy:profile.strategy,kind:'계산안',equity:plan.equity,
        remainingCash:plan.remainingCash,priceUpdatedAt:market.updatedAt,oldestPriceAt:plan.oldestPriceAt,
        quantityMode:profile.quantityMode,lines:plan.lines.map(({ticker,held,price,target,trade})=>({ticker,held,price,target,trade})),
      });
      if(alive.current) setMessage('수진 계산안을 저장했습니다. 주문 실행이나 보유수량 변경은 하지 않았습니다.');
    } catch(e) {if(alive.current) setError(explain(e));}
    finally {if(alive.current) setBusy('');}
  }
  async function runBacktest() {
    if (demo || busy) return;
    setBusy('backtest');setError('');setMessage('');setResult(null);
    try {
      if (!Number.isFinite(initial) || initial<=0 || initial>1e12) throw new Error('초기 금액을 확인하세요.');
      if (profile.backtestStart && profile.backtestEnd && profile.backtestStart>profile.backtestEnd) throw new Error('백테스트 기간을 확인하세요.');
      const payload = await request('backtest',{initial,...(profile.backtestStart?{start:profile.backtestStart}:{}),...(profile.backtestEnd?{end:profile.backtestEnd}:{})});
      if(alive.current) setResult(payload);
    } catch(e) {if(alive.current) setError(explain(e));}
    finally {if(alive.current) setBusy('');}
  }
  function downloadCsv() {
    if (!plan) return;
    const url=URL.createObjectURL(new Blob([sujinCsv(plan.lines)],{type:'text/csv;charset=utf-8'}));
    const link=document.createElement('a');link.href=url;link.download='수진_리밸런싱_계산안.csv';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  return <div className="sujin-workspace">
    <div className="sujin-topline"><div><span className="eyebrow">수진 · 고정비중 포트폴리오</span><h2>목표 비중으로, 차분하게.</h2>
      <p>HAA 자동 갱신을 함께 사용합니다. 별도 예약 작업 없이, 필요할 때 수동 갱신도 가능합니다.</p></div>
      <button disabled={!!busy||demo} onClick={refreshPrices}>{busy==='refresh'?'시세 조회 중…':'시세 수동 갱신'}</button></div>
    <p className="micro">최근 성공 갱신 {timeLabel(market?.updatedAt)} · {dirty.current?'저장하지 않은 수진 입력이 있습니다.':'보유수량은 수진 전용 공간에 저장됩니다.'}</p>
    {error&&<p className="error" role="alert">{error}</p>}{message&&<p className="success" role="status">{message}</p>}
    {status?.ok===false&&<p className="notice">수진 시세 갱신에 실패해 직전 성공값을 유지합니다. 현근 HAA 상태와는 별개입니다.</p>}
    <nav aria-label="수진 포트폴리오 메뉴">{[['overview','리밸런싱'],['backtest','장기 백테스트'],['history','저장 이력']].map(([id,label])=><button key={id} className={tab===id?'active':''} onClick={()=>setTab(id)}>{label}</button>)}</nav>
    {tab==='overview'&&<>
      <section className="panel sujin-target"><div className="section-title"><h2>매월 유지할 목표 비중</h2><span>모멘텀 신호 없음 · 월말 리밸런싱</span></div>
        <div className="sujin-weights">{Object.entries(SUJIN_WEIGHTS).map(([ticker,weight])=><div key={ticker}><strong>{ticker}</strong><span>{percent(weight)}</span></div>)}</div>
        <p className="micro">SPY와 SPYM 합산 20%. 기존 SPY는 매수·매도 없이 유지하고 SPYM으로 조정합니다.</p></section>
      <div className="workspace-grid">
        <section className="panel"><div className="section-title"><h2>수진 보유자산</h2><span>USD 기준 · 현근과 분리</span></div>
          {!profileReady&&<p className="notice">수진 저장 데이터를 불러오는 중입니다. 연결 확인 전에는 저장할 수 없습니다.</p>}
          <fieldset disabled={!profileReady||!!busy} className="sujin-fields">
            <label>총자산 기준<select value={profile.totalMode} onChange={e=>edit({totalMode:e.target.value as SujinProfile['totalMode']})}><option value="manual">총자산 직접 입력 · 기존 방식</option><option value="holdings">보유주식 평가액 + 현금</option></select></label>
            {profile.totalMode==='manual'?<label>총자산 · 평가금액 + 예수금 (USD)<input type="number" min="0" max="1000000000000" step="0.01" value={Number.isNaN(profile.totalBalance)?'':profile.totalBalance} onChange={e=>edit({totalBalance:e.target.valueAsNumber})}/></label>
              :<label>현금 잔액 (USD)<input type="number" min="0" max="1000000000000" step="0.01" value={Number.isNaN(profile.cash)?'':profile.cash} onChange={e=>edit({cash:e.target.valueAsNumber})}/></label>}
            <label>매수 목표 수량 단위<select value={profile.quantityMode} onChange={e=>edit({quantityMode:e.target.value as SujinProfile['quantityMode']})}><option value="fractional">소수점 0.01주</option><option value="whole">정수 1주</option></select></label>
            <div className="holdings-grid">{SUJIN_TICKERS.map(t=><label key={t}>{t}<input aria-label={`수진 ${t} 보유수량`} type="number" min="0" max="1000000000" step="any" value={Number.isNaN(profile.holdings[t])?'':profile.holdings[t]} onChange={e=>edit({holdings:{...profile.holdings,[t]:e.target.valueAsNumber}})}/></label>)}</div>
            <p className="micro">목표 수량은 선택한 단위로 내림합니다. SPY는 소수점 보유량도 그대로 유지하며 다른 종목의 기존 소수점 잔량 매도는 발생할 수 있습니다.</p>
          </fieldset>
          <button className="primary full" disabled={!!busy||demo||!profileReady} onClick={saveProfile}>수진 보유수량·설정 저장</button>
        </section>
        <section className="panel"><div className="section-title"><h2>수진 리밸런싱 계산안</h2><span>현재 가격 기준 · 주문 아님</span></div>
          {!market&&<p className="empty">아직 수진 시세가 없습니다. 자동 갱신을 기다리거나 ‘시세 수동 갱신’을 눌러 주세요.</p>}
          {planError&&<p className="notice">{planError}</p>}
          {plan&&<>
            <div className="balance"><span>계산 기준 총자산</span><strong>{money(plan.equity)}</strong><small>리밸런싱 후 예상 현금 {money(plan.remainingCash)}</small></div>
            <p className="micro">가장 오래된 사용 시세 {timeLabel(plan.oldestPriceAt)}</p>
            {stale&&<p className="notice">장외·지연 시세 또는 직전 성공값입니다. 주문 전 증권사 가격을 확인하세요.</p>}
            <div className="table-scroll"><table><thead><tr><th>종목</th><th>보유 → 목표</th><th>매매 수량</th><th>가격 USD</th></tr></thead><tbody>{plan.lines.map(l=><tr key={l.ticker}><th>{l.ticker}{l.ticker==='SPY'?' · 유지':''}</th><td>{l.held} → {l.target}</td><td className={l.trade<0?'negative':'positive'}>{l.trade>0?'+':''}{l.trade}</td><td title={timeLabel(market?.priceTimes[l.ticker])}>{money(l.price)}</td></tr>)}</tbody></table></div>
            <p className="micro">매도 후 매수하는 계산입니다. 수수료·세금·체결가격 차이는 제외합니다. 입력 총자산에는 예수금과 추가 투입할 금액이 포함되어야 합니다.</p>
            {stale&&<label className="check"><input type="checkbox" checked={acknowledged} onChange={e=>setAcknowledged(e.target.checked)}/>수진 계산안이 지연 또는 직전 성공 시세 기준임을 확인했습니다.</label>}
            <div className="sujin-actions"><button disabled={!!busy||demo||(stale&&!acknowledged)} onClick={savePlan}>수진 계산안 저장 · 주문 아님</button><button onClick={downloadCsv}>CSV 다운로드</button></div>
          </>}
        </section>
      </div>
      {plan&&<section className="panel sujin-comparison"><div className="section-title"><h2>현재 비중과 목표의 차이</h2><span>총자산·예수금 포함 기준</span></div>
        <div className="table-scroll"><table><thead><tr><th>자산군</th><th>현재</th><th>목표</th><th>차이 (%p)</th></tr></thead><tbody>{[...plan.groups].sort((a,b)=>Math.abs(b.deviation)-Math.abs(a.deviation)).map(g=><tr key={g.ticker}><th>{g.ticker}</th><td>{percent(g.current)}</td><td>{percent(g.weight)}</td><td className={g.deviation<0?'negative':'positive'}>{g.deviation>0?'+':''}{(g.deviation*100).toFixed(2)}</td></tr>)}</tbody></table></div>
      </section>}
    </>}
    {tab==='backtest'&&<section className="panel"><div className="section-title"><h2>수진 장기 백테스트</h2><span>고정비중 월별 리밸런싱</span></div>
      <p>수진 원본처럼 SPY 10% + SPYM 10%, PDBC 자체를 사용합니다. 모든 ETF의 공통 이력이 있는 완료된 월말부터 계산하며, HAA의 DBC 프록시는 적용하지 않습니다.</p>
      <div className="date-controls"><label>시작일<input type="date" value={profile.backtestStart} onChange={e=>edit({backtestStart:e.target.value})}/></label><label>종료일<input type="date" value={profile.backtestEnd} onChange={e=>edit({backtestEnd:e.target.value})}/></label><label>초기 금액 (USD)<input type="number" min="1" max="1000000000000" value={Number.isNaN(initial)?'':initial} onChange={e=>setInitial(e.target.valueAsNumber)}/></label><button onClick={()=>edit({backtestStart:'',backtestEnd:''})}>최대 기간</button><button className="primary" disabled={!!busy||demo||!profileReady} onClick={runBacktest}>{busy==='backtest'?'계산 중…':'수진 백테스트 실행'}</button></div>
      <p className="micro">백테스트 실행 시 필요한 이력을 조회하고 재사용합니다. 수진 원본의 이론 비중이며 실제 SPY 고정 보유, 수량 내림, 수수료·세금은 반영하지 않습니다. ^IRX가 없으면 임의 2.5%를 대입하지 않습니다.</p>
      {result?renderBacktest(result):<p className="empty">기간을 선택하거나 비워 두고 최대 가용 기간을 확인하세요.</p>}
    </section>}
    {tab==='history'&&<section className="panel"><h2>수진 리밸런싱 계산안</h2><p className="micro">최근 24건 · 월 표시는 저장 시점의 미국 동부 기준 월입니다. 저장은 실제 주문·월말 확정 체결을 의미하지 않습니다.</p>
      {records.length?records.map(r=><details key={r.id}><summary>{r.month} · {money(r.equity)} · {r.kind}</summary><p className="micro">가격 갱신 {timeLabel(r.priceUpdatedAt)}<br/>사용 시세 {timeLabel(r.oldestPriceAt)} · 예상 현금 {money(r.remainingCash)}</p><div className="table-scroll"><table><thead><tr><th>종목</th><th>보유 → 목표</th><th>매매</th><th>가격</th></tr></thead><tbody>{r.lines.map(l=><tr key={l.ticker}><th>{l.ticker}</th><td>{l.held} → {l.target}</td><td>{l.trade}</td><td>{money(l.price)}</td></tr>)}</tbody></table></div></details>):<p className="empty">아직 저장된 수진 계산안이 없습니다. 기존 Streamlit 이력은 자동으로 가져오지 않습니다.</p>}
    </section>}
  </div>;
}
