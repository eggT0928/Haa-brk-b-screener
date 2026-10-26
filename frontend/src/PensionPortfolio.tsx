import { useEffect, useRef, useState } from 'react';
import type { User } from 'firebase/auth';
import { addDoc, collection, doc, limit, onSnapshot, orderBy, query, serverTimestamp, setDoc } from 'firebase/firestore';
import { apiHeaders, db, demo } from './firebase';
import { percent, timeLabel } from './portfolio';
import { buildPensionPlan, defaultPensionProfile, PENSION_ASSETS, PENSION_CODES, pensionCsv, validatePensionProfile, won, type PensionMarket, type PensionProfile, type PensionRecord } from './pensionModel';
import type { UpdateStatus } from './types';
import './pension.css';

const explain=(e:unknown)=>e instanceof Error?e.message:'연금 요청을 처리하지 못했습니다.';
const sample:PensionMarket={currency:'KRW',quoteMode:'close',marketDate:'2026-10-23',updatedAt:'2026-10-23T07:00:00Z',
  prices:Object.fromEntries(PENSION_CODES.map(c=>[c,10000])),priceTimes:Object.fromEntries(PENSION_CODES.map(c=>[c,'2026-10-23T06:30:00Z']))};

export function PensionPortfolio({user,ownerUid,now}:{user:User|null;ownerUid?:string;now:number}){
  const [profile,setProfile]=useState<PensionProfile>(()=>({...defaultPensionProfile(),totalBalance:demo?10000000:0}));
  const [market,setMarket]=useState<PensionMarket|null>(demo?sample:null),[status,setStatus]=useState<UpdateStatus|null>(null);
  const [ready,setReady]=useState(demo),[busy,setBusy]=useState(''),[error,setError]=useState(''),[message,setMessage]=useState('');
  const [tab,setTab]=useState('overview'),[records,setRecords]=useState<PensionRecord[]>([]),[ack,setAck]=useState(false);
  const dirty=useRef(false),alive=useRef(true),activeRequest=useRef<AbortController|null>(null);
  useEffect(()=>{
    alive.current=true;
    if(demo||!db||!user||!ownerUid)return()=>{alive.current=false;activeRequest.current?.abort();};
    const fail=(e:unknown)=>{if(alive.current){setReady(false);setError(`연금 저장 데이터 연결 오류: ${explain(e)}`);}};
    const stops=[
      onSnapshot(doc(db,'users',ownerUid,'portfolios','pension'),s=>{
        if(!alive.current||dirty.current)return;
        const next=s.exists()?s.data() as PensionProfile:defaultPensionProfile();
        try{validatePensionProfile(next);setProfile(next);setReady(true);}catch(e){fail(e);}
      },fail),
      onSnapshot(doc(db,'market','pension'),s=>{if(alive.current){setMarket(s.exists()?s.data() as PensionMarket:null);setAck(false);}},e=>{if(alive.current)setError(explain(e));}),
      onSnapshot(doc(db,'status','pensionQuotes'),s=>{if(alive.current){setStatus(s.exists()?s.data() as UpdateStatus:null);setAck(false);}},e=>{if(alive.current)setError(explain(e));}),
      onSnapshot(query(collection(db,'users',ownerUid,'portfolios','pension','rebalances'),orderBy('createdAt','desc'),limit(24)),s=>{if(alive.current)setRecords(s.docs.map(d=>({...d.data(),id:d.id} as PensionRecord)));},e=>{if(alive.current)setError(explain(e));}),
    ];
    return()=>{alive.current=false;activeRequest.current?.abort();stops.forEach(stop=>stop());};
  },[user?.uid,ownerUid]);
  function edit(next:Partial<PensionProfile>){if(!ready||busy)return;dirty.current=true;setProfile(p=>({...p,...next}));setMessage('');}
  let plan:ReturnType<typeof buildPensionPlan>|null=null,planError='';
  if(ready&&market&&profile.totalBalance>0){try{plan=buildPensionPlan(profile,market);}catch(e){planError=explain(e);}}
  const weightSum=Object.values(profile.weights).reduce((s,w)=>s+w,0);
  const stale=!market||market.quoteMode!=='intraday'||status?.ok===false||PENSION_CODES.some(c=>{
    const time=Date.parse(market.priceTimes[c]);return!Number.isFinite(time)||now-time>30*60000||time-now>60000;
  });
  async function refresh(){
    if(demo||busy)return;setBusy('refresh');setError('');setMessage('');
    const controller=new AbortController();activeRequest.current=controller;
    const timer=setTimeout(()=>controller.abort(),60000);
    try{
      const res=await fetch('/api/pension/refresh',{method:'POST',headers:{...await apiHeaders(),'Content-Type':'application/json'},body:'{}',signal:controller.signal});
      if(!res.headers.get('content-type')?.includes('application/json'))throw new Error('연금 시세 서버에 연결할 수 없습니다.');
      const data=await res.json();if(!res.ok)throw new Error(data.error??'연금 시세 조회에 실패했습니다.');
      if(alive.current){setMarket(data.market);setAck(false);setMessage(data.message);}
    }catch(e){if(alive.current)setError(controller.signal.aborted?'조회 시간이 길어 중단했습니다. 잠시 후 다시 갱신하세요.':explain(e));}
    finally{clearTimeout(timer);if(activeRequest.current===controller)activeRequest.current=null;if(alive.current)setBusy('');}
  }
  async function save(){
    if(demo||!db||!user||!ownerUid||!ready||busy)return;setBusy('save');setError('');setMessage('');
    try{validatePensionProfile(profile);await setDoc(doc(db,'users',ownerUid,'portfolios','pension'),{...profile,updatedAt:serverTimestamp()});
      if(alive.current){dirty.current=false;setMessage('연금 보유수량·총자산·원금·목표 비중을 저장했습니다.');}}
    catch(e){if(alive.current)setError(explain(e));}finally{if(alive.current)setBusy('');}
  }
  async function record(){
    if(demo||!db||!user||!ownerUid||!ready||!plan||!market||busy||(stale&&!ack))return;
    setBusy('record');setError('');setMessage('');
    try{
      const month=new Intl.DateTimeFormat('sv-SE',{timeZone:'Asia/Seoul',year:'numeric',month:'2-digit'}).format(now);
      await addDoc(collection(db,'users',ownerUid,'portfolios','pension','rebalances'),{createdAt:serverTimestamp(),month,strategy:profile.strategy,currency:'KRW',kind:'계산안',
        equity:plan.equity,principal:profile.principal,remainingCash:plan.remainingCash,weights:profile.weights,
        priceUpdatedAt:market.updatedAt,oldestPriceAt:plan.oldestPriceAt,lines:plan.lines.map(({held,price,target})=>({held,price,target}))});
      if(alive.current)setMessage('연금 계산안을 이력에 저장했습니다.');
    }catch(e){if(alive.current)setError(explain(e));}finally{if(alive.current)setBusy('');}
  }
  function download(){if(!plan)return;const url=URL.createObjectURL(new Blob([pensionCsv(plan,profile)],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download='마법의연금_리밸런싱_계산안.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
  return <div className="pension-workspace">
    <div className="sujin-topline"><div><span className="eyebrow">연금 · 원화 국내 상장 ETF</span><h2>마법의 연금굴리기</h2><p>9개 ETF의 목표 비중과 보유수량을 한곳에서 관리하세요.</p></div>
      <button onClick={refresh} disabled={demo||!!busy}>{busy==='refresh'?'연금 시세 조회 중…':'연금 시세 갱신'}</button></div>
    <p className="micro">최근 성공 조회 {timeLabel(market?.updatedAt)} · {market?.quoteMode==='intraday'?'Yahoo 장중 시세':market?'종가 기준':'시세 대기'}{market&&` · 가격 기준 거래일 ${market.marketDate}`}</p>
    <p className="micro">연금 시세는 갱신 버튼을 누를 때 조회합니다. 최근 성공값은 함께 재사용합니다. {dirty.current?'저장하지 않은 연금 입력이 있습니다.':'연금 정보는 가족 공유 공간에 따로 저장됩니다.'}</p>
    {error&&<p className="error" role="alert">{error}</p>}{message&&<p className="success" role="status">{message}</p>}
    {status?.ok===false&&<p className="notice">연금 시세 조회에 실패해 직전 성공값과 원래 가격 시각을 유지합니다.</p>}
    <nav aria-label="연금 포트폴리오 메뉴">{[['overview','연금 리밸런싱'],['history','연금 저장 이력']].map(([id,label])=><button key={id} className={tab===id?'active':''} onClick={()=>setTab(id)}>{label}</button>)}</nav>
    {tab==='overview'&&<>
      <section className="panel pension-inputs"><div className="section-title"><h2>연금 자산과 목표 비중</h2><span>KRW · 정수 1주</span></div>
        {!ready&&<p className="notice">저장된 연금 정보를 확인하는 중입니다. 확인 전에는 수정·저장할 수 없습니다.</p>}
        <fieldset disabled={!ready||!!busy} className="sujin-fields">
          <div className="pension-totals"><label>연금 총자산 · 평가금액 + 예수금 (원)<input type="number" min="0" max="1000000000000" step="1" value={Number.isNaN(profile.totalBalance)?'':profile.totalBalance} onChange={e=>edit({totalBalance:e.target.valueAsNumber})}/></label>
            <label>연금 원금 · 누적 투자금 (원)<input type="number" min="0" max="1000000000000" step="1" value={Number.isNaN(profile.principal)?'':profile.principal} onChange={e=>edit({principal:e.target.valueAsNumber})}/></label></div>
          <div className="section-title"><p>목표 비중 합계 <strong className={Math.abs(weightSum-1)>1e-8?'negative':''}>{percent(weightSum)}</strong></p><div className="pension-actions"><button onClick={()=>edit({weights:defaultPensionProfile().weights})}>기본 비중 복원</button><button onClick={()=>edit({targetOverrides:defaultPensionProfile().targetOverrides})}>목표수량 자동 계산으로</button></div></div>
          <div className="pension-rows">{PENSION_ASSETS.map(a=>{const line=plan?.lines.find(l=>l.ticker===a.code);return <div className="pension-row" key={a.code}>
            <div className="pension-asset"><strong>{a.name}</strong><span>{a.group} · KRX:{a.code}</span><small>가격 {market&&market.prices[a.code]>0?won(market.prices[a.code]):'—'}</small></div>
            <label>목표 비중 (%)<input aria-label={`연금 ${a.code} 목표 비중 (%)`} type="number" min="0" max="100" step="0.1" value={Number.isNaN(profile.weights[a.code])?'':Number((profile.weights[a.code]*100).toFixed(8))} onChange={e=>edit({weights:{...profile.weights,[a.code]:e.target.valueAsNumber/100}})}/></label>
            <label>보유수량<input aria-label={`연금 ${a.code} 보유수량`} type="number" min="0" max="1000000000" step="1" value={Number.isNaN(profile.holdings[a.code])?'':profile.holdings[a.code]} onChange={e=>edit({holdings:{...profile.holdings,[a.code]:e.target.valueAsNumber}})}/></label>
            <label>목표 보유수량 · 빈칸은 자동<input aria-label={`연금 ${a.code} 목표 보유수량`} type="number" min="0" max="1000000000" step="1" placeholder={line?String(Math.floor(line.calculated)):'자동'} value={profile.targetOverrides[a.code]??''} onChange={e=>edit({targetOverrides:{...profile.targetOverrides,[a.code]:e.target.value===''?null:e.target.valueAsNumber}})}/></label>
          </div>;})}</div>
        </fieldset>
        <p className="micro">목표 보유수량은 리밸런싱 후 보유할 전체 수량입니다. 0을 입력하면 전량 매도 대상으로 계산합니다. 비워 두면 총자산 × 목표 비중 ÷ 가격을 정수로 내림합니다. 상품명은 기존 앱 표기이며 종목코드로 구분합니다.</p>
        <button className="primary" onClick={save} disabled={demo||!ready||!!busy}>연금 보유수량·설정 저장</button>
      </section>
      <section className="panel pension-plan"><div className="section-title"><h2>연금 리밸런싱 계산안</h2><span>현재 입력 기준</span></div>
        {!market&&<p className="empty">‘연금 시세 갱신’을 눌러 가격을 불러오세요. 기존 Streamlit 입력은 자동으로 이전되지 않으므로 처음 한 번 입력·저장해 주세요.</p>}
        {market&&profile.totalBalance<=0&&<p className="empty">총자산을 입력하면 목표 수량을 계산합니다.</p>}{planError&&<p className="error" role="alert">{planError}</p>}
        {plan&&<>
          <div className="metric-grid">{[['입력 총자산',won(plan.equity)],['보유 ETF 평가액',won(plan.holdingsValue)],['리밸런싱 후 예상 예수금',won(plan.remainingCash)],['입력 총자산 기준 손익',plan.profit===null?'—':won(plan.profit)],['원금 대비 수익률',percent(plan.profitRate)]].map(([label,value])=><div key={label}><span>{label}</span><strong>{value}</strong></div>)}</div>
          <p className="micro">원금 0은 미입력으로 처리합니다. 손익은 예수금을 포함한 입력 총자산에서 원금을 뺀 값입니다. 가장 오래된 사용 시세 {timeLabel(plan.oldestPriceAt)}</p>
          <div className="table-scroll"><table><thead><tr><th>종목</th><th>목표 배분액</th><th>계산 수량</th><th>보유 → 최종 목표</th><th>매매 수량</th><th>목표 금액</th></tr></thead><tbody>{plan.lines.map(l=><tr key={l.ticker}><th>{l.ticker}</th><td>{won(l.allocation)}</td><td>{l.calculated.toFixed(2)}</td><td>{l.held} → {l.target}</td><td className={l.trade<0?'negative':'positive'}>{l.trade>0?'+':''}{l.trade}</td><td>{won(l.target*l.price)}</td></tr>)}</tbody></table></div>
          <p className="micro">매매 수량은 최종 목표에서 보유수량을 뺀 값입니다. 계산안 저장은 실제 주문이나 보유수량 변경을 수행하지 않습니다. 수수료·세금·호가 차이는 제외합니다.</p>
          {stale&&<label className="check"><input type="checkbox" checked={ack} onChange={e=>setAck(e.target.checked)}/>연금 계산안이 종가·지연 또는 직전 성공 시세 기준임을 확인했습니다.</label>}
          <div className="pension-actions"><button onClick={record} disabled={demo||!!busy||(stale&&!ack)}>연금 계산안 저장</button><button onClick={download}>연금 CSV 다운로드</button></div>
        </>}
      </section>
    </>}
    {tab==='history'&&<section className="panel"><h2>연금 리밸런싱 이력</h2><p className="micro">최근 24건 · 계산안 저장 당시 입력과 가격 기준</p>{records.length?records.map(r=><details key={r.id}><summary>{r.month} · {won(r.equity)} · {r.kind}</summary><p className="micro">가격 조회 {timeLabel(r.priceUpdatedAt)} · 가장 오래된 가격 {timeLabel(r.oldestPriceAt)}</p><div className="table-scroll"><table><thead><tr><th>종목</th><th>비중</th><th>보유 → 목표</th><th>매매</th><th>가격</th></tr></thead><tbody>{r.lines.map((l,i)=><tr key={PENSION_CODES[i]}><th>{PENSION_CODES[i]}</th><td>{percent(r.weights[PENSION_CODES[i]])}</td><td>{l.held} → {l.target}</td><td>{l.target-l.held}</td><td>{won(l.price)}</td></tr>)}</tbody></table></div></details>):<p className="empty">저장한 연금 계산안이 없습니다.</p>}</section>}
  </div>;
}
