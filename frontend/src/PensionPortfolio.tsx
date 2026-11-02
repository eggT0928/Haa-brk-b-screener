import { useEffect, useRef, useState } from 'react';
import type { User } from 'firebase/auth';
import { addDoc, collection, doc, limit, onSnapshot, orderBy, query, serverTimestamp, setDoc } from 'firebase/firestore';
import { apiHeaders, db, demo } from './firebase';
import { percent, timeLabel } from './portfolio';
import { buildPensionPlan, defaultPensionProfile, effectivePensionWeights, GOLD_CODES, PENSION_PATH, PENSION_ASSETS, PENSION_CODES, pensionAllocation, pensionDifference, pensionCsv, validatePensionProfile, won, type PensionMarket, type PensionProfile, type PensionRecord } from './pensionModel';
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
  const [showFraction,setShowFraction]=useState(false);
  const dirty=useRef(false),alive=useRef(true),activeRequest=useRef<AbortController|null>(null);
  useEffect(()=>{
    alive.current=true;
    if(demo||!db||!user||!ownerUid)return()=>{alive.current=false;activeRequest.current?.abort();};
    const fail=(e:unknown)=>{if(alive.current){setReady(false);setError(`연금 저장 데이터 연결 오류: ${explain(e)}`);}};
    const stops=[
      onSnapshot(doc(db,'users',ownerUid,'portfolios',PENSION_PATH),s=>{
        if(!alive.current||dirty.current)return;
        const next=s.exists()?s.data() as PensionProfile:defaultPensionProfile();
        try{validatePensionProfile(next);setProfile(next);setReady(true);}catch(e){fail(e);}
      },fail),
      onSnapshot(doc(db,'market','pensionV2'),s=>{if(alive.current){setMarket(s.exists()?s.data() as PensionMarket:null);setAck(false);}},e=>{if(alive.current)setError(explain(e));}),
      onSnapshot(doc(db,'status','pensionQuotesV2'),s=>{if(alive.current){setStatus(s.exists()?s.data() as UpdateStatus:null);setAck(false);}},e=>{if(alive.current)setError(explain(e));}),
      onSnapshot(query(collection(db,'users',ownerUid,'portfolios',PENSION_PATH,'rebalances'),orderBy('createdAt','desc'),limit(24)),s=>{if(alive.current)setRecords(s.docs.map(d=>{const data=d.data();return {...data,id:d.id,lines:PENSION_CODES.map((_,i)=>({held:data.lines[i*3],price:data.lines[i*3+1],target:data.lines[i*3+2]}))} as PensionRecord;}));},e=>{if(alive.current)setError(explain(e));}),
    ];
    return()=>{alive.current=false;activeRequest.current?.abort();stops.forEach(stop=>stop());};
  },[user?.uid,ownerUid]);
  function edit(next:Partial<PensionProfile>){if(!ready||busy)return;dirty.current=true;setProfile(p=>({...p,...next}));setMessage('');}
  let plan:ReturnType<typeof buildPensionPlan>|null=null,planError='';
  if(ready&&market&&(profile.totalBalance>0||profile.cashBalance!==undefined||Object.values(profile.holdings).some(n=>n>0))){try{plan=buildPensionPlan(profile,market);}catch(e){planError=explain(e);}}
  const displayLines=plan?.lines.map(l=>{
    const target=showFraction&&profile.targetOverrides[l.ticker]===null?Math.floor(l.calculated*100)/100:l.target;
    return {...l,displayTarget:target,displayTrade:Math.round((target-l.held)*100)/100};
  })??[];
  const displayCash=plan?plan.equity-displayLines.reduce((sum,l)=>sum+l.displayTarget*l.price,0):0;
  const weights=effectivePensionWeights(profile,market);
  const allocation=ready?pensionAllocation(profile,market):null;
  const weightSum=Object.values(profile.weights).reduce((s,w)=>s+w,0)+profile.goldTargetWeight;
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
    try{validatePensionProfile(profile);await setDoc(doc(db,'users',ownerUid,'portfolios',PENSION_PATH),{...Object.fromEntries(Object.entries(profile).filter(([,value])=>value!==undefined)),updatedAt:serverTimestamp()});
      if(alive.current){dirty.current=false;setMessage('연금 보유수량·총자산·원금·목표 비중을 저장했습니다.');}}
    catch(e){if(alive.current)setError(explain(e));}finally{if(alive.current)setBusy('');}
  }
  async function record(){
    if(demo||!db||!user||!ownerUid||!ready||!plan||!market||busy||(stale&&!ack))return;
    setBusy('record');setError('');setMessage('');
    try{
      const month=new Intl.DateTimeFormat('sv-SE',{timeZone:'Asia/Seoul',year:'numeric',month:'2-digit'}).format(now);
      await addDoc(collection(db,'users',ownerUid,'portfolios',PENSION_PATH,'rebalances'),{createdAt:serverTimestamp(),month,strategy:profile.strategy,currency:'KRW',kind:'계산안',
        equity:plan.equity,principal:profile.principal,remainingCash:plan.remainingCash,weights:plan.weights,
        priceUpdatedAt:market.updatedAt,oldestPriceAt:plan.oldestPriceAt,lines:plan.lines.flatMap(({held,price,target})=>[held,price,target])});
      if(alive.current)setMessage('연금 계산안을 이력에 저장했습니다.');
    }catch(e){if(alive.current)setError(explain(e));}finally{if(alive.current)setBusy('');}
  }
  function download(){if(!plan)return;const url=URL.createObjectURL(new Blob([pensionCsv(plan,profile)],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download='마법의연금_리밸런싱_계산안.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
  return <div className="pension-workspace">
    <div className="sujin-topline"><div><span className="eyebrow">연금 · 원화 국내 상장 ETF</span><h2>연금저축 리밸런싱</h2><p>12개 ETF의 목표 비중과 보유수량을 한곳에서 관리하세요.</p></div>
      <button onClick={refresh} disabled={demo||!!busy}>{busy==='refresh'?'연금 시세 조회 중…':'연금 시세 갱신'}</button></div>
    <p className="micro">최근 성공 조회 {timeLabel(market?.updatedAt)} · {market?.quoteMode==='intraday'?`${market.source??'온라인'} 최근 시세`:market?'종가 기준':'시세 대기'}{market&&` · 가격 기준 거래일 ${market.marketDate}`}</p>
    <p className="micro">한국 거래일 09:00~15:30에 10분 간격으로 갱신합니다. 수동 갱신도 가능합니다. {dirty.current?'저장하지 않은 연금 입력이 있습니다.':'연금 정보는 가족 공유 공간에 따로 저장됩니다.'}</p>
    {error&&<p className="error" role="alert">{error}</p>}{message&&<p className="success" role="status">{message}</p>}
    {status?.ok===false&&<p className="notice">연금 시세 조회에 실패해 직전 성공값과 원래 가격 시각을 유지합니다.</p>}
    <nav aria-label="연금 포트폴리오 메뉴">{[['overview','연금 리밸런싱'],['history','연금 저장 이력']].map(([id,label])=><button key={id} className={tab===id?'active':''} onClick={()=>setTab(id)}>{label}</button>)}</nav>
    {tab==='overview'&&<section className="panel pension-sheet" aria-label="연금 현재 보유 비중">
      <div className="section-title"><h2>연금 리밸런싱 계산안</h2><span>KRW · 현재 시세 기준</span></div>
      {!ready&&<p className="notice">저장된 연금 정보를 확인하는 중입니다. 확인 전에는 수정·저장할 수 없습니다.</p>}
      {allocation?.assumedCash&&<p className="notice">입력한 총자산이 현재 보유 ETF 평가액보다 작아, 예수금을 0원으로 가정하고 보유 평가액으로 계산합니다. 실제 예수금이 있다면 예수금에 입력하세요.</p>}
      <fieldset disabled={!ready||!!busy} className="sujin-fields">
        <div className="pension-summary-inputs">
          <label>기존 연금 총자산 (원)<input aria-label="기존 연금 총자산 · 평가금액 + 예수금 (원)" type="number" min="0" max="1000000000000" step="1" value={Number.isNaN(profile.totalBalance)?'':profile.totalBalance} onChange={e=>edit({totalBalance:e.target.valueAsNumber})}/></label>
          <label>예수금 (원)<input aria-label="연금 예수금 (원)" type="number" min="0" max="1000000000000" step="1" placeholder="미입력 시 총자산 사용" value={profile.cashBalance===undefined||Number.isNaN(profile.cashBalance)?'':profile.cashBalance} onChange={e=>edit({cashBalance:e.target.value===''?undefined:e.target.valueAsNumber})}/></label>
          <label>원금 (원)<input aria-label="연금 원금 · 누적 투자금 (원)" type="number" min="0" max="1000000000000" step="1" value={Number.isNaN(profile.principal)?'':profile.principal} onChange={e=>edit({principal:e.target.valueAsNumber})}/></label>
          <label>금 ETF 합산 목표 (%)<input aria-label="금 ETF 합산 목표 비중 (%)" type="number" min="0" max="100" step="0.1" value={Number.isNaN(profile.goldTargetWeight)?'':profile.goldTargetWeight*100} onChange={e=>edit({goldTargetWeight:e.target.valueAsNumber/100})}/></label>
        </div>
        <div className="pension-sheet-actions"><span>목표 비중 합계 <strong className={Math.abs(weightSum-1)>1e-8?'negative':''}>{percent(weightSum)}</strong></span><button type="button" onClick={()=>edit({weights:defaultPensionProfile().weights,goldTargetWeight:.20})}>기본 비중 복원</button><button type="button" onClick={()=>edit({targetOverrides:defaultPensionProfile().targetOverrides})}>목표수량 자동 계산으로</button>
          <label>수량 표시<select aria-label="연금 수량 표시 방식" value={showFraction?'fractional':'whole'} onChange={e=>setShowFraction(e.target.value==='fractional')}><option value="whole">정수 · 1주</option><option value="fractional">소수점 · 0.01주 (참고)</option></select></label>
        </div>
        {plan&&<div className="pension-summary-metrics">{[['계산 기준 총자산',won(plan.equity)],['보유 ETF 평가액',won(plan.holdingsValue)],[showFraction?'소수점 계산상 잔액':'리밸런싱 후 예상 예수금',won(showFraction?displayCash:plan.remainingCash)],['계산 기준 손익',plan.profit===null?'—':won(plan.profit)],['원금 대비 수익률',percent(plan.profitRate)]].map(([label,value])=><div key={label}><span>{label}</span><strong>{value}</strong></div>)}</div>}
        {!market&&<p className="empty">‘연금 시세 갱신’을 눌러 가격을 불러오세요. 기존 Streamlit 입력은 자동으로 이전되지 않으므로 처음 한 번 입력·저장해 주세요. 이전 9종목 구성의 입력과 이력은 별도로 보존됩니다.</p>}
        {market&&!plan&&!planError&&<p className="empty">보유수량과 예수금 또는 기존 총자산을 입력하면 목표 수량을 계산합니다.</p>}{planError&&<p className="error" role="alert">{planError}</p>}
        {showFraction&&<p className="micro">소수점 수량은 목표 비중에 맞춘 계산 참고값이며 0.01주 단위로 내림합니다. 직접 입력한 목표수량은 그대로 적용됩니다. 거래 가능한 수량 단위를 확인해 주세요. 소수점 화면에서는 계산안 저장과 CSV 다운로드가 제한됩니다.</p>}
        {showFraction&&plan&&displayCash<-.0001&&<p className="notice">직접 입력한 목표수량을 포함한 소수점 계산상 금액이 총자산을 초과합니다. 입력 수량을 조정하세요.</p>}
        <p className="micro pension-sheet-hint">종목별 입력과 계산 결과를 한 줄에서 확인하세요. 표를 좌우로 밀어 모든 항목을 볼 수 있습니다.</p>
        <div className="table-scroll pension-sheet-scroll" tabIndex={0} aria-label="연금 종목별 표 가로 스크롤"><table className="pension-sheet-table"><thead><tr><th scope="col">종목</th><th scope="col">목표 비중<br/>(%)</th><th scope="col">목표 배분액</th><th scope="col">현재 가격</th><th scope="col">계산 수량</th><th scope="col">최종 목표<br/>수량 입력<br/>(빈칸은 자동)</th><th scope="col">보유 수량<br/>입력</th><th scope="col">현재 평가액</th><th scope="col">현재 비중</th><th scope="col">목표 차이</th><th scope="col">매매 수량</th><th scope="col">조정 후<br/>평가액</th><th scope="col">조정 후<br/>비중</th></tr></thead><tbody>
          {PENSION_ASSETS.map(a=>{const line=displayLines.find(l=>l.ticker===a.code);const current=allocation?.rows.find(r=>r.code===a.code);const price=market?.prices?.[a.code];const target=weights[a.code];const validPrice=Number.isFinite(price)&&price!>0;const finalValue=line?line.displayTarget*line.price:null;
            return <tr key={a.code}><th scope="row"><strong>{a.name}</strong><small>{a.group} · {a.code}</small></th>
              <td><input disabled={GOLD_CODES.includes(a.code)} aria-label={`연금 ${a.code} 목표 비중 (%)`} type="number" min="0" max="100" step="0.1" value={Number.isNaN(target)?'':Number((target*100).toFixed(8))} onChange={e=>edit({weights:{...profile.weights,[a.code]:e.target.valueAsNumber/100}})}/></td>
              <td>{line?won(line.allocation):'—'}</td><td title={market?.priceTimes?.[a.code]?timeLabel(market.priceTimes[a.code]):''}>{validPrice?won(price!):'—'}</td><td>{line?line.calculated.toFixed(2):'—'}</td>
              <td><input aria-label={`연금 ${a.code} 목표 보유수량`} type="number" min="0" max="1000000000" step="1" placeholder={line?String(Math.floor(line.calculated)):'자동'} value={profile.targetOverrides[a.code]??''} onChange={e=>edit({targetOverrides:{...profile.targetOverrides,[a.code]:e.target.value===''?null:e.target.valueAsNumber}})}/></td>
              <td><input aria-label={`연금 ${a.code} 보유수량`} type="number" min="0" max="1000000000" step="1" value={Number.isNaN(profile.holdings[a.code])?'':profile.holdings[a.code]} onChange={e=>edit({holdings:{...profile.holdings,[a.code]:e.target.valueAsNumber}})}/></td>
              <td>{current?won(current.value):'—'}</td><td>{current?percent(current.currentWeight):'—'}</td><td>{current?pensionDifference(current.currentWeight,target):'—'}</td>
              <td className={line&&line.displayTrade<0?'negative':'positive'}>{line?`${line.displayTrade>0?'+':''}${showFraction?line.displayTrade.toFixed(2):line.displayTrade}`:'—'}{line&&<small className="pension-computed">{line.held} → {showFraction?line.displayTarget.toFixed(2):line.displayTarget}</small>}</td>
              <td>{finalValue===null?'—':won(finalValue)}</td><td>{finalValue===null||!plan?'—':percent(finalValue/plan.equity)}</td>
            </tr>;})}
          {allocation&&<tr className="pension-total-row"><th scope="row">금 ETF 합계</th><td>{percent(profile.goldTargetWeight)}</td><td>{plan?won(plan.equity*profile.goldTargetWeight):'—'}</td><td>—</td><td>—</td><td>—</td><td>—</td><td>{won(allocation.goldWeight*allocation.equity)}</td><td>{percent(allocation.goldWeight)}</td><td>{pensionDifference(allocation.goldWeight,profile.goldTargetWeight)}</td><td>—</td><td>{plan?won(displayLines.filter(l=>GOLD_CODES.includes(l.ticker)).reduce((sum,l)=>sum+l.displayTarget*l.price,0)):'—'}</td><td>{plan?percent(displayLines.filter(l=>GOLD_CODES.includes(l.ticker)).reduce((sum,l)=>sum+l.displayTarget*l.price,0)/plan.equity):'—'}</td></tr>}
          {allocation&&<tr className="pension-total-row"><th scope="row">{allocation.explicitCash?'현재 예수금':'현재 예수금 (추정)'}</th><td>0.00%</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>{won(allocation.cash)}</td><td>{percent(allocation.cashWeight)}</td><td>{pensionDifference(allocation.cashWeight,0)}</td><td>—</td><td>{plan?won(showFraction?displayCash:plan.remainingCash):'—'}</td><td>{plan?percent((showFraction?displayCash:plan.remainingCash)/plan.equity):'—'}</td></tr>}
        </tbody></table></div>
      </fieldset>
      <p className="micro">예수금을 입력하면 보유 ETF의 현재 평가액과 합산해 총자산을 자동 계산합니다. 예수금이 없다면 0을 입력하세요. 비워 두면 기존 입력 총자산을 사용하며, 보유 평가액보다 낮으면 예수금 0원으로 가정합니다.</p>
      <p className="micro">TIGER·ACE 금 ETF는 합산 목표를 현재 평가액 비율로 나눕니다. 최종 목표수량은 리밸런싱 후 보유할 전체 수량이며 빈칸은 자동, 0은 전량 매도입니다. 현재 비중은 예수금 포함 총자산 기준이며 목표 차이는 현재 − 목표입니다.</p>
      <p className="micro">조정 후 평가액·비중은 표시된 목표수량과 현재 가격으로 계산한 예상값입니다. 매매 수량은 최종 목표 − 보유수량입니다. 실제 주문이나 보유수량 변경은 실행하지 않으며 수수료·세금·호가 차이는 제외합니다.</p>
      {plan&&<p className="micro">원금 0은 미입력으로 처리합니다. 손익은 계산 기준 총자산에서 원금을 뺀 값입니다. 가장 오래된 사용 시세 {timeLabel(plan.oldestPriceAt)}</p>}
      {plan&&stale&&<label className="check"><input type="checkbox" checked={ack} onChange={e=>setAck(e.target.checked)}/>연금 계산안이 종가·지연 또는 직전 성공 시세 기준임을 확인했습니다.</label>}
      <div className="pension-actions"><button className="primary" onClick={save} disabled={demo||!ready||!!busy}>연금 보유수량·설정 저장</button><button onClick={record} disabled={demo||!!busy||!plan||showFraction||(stale&&!ack)}>연금 계산안 저장</button><button onClick={download} disabled={!plan||showFraction}>연금 CSV 다운로드</button></div>
    </section>}
    {tab==='history'&&<section className="panel"><h2>연금 리밸런싱 이력</h2><p className="micro">최근 24건 · 계산안 저장 당시 입력과 가격 기준</p>{records.length?records.map(r=><details key={r.id}><summary>{r.month} · {won(r.equity)} · {r.kind}</summary><p className="micro">가격 조회 {timeLabel(r.priceUpdatedAt)} · 가장 오래된 가격 {timeLabel(r.oldestPriceAt)}</p><div className="table-scroll"><table><thead><tr><th>종목</th><th>비중</th><th>보유 → 목표</th><th>매매</th><th>가격</th></tr></thead><tbody>{r.lines.map((l,i)=><tr key={PENSION_CODES[i]}><th>{PENSION_CODES[i]}</th><td>{percent(r.weights[PENSION_CODES[i]])}</td><td>{l.held} → {l.target}</td><td>{l.target-l.held}</td><td>{won(l.price)}</td></tr>)}</tbody></table></div></details>):<p className="empty">저장한 연금 계산안이 없습니다.</p>}</section>}
  </div>;
}
