import { useEffect, useRef, useState } from 'react';
import { onAuthStateChanged, type User } from 'firebase/auth';
import { addDoc, collection, doc, limit, onSnapshot, orderBy, query, serverTimestamp, setDoc } from 'firebase/firestore';
import { apiHeaders, auth, configured, db, demo, login, logout } from './firebase';
import { demoConfirmed, demoMarket, demoPreview } from './demo';
import { Curve } from './Curve';
import { buildPlan, defaultProfile, money, percent, timeLabel, validateProfile } from './portfolio';
import { TICKERS, type Backtest, type Market, type Profile, type Rebalance, type Signal, type UpdateStatus } from './types';

const explainError = (e: unknown) => e instanceof Error ? e.message : '요청 처리에 실패했습니다.';
const weightsLabel = (w: Record<string, number>) => Object.entries(w).map(([t,n]) => `${t === 'CASH' ? '현금' : t} ${(n * 100).toFixed(0)}%`).join(' · ');

function SignalCard({ signal, expected }: { signal: Signal | null; expected?: boolean }) {
  return <section className={`signal-card ${expected ? 'preview' : ''}`}>
    <div className="eyebrow">{expected ? '이번 달을 지금 마감한다면' : '완료된 월말 · 실전 기준'}</div>
    <div className="card-title"><h2>{expected ? '예상 신호' : '확정 신호'}</h2><span className={`badge ${expected ? 'amber' : ''}`}>{signal ? signal.month : '대기 중'}</span></div>
    {signal ? <>
      <p className="regime">{signal.regime} <span>TIP {percent(signal.scores.TIP)}</span></p>
      <div className="asset-chips">{Object.entries(signal.weights).map(([t,w]) => <span key={t}><strong>{t === 'CASH' ? '현금' : t}</strong><small>{(w * 100).toFixed(0)}%</small></span>)}</div>
      <p className="micro">{expected ? 'Yahoo 가격으로 1·3·6·12개월 모멘텀과 자산 선택을 다시 계산합니다. 실제 리밸런싱에는 사용하지 않습니다.' : 'HAA 80% + BRK-B 20%. 모멘텀 신호는 SPY, 실전 원자재는 PDBC 기준입니다.'}</p>
      <div className="card-footer">{expected ? '최저 시세 시각' : '기준 종가 시각'}<br />{timeLabel(expected ? signal.oldestPriceAt ?? signal.asOf : signal.asOf)}</div>
    </> : <div className="empty">아직 저장된 신호가 없습니다.<br />첫 예약 갱신 완료 후 표시됩니다.</div>}
  </section>;
}

export function BacktestView({ result }: { result: Backtest }) {
  return <div className="backtest-result">
    <p className="micro">적용 {result.actualStart} ~ {result.actualEnd} / 최대 {result.availableStart} ~ {result.availableEnd}<br />가격 캐시 갱신: {timeLabel(result.dataUpdatedAt)}</p>
    {result.warnings.map(w => <p className="notice" key={w}>{w}</p>)}
    <div className="metric-grid">{[
      ['누적 수익률', percent(result.metrics.totalReturn)], ['연복리 수익률', percent(result.metrics.cagr)],
      ['최대 낙폭', percent(result.metrics.mdd)], ['연환산 변동성', percent(result.metrics.volatility)],
      ['샤프 비율', result.metrics.sharpe?.toFixed(2) ?? '—'], ['무위험수익률 · ^IRX 평균', percent(result.metrics.riskFree)],
    ].map(([label,value]) => <div key={label}><span>{label}</span><strong>{value}</strong></div>)}</div>
    <Curve data={result.curve} />
    <details><summary>월말 낙폭 추이</summary><Curve data={result.curve} drawdown /></details>
    <details><summary>연도별 수익률 / 월별 수익률</summary>
      <div className="year-bars">{result.yearly.map(v => <div key={v.year}><span>{v.year}</span><strong className={v.return < 0 ? 'negative' : 'positive'}>{percent(v.return)}</strong></div>)}</div>
      <div className="table-scroll"><table><thead><tr><th>연도</th>{Array.from({length:12},(_,i) => <th key={i}>{i+1}월</th>)}</tr></thead><tbody>
        {[...new Set(result.monthly.map(r => r.month.slice(0,4)))].reverse().map(y => <tr key={y}><th>{y}</th>{Array.from({length:12},(_,i) => {
          const item = result.monthly.find(r => r.month === `${y}-${String(i+1).padStart(2,'0')}`);
          return <td key={i} className={item ? (item.return < 0 ? 'negative heat-red' : 'positive heat-green') : ''}>{item ? percent(item.return) : '—'}</td>;
        })}</tr>)}
      </tbody></table></div>
      <p className="micro">첫해·마지막해는 선택 기간의 부분연도일 수 있습니다.</p>
    </details>
    <details><summary>백테스트 리밸런싱 전체 이력 ({result.history.length}개월)</summary><div className="table-scroll"><table>
      <thead><tr><th>신호 월말</th><th>수익 적용 월</th><th>목표 비중</th><th>월 수익률</th></tr></thead>
      <tbody>{[...result.history].reverse().map(h => <tr key={h.returnMonth}><td>{h.signalMonth}</td><td>{h.returnMonth}</td><td>{weightsLabel(h.weights)}</td><td>{percent(h.return)}</td></tr>)}</tbody>
    </table></div></details>
  </div>;
}

export default function App() {
  const [user, setUser] = useState<User | null>(null);
  const [approved, setApproved] = useState(demo);
  const [tab, setTab] = useState('overview');
  const [profile, setProfile] = useState<Profile>(defaultProfile);
  const [confirmed, setConfirmed] = useState<Signal | null>(demo ? demoConfirmed : null);
  const [preview, setPreview] = useState<Signal | null>(demo ? demoPreview : null);
  const [market, setMarket] = useState<Market | null>(demo ? demoMarket : null);
  const [statuses, setStatuses] = useState<Record<string, UpdateStatus>>({});
  const [history, setHistory] = useState<Signal[]>([]);
  const [rebalances, setRebalances] = useState<Rebalance[]>([]);
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');
  const [saving, setSaving] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<Backtest | null>(null);
  const [initial, setInitial] = useState(10000);
  const [acknowledged, setAcknowledged] = useState(false);
  const [now, setNow] = useState(Date.now());
  const dirty = useRef(false);
  const requestGeneration = useRef(0);
  useEffect(() => { const id = setInterval(() => setNow(Date.now()), 60000); return () => clearInterval(id); }, []);
  useEffect(() => auth ? onAuthStateChanged(auth, setUser) : undefined, []);
  useEffect(() => {
    if (demo || !db) return;
    setApproved(false); setProfile(defaultProfile()); dirty.current = false;
    setConfirmed(null); setPreview(null); setMarket(null); setHistory([]); setRebalances([]); setStatuses({}); setResult(null);
    requestGeneration.current++; setLoading(false); setError(''); setMessage('');
    if (!user) return;
    return onSnapshot(doc(db, 'access', user.uid), s => setApproved(s.data()?.enabled === true), e => setError(explainError(e)));
  }, [user]);
  useEffect(() => {
    if (!db || !user || !approved || demo) return;
    const failure = (e: unknown) => setError(`데이터를 불러오지 못했습니다: ${explainError(e)}`);
    const subscriptions = [
      onSnapshot(doc(db,'users',user.uid), s => { if (s.exists() && !dirty.current) setProfile(s.data() as Profile); }, failure),
      onSnapshot(doc(db,'signals','confirmed'), s => setConfirmed(s.exists() ? s.data() as Signal : null), failure),
      onSnapshot(doc(db,'signals','preview'), s => setPreview(s.exists() ? s.data() as Signal : null), failure),
      onSnapshot(doc(db,'market','current'), s => { setMarket(s.exists() ? s.data() as Market : null); setAcknowledged(false); }, failure),
      ...['intraday','daily'].map(job => onSnapshot(doc(db!,'status',job), s => { if(s.exists()) setStatuses(old => ({...old,[job]:s.data() as UpdateStatus})); }, failure)),
      onSnapshot(query(collection(db,'signalHistory'),orderBy('month','desc'),limit(24)), s => setHistory(s.docs.map(d => d.data() as Signal)), failure),
      onSnapshot(query(collection(db,'users',user.uid,'rebalances'),orderBy('createdAt','desc'),limit(24)), s => setRebalances(s.docs.map(d => ({...d.data(),id:d.id} as Rebalance))), failure),
    ];
    return () => subscriptions.forEach(stop => stop());
  }, [user, approved]);
  const edit = (update: Partial<Profile>) => { dirty.current = true; setProfile(p => ({...p,...update})); setMessage(''); };
  let plan: ReturnType<typeof buildPlan> | null = null, planError = '';
  if (confirmed && market) { try { plan = buildPlan(profile, confirmed, market); } catch(e) { planError = explainError(e); } }
  const stale = !market || Object.entries(market.priceTimes).some(([ticker,t]) =>
    (profile.holdings[ticker] > 0 || plan?.lines.some(l => l.ticker === ticker)) && now - new Date(t).getTime() > 30*60000);
  const failed = Object.values(statuses).some(s => !s.ok);
  async function saveProfile() {
    if (!db || !user || demo) return;
    setSaving(true); setError('');
    try { validateProfile(profile); await setDoc(doc(db,'users',user.uid), {...profile,updatedAt:serverTimestamp()}); dirty.current = false; setMessage('보유수량과 설정을 저장했습니다.'); }
    catch(e) { setError(explainError(e)); } finally { setSaving(false); }
  }
  async function recordPlan() {
    if (!db || !user || !plan || !confirmed || demo) return;
    setSaving(true); setError('');
    try {
      await addDoc(collection(db,'users',user.uid,'rebalances'), { createdAt:serverTimestamp(),signalMonth:confirmed.month,
        sp500:profile.sp500,equity:plan.equity,lines:plan.lines,kind:'계산안' });
      setMessage('계산안을 이력에 저장했습니다. 실제 주문·보유수량 변경은 수행하지 않았습니다.');
    } catch(e) { setError(explainError(e)); } finally { setSaving(false); }
  }
  async function runBacktest() {
    const generation = ++requestGeneration.current;
    setLoading(true); setError(''); setResult(null);
    try {
      const params = new URLSearchParams({initial:String(initial)});
      if(profile.backtestStart) params.set('start',profile.backtestStart);
      if(profile.backtestEnd) params.set('end',profile.backtestEnd);
      const res = await fetch(`/api/backtest?${params}`, {headers:await apiHeaders(), signal:AbortSignal.timeout(60000)});
      const contentType = res.headers.get('content-type') ?? '';
      if (!contentType.includes('application/json')) throw new Error('백테스트 서버에 연결할 수 없습니다. Hosting 연결을 확인하세요.');
      const data = await res.json(); if(!res.ok) throw new Error(data.error);
      if(generation === requestGeneration.current) setResult(data);
    } catch(e) { if(generation === requestGeneration.current) setError(explainError(e)); }
    finally { if(generation === requestGeneration.current) setLoading(false); }
  }
  return <>
    <header><a className="brand" href="#"><span className="brand-mark">H</span><span>HAA <em>포트폴리오</em></span></a>
      <div className="account">{demo ? <span>화면 검토 모드</span> : user ? <><span>{user.displayName ?? user.email}</span><button className="quiet" onClick={() => logout().catch(e => setError(explainError(e)))}>로그아웃</button></> : <button disabled={!configured} onClick={() => login().catch(e => setError(explainError(e)))}>Google 로그인</button>}</div>
    </header>
    <main>
      <div className="page-heading"><div><p className="eyebrow">HAA 80 / BRK-B 20</p><h1>월말의 원칙,<br className="mobile-only" /> 오늘의 포트폴리오.</h1><p>확정된 기준은 지키고, 다음 달의 변화를 살펴보세요.</p></div><div className="stamp"><span className="dot" />{failed ? '직전 성공값 표시 중' : market ? '공용 데이터 캐시' : '데이터 연결 대기'}<small>최근 성공 갱신<br />{timeLabel(market?.updatedAt)}</small></div></div>
      {demo && <div className="notice">샘플 데이터입니다. 실제 시세·투자 신호가 아니며 로그인, 저장 및 백테스트 서버 호출은 비활성화되어 있습니다.</div>}
      {!demo && !configured && <div className="notice">Firebase 연결 설정이 아직 없습니다. README의 환경설정 절차를 완료한 뒤 다시 빌드하세요. 실제 신호는 표시하지 않습니다.</div>}
      {!demo && configured && !user && <div className="notice">로그인 후 승인된 계정만 신호와 개인 보유수량을 볼 수 있습니다.</div>}
      {!demo && user && !approved && <div className="notice">관리자 승인을 기다리고 있습니다. 관리자에게 이 사용자 ID를 전달하세요: <code>{user.uid}</code></div>}
      {error && <div className="error" role="alert">{error}<button className="quiet" onClick={() => setError('')}>닫기</button></div>}
      {message && <div className="success" role="status">{message}</div>}
      {approved && <>
        {failed && <div className="notice">갱신 실패: 마지막 성공 신호를 유지하고 있습니다. {Object.entries(statuses).filter(([,s])=>!s.ok).map(([j,s])=> `${j === 'daily' ? '일일' : '장중'} 시도 ${timeLabel(s.lastAttemptAt)}`).join(' / ')}</div>}
        <nav aria-label="포트폴리오 메뉴">{[['overview','신호 & 리밸런싱'],['backtest','장기 백테스트'],['history','저장 이력']].map(([id,label]) => <button key={id} className={tab===id?'active':''} onClick={()=>setTab(id)}>{label}</button>)}</nav>
        {tab === 'overview' && <>
          <div className="signal-grid"><SignalCard signal={confirmed}/><SignalCard signal={preview} expected/></div>
          <details className="panel momentum"><summary>전체 자산군 모멘텀 비교 <span>신호 기준 SPY · 1 / 3 / 6 / 12개월</span></summary>
            <div className="table-scroll"><table><thead><tr><th>자산</th><th>확정 점수</th><th>예상 점수</th>{[1,3,6,12].map(n=><th key={n}>예상 {n}개월</th>)}</tr></thead>
              <tbody>{TICKERS.filter(t=>t!=='SPYM').map(t=><tr key={t}><th>{t}{t==='TIP'?' · 카나리아':''}</th><td>{percent(confirmed?.scores[t])}</td><td>{percent(preview?.scores[t])}</td>{[1,3,6,12].map(n=><td key={n}>{percent(preview?.returns[String(n)]?.[t])}</td>)}</tr>)}</tbody></table></div>
          </details>
          <div className="workspace-grid">
            <section className="panel holdings"><div className="section-title"><h2>나의 보유자산</h2><span>USD 기준</span></div>
              <label>실전 S&P500 ETF<select value={profile.sp500} onChange={e=>edit({sp500:e.target.value as Profile['sp500']})}><option>SPY</option><option>SPYM</option></select></label>
              <p className="micro">선택 ETF는 실행 수량에만 반영합니다. 모멘텀과 백테스트는 항상 SPY입니다.</p>
              <label>현금 잔액 (USD)<input type="number" min="0" max="1000000000000" step="0.01" value={Number.isNaN(profile.cash)?'':profile.cash} onChange={e=>edit({cash:e.target.valueAsNumber})}/></label>
              <div className="holdings-grid">{TICKERS.map(t=><label key={t}>{t}<input aria-label={`${t} 보유수량`} type="number" min="0" max="1000000000" step="any" value={Number.isNaN(profile.holdings[t])?'':profile.holdings[t]} onChange={e=>edit({holdings:{...profile.holdings,[t]:e.target.valueAsNumber}})}/></label>)}</div>
              <button className="primary full" disabled={saving || demo} onClick={saveProfile}>{saving?'저장 중…':'보유수량·설정 저장'}</button>
            </section>
            <section className="panel rebalance"><div className="section-title"><h2>리밸런싱 계산안</h2><span>{confirmed?.month ?? '—'} 확정 신호 기준</span></div>
              {planError && <p className="error">{planError}</p>}
              {plan ? <>
                <div className="balance"><span>현재 평가금액</span><strong>{money(plan.equity)}</strong><small>정수 수량 매수 후 예상 현금 {money(plan.remainingCash)}</small></div>
                {(stale || failed) && <p className="notice">장외·지연 시세 또는 직전 성공값입니다. 거래 전 증권사 가격을 확인하세요.</p>}
                <div className="table-scroll"><table><thead><tr><th>실전 종목</th><th>비중</th><th>보유 → 목표</th><th>매매 수량</th><th>평가 가격</th></tr></thead><tbody>
                  {plan.lines.map(l=><tr key={l.ticker}><th>{l.ticker}</th><td>{(l.weight*100).toFixed(0)}%</td><td>{l.held} → {l.target}</td><td className={l.trade<0?'negative':'positive'}>{l.trade>0?'+':''}{l.trade}</td><td title={timeLabel(market?.priceTimes[l.ticker])}>{money(l.price)}</td></tr>)}
                </tbody></table></div>
                <p className="micro">매도 후 매수하는 계산입니다. 수수료·세금·환전·호가 차이는 제외하며, 보유자산은 자동 변경하지 않습니다. 가격 셀에 마우스를 올리면 시세 시각을 확인할 수 있습니다.</p>
                {(stale || failed) && <label className="check"><input type="checkbox" checked={acknowledged} onChange={e=>setAcknowledged(e.target.checked)}/>지연 또는 직전 성공 시세인 점을 확인했습니다.</label>}
                <button disabled={saving || demo || ((stale || failed) && !acknowledged)} onClick={recordPlan}>계산안 이력 저장 · 주문 아님</button>
              </> : !planError && <p className="empty">확정 신호와 가격이 준비되면 보유수량에 맞는 계산안이 표시됩니다.</p>}
            </section>
          </div>
        </>}
        {tab === 'backtest' && <section className="panel"><div className="section-title"><h2>장기 백테스트</h2><span>HAA 80% + BRK-B 20%</span></div>
          <p>실전 ETF 선택과 무관하게 SPY를 사용합니다. 현재 원본 코드와 동일하게 PDBC 슬롯은 <strong>전 기간 DBC 프록시</strong>입니다.</p>
          <div className="date-controls"><label>시작일<input type="date" value={profile.backtestStart} onChange={e=>edit({backtestStart:e.target.value})}/></label><label>종료일<input type="date" value={profile.backtestEnd} onChange={e=>edit({backtestEnd:e.target.value})}/></label><label>초기 금액 (USD)<input type="number" min="1" max="1000000000000" value={Number.isNaN(initial)?'':initial} onChange={e=>setInitial(e.target.valueAsNumber)}/></label><button className="quiet" onClick={()=>edit({backtestStart:'',backtestEnd:''})}>최대 기간</button><button className="primary" disabled={loading || demo} onClick={runBacktest}>{loading?'계산 중…':'백테스트 실행'}</button></div>
          <p className="micro">기간을 비워두면 최대 가용 범위를 사용합니다. 12개월 준비기간 이후부터 계산하며, 요청 기간 안의 실제 월말로 조정합니다. 중간 월 누락 시 계산을 중단합니다.</p>
          {result ? <BacktestView result={result}/> : <div className="empty">{loading?'공용 가격 캐시에서 백테스트를 계산하고 있습니다.':'기간을 선택해 장기 성과를 확인하세요. 방문자별 Yahoo 재조회 없이 서버의 공용 캐시를 사용합니다.'}</div>}
          <p className="micro">월말 신호는 다음 달 수익에 적용합니다. 수정주가의 배당·분할을 반영하며 수수료·슬리피지·세금은 제외합니다. 낙폭은 월말 관측 기준입니다. 샤프는 기존 방식인 (CAGR − 선택 기간 ^IRX 평균) / 연환산 변동성입니다. 과거 성과는 미래 수익을 보장하지 않습니다.</p>
        </section>}
        {tab === 'history' && <div className="history-grid"><section className="panel"><h2>월말 확정 신호</h2><p className="micro">운영 시작 이후 저장한 확정본 · 최근 24건 · 신호 기준 티커 표시</p>{history.length?<div className="table-scroll"><table><thead><tr><th>신호 월</th><th>상태</th><th>목표 자산</th></tr></thead><tbody>{history.map(h=><tr key={h.month}><td>{h.month}</td><td>{h.regime}</td><td>{weightsLabel(h.weights)}</td></tr>)}</tbody></table></div>:<p className="empty">저장된 확정 이력이 없습니다.</p>}</section>
          <section className="panel"><h2>나의 리밸런싱 계산안</h2><p className="micro">최근 24건. 저장은 주문 체결을 의미하지 않습니다.</p>{rebalances.length?rebalances.map(r=><details key={r.id}><summary>{r.signalMonth} · {r.sp500} · {money(r.equity)} · {r.kind}</summary><div className="table-scroll"><table><thead><tr><th>종목</th><th>목표</th><th>매매</th><th>사용 가격</th></tr></thead><tbody>{r.lines.map(l=><tr key={l.ticker}><td>{l.ticker}</td><td>{l.target}</td><td>{l.trade}</td><td>{money(l.price)}</td></tr>)}</tbody></table></div></details>):<p className="empty">아직 저장한 계산안이 없습니다.</p>}</section></div>}
      </>}
      <footer><span>HAA 포트폴리오</span><p>개인용 투자 보조 도구 · Yahoo Finance 시세는 지연되거나 수정될 수 있습니다.<br />예상 신호는 확정 신호가 아니며, 자동 주문을 실행하지 않습니다.</p></footer>
    </main>
  </>;
}
