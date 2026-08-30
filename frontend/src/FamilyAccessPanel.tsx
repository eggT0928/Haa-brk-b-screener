import { useEffect, useRef, useState } from 'react';
import { accessRequest } from './familyAccess';
import { timeLabel } from './portfolio';
import './family.css';

type Member={id:string;email:string;enabled:boolean;role:string};
type Invite={id:string;email:string;status:string;expiresAt:string};
type Request={id:string;email:string;status:string};
type Management={members:Member[];invites:Invite[];requests:Request[]};
type Action={action:string;email?:string;uid?:string};
export function FamilyAccessPanel(){
  const [data,setData]=useState<Management|null>(null),[email,setEmail]=useState(''),[consent,setConsent]=useState(false);
  const [busy,setBusy]=useState(false),[error,setError]=useState(''),[message,setMessage]=useState(''),[pending,setPending]=useState<{body:Action;label:string}|null>(null);
  const live=useRef(true);
  const link=window.location.origin+'/';
  const fail=(e:unknown)=>e instanceof Error?e.message:'처리하지 못했습니다.';
  async function reload(){
    setBusy(true);setError('');
    try{const result=await accessRequest('manage');if(live.current)setData(result);}
    catch(e){if(live.current)setError(fail(e));}
    finally{if(live.current)setBusy(false);}
  }
  useEffect(()=>{live.current=true;void reload();return()=>{live.current=false;};},[]);
  async function apply(body:Action){
    setBusy(true);setError('');setMessage('');
    try{
      const result=await accessRequest('manage',body);
      if(!live.current)return;
      setMessage(result.message);setPending(null);
      if(body.action==='invite'){setEmail('');setConsent(false);}
      const updated=await accessRequest('manage');
      if(live.current)setData(updated);
    }catch(e){if(live.current)setError(fail(e));}
    finally{if(live.current)setBusy(false);}
  }
  return <section className="panel family-panel" aria-label="가족 초대 및 권한 관리">
    <div className="section-title"><h2>가족·권한 관리</h2><button onClick={reload} disabled={busy}>목록 새로고침</button></div>
    <p>관리자만 초대·승인·접근 해제를 할 수 있습니다. 구성원은 현근·수진의 보유수량과 이력을 함께 보고 수정할 수 있지만, 다른 사람을 승인할 수는 없습니다.</p>
    <p className="notice">Firebase 프로젝트 소유자 권한과는 별개입니다. 여기서 초대해도 Firebase 관리 권한은 부여하지 않습니다.</p>
    {error&&<p className="error" role="alert">{error}</p>}{message&&<p className="success" role="status">{message}</p>}
    <form className="family-invite" onSubmit={e=>{e.preventDefault();if(consent&&!busy)void apply({action:'invite',email});}}>
      <label>초대할 Google 이메일<input type="email" required maxLength={254} value={email} onChange={e=>setEmail(e.target.value)} disabled={busy} placeholder="example@gmail.com"/></label>
      <label className="check"><input type="checkbox" checked={consent} onChange={e=>setConsent(e.target.checked)} disabled={busy}/>이 계정에 두 포트폴리오의 조회·편집 권한을 허용합니다.</label>
      <button className="primary" disabled={busy||!consent||!email}>이메일 초대 등록</button>
    </form>
    <p className="micro">초대는 14일간 유효합니다. 이메일은 자동 발송되지 않습니다. 아래 링크를 전달하면 등록된 이메일의 Google 로그인 시 자동 승인됩니다.</p>
    <div className="family-link"><a href={link}>{link}</a><button onClick={async()=>{try{await navigator.clipboard.writeText(link);setMessage('초대용 앱 링크를 복사했습니다.');}catch{setError('복사하지 못했습니다. 표시된 링크를 직접 복사하세요.');}}}>앱 링크 복사</button></div>
    {pending&&<div className="notice" role="group" aria-label="권한 변경 확인"><p>{pending.label} 진행할까요? 접근 해제·초대 취소는 포트폴리오 데이터를 삭제하지 않습니다.</p><div className="family-actions"><button disabled={busy} onClick={()=>apply(pending.body)}>변경 확인</button><button disabled={busy} onClick={()=>setPending(null)}>취소</button></div></div>}
    <h3>승인 요청</h3>
    {data?.requests.filter(r=>r.status==='pending').length?data.requests.filter(r=>r.status==='pending').map(r=><div className="family-row" key={r.id}><span>{r.email}</span><div className="family-actions"><button disabled={busy} onClick={()=>setPending({body:{action:'approve',uid:r.id},label:`${r.email} 계정에 공동 편집 권한을 승인합니다.`})}>승인</button><button disabled={busy} onClick={()=>setPending({body:{action:'reject',uid:r.id},label:`${r.email} 계정의 요청을 거절합니다.`})}>거절</button></div></div>):<p className="micro">대기 중인 승인 요청이 없습니다.</p>}
    <h3>구성원</h3>
    {data?.members.map(m=><div className="family-row" key={m.id}><div><strong>{m.email}</strong><p className="micro">{m.role==='admin'?'관리자':m.enabled?'구성원 · 공동 편집':'접근 해제됨'}</p></div>{m.role!=='admin'&&m.enabled&&<button disabled={busy} onClick={()=>setPending({body:{action:'revoke',uid:m.id},label:`${m.email} 계정의 접근을 해제합니다.`})}>접근 해제</button>}</div>)}
    <h3>등록한 초대</h3>
    {data?.invites.length?data.invites.map(i=><div className="family-row" key={i.id}><div><strong>{i.email}</strong><p className="micro">{({pending:Date.parse(i.expiresAt)<Date.now()?'만료됨':'로그인 대기',accepted:'가입 완료',cancelled:'취소됨',revoked:'접근 해제됨'} as Record<string,string>)[i.status]??i.status} · 만료 {timeLabel(i.expiresAt)}</p></div>{i.status==='pending'&&<button disabled={busy} onClick={()=>setPending({body:{action:'cancel',email:i.email},label:`${i.email} 초대를 취소합니다.`})}>초대 취소</button>}</div>):<p className="micro">등록한 초대가 없습니다.</p>}
    <p className="micro">해제된 구성원을 다시 허용하려면 이메일로 재초대하세요. 관리자 본인의 권한은 이 화면에서 해제할 수 없습니다.</p>
  </section>;
}
