import { useCallback, useEffect, useRef, useState } from 'react';
import type { User } from 'firebase/auth';
import { doc, onSnapshot } from 'firebase/firestore';
import { apiHeaders, db, demo } from './firebase';

export type AccessGrant = { uid: string; email?: string; enabled: boolean; ownerUid?: string; role?: 'admin'|'member'; requestStatus?: string };
export async function accessRequest(path: 'session'|'request'|'manage', body?: object, signal?: AbortSignal) {
  const response=await fetch(`/api/access/${path}`,{method:body?'POST':'GET',headers:{...await apiHeaders(),'Content-Type':'application/json'},
    ...(body?{body:JSON.stringify(body)}:{}),signal:signal??AbortSignal.timeout(20000)});
  if (!response.headers.get('content-type')?.includes('application/json')) throw new Error('가족 권한 서버에 연결하지 못했습니다. 새로고침 후 다시 시도하세요.');
  const result=await response.json();
  if(!response.ok) throw new Error(result.error??'권한 요청을 처리하지 못했습니다.');
  return result;
}
export function useFamilyAccess(user: User|null) {
  const [value,setValue]=useState<AccessGrant|null>(null),[error,setError]=useState(''),[busy,setBusy]=useState(false),[message,setMessage]=useState('');
  const generation=useRef(0);
  const access=value?.uid===user?.uid?value:null;
  const refresh=useCallback(async()=>{
    if(!user||demo) return;
    const current=generation.current;
    setBusy(true);setError('');
    try{
      const result=await accessRequest('session',{});
      // 권한 자체는 실시간 문서만 신뢰한다. 늦게 도착한 응답이 해제 상태를 되돌리지 못한다.
      if(current===generation.current)setValue(old=>({...old,uid:user.uid,enabled:old?.enabled===true,requestStatus:result.requestStatus}));
    }
    catch(e){if(current===generation.current)setError(e instanceof Error?e.message:'권한을 확인하지 못했습니다.');}
    finally{if(current===generation.current)setBusy(false);}
  },[user?.uid]);
  useEffect(()=>{
    generation.current++;setValue(null);setError('');setMessage('');
    if(!user||!db||demo) return;
    let live=true;
    const stop=onSnapshot(doc(db,'access',user.uid),s=>{
      if(!live)return;
      const d=s.data();
      // 권한 해제는 API 응답을 기다리지 않고 구독에서 바로 반영한다.
      setValue(old=>({uid:user.uid,enabled:d?.enabled===true,ownerUid:d?.ownerUid??user.uid,role:d?.role??'member',email:user.email??undefined,requestStatus:old?.requestStatus}));
    },e=>{if(live){setValue(null);setError(e.message);}});
    void refresh();
    return ()=>{live=false;generation.current++;stop();};
  },[user?.uid,refresh]);
  async function requestApproval(){
    if(!user||busy)return;
    const current=generation.current;setBusy(true);setError('');
    try{const result=await accessRequest('request',{});if(current===generation.current)setMessage(result.message);}
    catch(e){if(current===generation.current)setError(e instanceof Error?e.message:'승인 요청에 실패했습니다.');}
    finally{if(current===generation.current)setBusy(false);}
  }
  return {access,busy,error,message,refresh,requestApproval};
}
