import { act, cleanup, renderHook, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import type { User } from 'firebase/auth';
const fixture=vi.hoisted(()=>({callbacks:new Map<string,(s:unknown)=>void>()}));
vi.mock('./firebase',()=>({demo:false,db:{},apiHeaders:async()=>({Authorization:'Bearer fixture'})}));
vi.mock('firebase/firestore',()=>({doc:(_db:unknown,...parts:string[])=>parts.join('/'),onSnapshot:(ref:string,cb:(s:unknown)=>void)=>{fixture.callbacks.set(ref,cb);return()=>fixture.callbacks.delete(ref);}}));
import { useFamilyAccess } from './familyAccess';
const user={uid:'wife',email:'wife@example.com'} as User;
const response=(data:object,ok=true)=>({ok,headers:new Headers({'content-type':'application/json'}),json:async()=>data}) as Response;
const emit=(data:unknown,uid='wife')=>act(()=>fixture.callbacks.get('access/'+uid)?.({data:()=>data}));
beforeEach(()=>{fixture.callbacks.clear();vi.stubGlobal('fetch',vi.fn().mockResolvedValue(response({enabled:false,requestStatus:'pending'})));});
afterEach(()=>{cleanup();vi.unstubAllGlobals();});

it('로그인 시 초대를 확인하지만 승인 요청은 사용자가 눌러야 전송한다',async()=>{
  const view=renderHook(()=>useFamilyAccess(user));
  await waitFor(()=>expect(view.result.current.busy).toBe(false));
  expect(fetch).toHaveBeenCalledTimes(1);
  expect(fetch).toHaveBeenCalledWith('/api/access/session',expect.objectContaining({method:'POST'}));
  await act(()=>view.result.current.requestApproval());
  expect(fetch).toHaveBeenLastCalledWith('/api/access/request',expect.objectContaining({method:'POST'}));
});
it('공유 대상과 권한은 서버 문서 구독으로만 결정한다',async()=>{
  vi.mocked(fetch).mockResolvedValue(response({uid:'wife',enabled:true,ownerUid:'forged-response',role:'admin'}));
  const view=renderHook(()=>useFamilyAccess(user));
  await waitFor(()=>expect(view.result.current.busy).toBe(false));
  expect(view.result.current.access?.enabled).toBe(false);
  emit({enabled:true,ownerUid:'owner',role:'member'});
  expect(view.result.current.access).toMatchObject({enabled:true,ownerUid:'owner',role:'member'});
});
it('늦은 세션 응답이 실시간 접근 해제를 되돌릴 수 없다',async()=>{
  let finish!:(res:Response)=>void;
  vi.mocked(fetch).mockReturnValue(new Promise(resolve=>{finish=resolve;}));
  const view=renderHook(()=>useFamilyAccess(user));
  emit({enabled:true,ownerUid:'owner',role:'member'});
  emit({enabled:false,ownerUid:'owner',role:'member'});
  await act(async()=>finish(response({uid:'wife',enabled:true,ownerUid:'owner',role:'member'})));
  expect(view.result.current.access?.enabled).toBe(false);
});
it('계정 전환 시 이전 세션의 권한과 응답을 재사용하지 않는다',async()=>{
  let finish!:(res:Response)=>void;
  vi.mocked(fetch).mockReturnValueOnce(new Promise(resolve=>{finish=resolve;}));
  const view=renderHook(({current})=>useFamilyAccess(current),{initialProps:{current:user as User|null}});
  emit({enabled:true,ownerUid:'owner',role:'member'});
  view.rerender({current:null});
  await act(async()=>finish(response({uid:'wife',enabled:true})));
  expect(view.result.current.access).toBeNull();expect(fixture.callbacks.size).toBe(0);
});
it('서버 실패는 승인 성공으로 표시하지 않는다',async()=>{
  vi.mocked(fetch).mockResolvedValue(response({error:'Google 계정 확인 필요'},false));
  const view=renderHook(()=>useFamilyAccess(user));
  await waitFor(()=>expect(view.result.current.error).toBe('Google 계정 확인 필요'));
  expect(view.result.current.access?.enabled).not.toBe(true);
});
