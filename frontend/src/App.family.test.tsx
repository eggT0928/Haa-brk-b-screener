import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
const fixture=vi.hoisted(()=>({role:'member',enabled:true,docs:new Map<string,unknown>(),paths:[] as string[],write:vi.fn(),add:vi.fn()}));
vi.mock('./firebase',()=>({demo:false,configured:true,auth:{},db:{},login:vi.fn(),logout:vi.fn(),apiHeaders:vi.fn()}));
vi.mock('firebase/auth',()=>({onAuthStateChanged:(_auth:unknown,cb:(u:unknown)=>void)=>{cb({uid:'wife',email:'wife@example.com'});return()=>{};}}));
vi.mock('./familyAccess',()=>({useFamilyAccess:()=>({access:{enabled:fixture.enabled,role:fixture.role,ownerUid:'owner'},busy:false,error:'',message:'',refresh:vi.fn(),requestApproval:vi.fn()})}));
vi.mock('./FamilyAccessPanel',()=>({FamilyAccessPanel:()=> <div>관리 화면</div>}));
vi.mock('firebase/firestore',()=>({
  doc:(_db:unknown,...p:string[])=>p.join('/'),collection:(_db:unknown,...p:string[])=>p.join('/'),query:(p:string)=>p,orderBy:vi.fn(),limit:vi.fn(),serverTimestamp:()=>({seconds:1}),
  onSnapshot:(p:string,cb:(s:unknown)=>void)=>{fixture.paths.push(p);cb({exists:()=>fixture.docs.has(p),data:()=>fixture.docs.get(p),docs:[]});return()=>{};},
  setDoc:fixture.write,addDoc:fixture.add,
}));
import App from './App';
import { defaultProfile } from './portfolio';
import { demoConfirmed, demoMarket } from './demo';
beforeEach(()=>{fixture.docs.clear();fixture.paths=[];fixture.write.mockReset();fixture.add.mockReset();fixture.role='member';fixture.enabled=true;fixture.docs.set('users/owner',{...defaultProfile(),cash:123});});
afterEach(cleanup);
it('아내 계정은 기존 현근 보유수량을 읽고 같은 경로에 저장한다',async()=>{
  render(<App/>);expect((screen.getByLabelText('현금 잔액 (USD)') as HTMLInputElement).value).toBe('123');
  fireEvent.change(screen.getByLabelText('현금 잔액 (USD)'),{target:{value:'125'}});
  fireEvent.click(screen.getByRole('button',{name:'보유수량·설정 저장'}));
  await waitFor(()=>expect(fixture.write).toHaveBeenCalledWith('users/owner',expect.objectContaining({cash:125})));
  expect(fixture.paths).not.toContain('users/wife');
  expect(screen.queryByRole('button',{name:'가족·권한 관리'})).toBeNull();
});
it('관리자만 가족 관리 화면을 열 수 있다',()=>{
  fixture.role='admin';render(<App/>);
  fireEvent.click(screen.getByRole('button',{name:'가족·권한 관리'}));
  expect(screen.getByText('관리 화면')).toBeTruthy();
  expect(screen.queryByRole('spinbutton',{name:'현금 잔액 (USD)'})).toBeNull();
});
it('접근 해제 시 포트폴리오를 숨기고 승인 요청만 제공한다',()=>{
  fixture.enabled=false;render(<App/>);
  expect(screen.queryByRole('spinbutton',{name:'현금 잔액 (USD)'})).toBeNull();
  expect(screen.getByRole('button',{name:'관리자에게 승인 요청'})).toBeTruthy();
  expect(fixture.paths).not.toContain('users/owner');
});
it('기존 HAA 설정의 누락 단위는 정수로 표시하고 소수점 선택을 공유 문서에 저장한다',async()=>{
  const legacy=defaultProfile();delete legacy.quantityMode;fixture.docs.set('users/owner',legacy);
  render(<App/>);
  expect((screen.getByLabelText('HAA 수량 단위') as HTMLSelectElement).value).toBe('whole');
  fireEvent.change(screen.getByLabelText('HAA 수량 단위'),{target:{value:'fractional'}});
  fireEvent.click(screen.getByRole('button',{name:'보유수량·설정 저장'}));
  await waitFor(()=>expect(fixture.write).toHaveBeenCalledWith('users/owner',expect.objectContaining({quantityMode:'fractional'})));
});
it('계산안 이력에도 당시 수량 단위를 저장한다',async()=>{
  fixture.docs.set('users/owner',{...defaultProfile(),quantityMode:'fractional'});
  fixture.docs.set('signals/confirmed',demoConfirmed);fixture.docs.set('market/current',demoMarket);
  render(<App/>);
  fireEvent.click(screen.getByRole('checkbox',{name:'지연 또는 직전 성공 시세인 점을 확인했습니다.'}));
  fireEvent.click(screen.getByRole('button',{name:'계산안 이력 저장 · 주문 아님'}));
  await waitFor(()=>expect(fixture.add).toHaveBeenCalledWith('users/owner/rebalances',expect.objectContaining({quantityMode:'fractional',lines:expect.arrayContaining([expect.objectContaining({ticker:'SPY',target:3.333333})])})));
});
