import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import type { User } from 'firebase/auth';
const fixture=vi.hoisted(()=>({docs:new Map<string,unknown>(),callbacks:new Map<string,(s:unknown)=>void>(),write:vi.fn(),add:vi.fn()}));
vi.mock('./firebase',()=>({demo:false,db:{},apiHeaders:async()=>({Authorization:'Bearer fixture'})}));
vi.mock('firebase/firestore',()=>({
  doc:(_db:unknown,...p:string[])=>p.join('/'),collection:(_db:unknown,...p:string[])=>p.join('/'),query:(p:string)=>p,orderBy:vi.fn(),limit:vi.fn(),serverTimestamp:()=>({seconds:1}),
  onSnapshot:(p:string,cb:(s:unknown)=>void)=>{fixture.callbacks.set(p,cb);cb({exists:()=>fixture.docs.has(p),data:()=>fixture.docs.get(p),docs:[]});return()=>fixture.callbacks.delete(p);},
  setDoc:async(p:string,d:unknown)=>{fixture.write(p,d);fixture.docs.set(p,d);},addDoc:fixture.add,
}));
import { PensionPortfolio } from './PensionPortfolio';
import { defaultPensionProfile, PENSION_CODES } from './pensionModel';
const now=Date.parse('2026-10-26T08:00:00Z');
const market={currency:'KRW',quoteMode:'close',marketDate:'2026-10-26',updatedAt:new Date(now).toISOString(),prices:Object.fromEntries(PENSION_CODES.map(c=>[c,10000])),priceTimes:Object.fromEntries(PENSION_CODES.map(c=>[c,'2026-10-26T06:30:00Z']))};
const show=()=>render(<PensionPortfolio user={{uid:'wife'} as User} ownerUid="owner" now={now}/>);
beforeEach(()=>{fixture.docs.clear();fixture.callbacks.clear();fixture.write.mockReset();fixture.add.mockReset();fixture.docs.set('users/owner/portfolios/pension-v2',{...defaultPensionProfile(),totalBalance:1000000});fixture.docs.set('market/pensionV2',market);vi.stubGlobal('fetch',vi.fn());});
afterEach(()=>{cleanup();vi.unstubAllGlobals();});
it('탭을 열어도 Yahoo 호출 없이 가족 공유 연금 문서를 읽는다',()=>{
  show();expect(fetch).not.toHaveBeenCalled();expect(fixture.callbacks.has('users/owner/portfolios/pension-v2')).toBe(true);expect(fixture.callbacks.has('users/wife/portfolios/pension-v2')).toBe(false);
});
it('보유·비중·직접 목표를 연금 경로에 저장하고 재방문해 복원한다',async()=>{
  const view=show();fireEvent.change(screen.getByLabelText('연금 379800 보유수량'),{target:{value:'5'}});fireEvent.change(screen.getByLabelText('연금 379800 목표 보유수량'),{target:{value:'0'}});
  fireEvent.click(screen.getByRole('button',{name:'연금 보유수량·설정 저장'}));
  await waitFor(()=>expect(fixture.write).toHaveBeenCalledTimes(1));expect(fixture.write.mock.calls[0][0]).toBe('users/owner/portfolios/pension-v2');
  expect(fixture.write.mock.calls[0][1].targetOverrides['379800']).toBe(0);
  view.unmount();show();expect((screen.getByLabelText('연금 379800 목표 보유수량') as HTMLInputElement).value).toBe('0');
});
it('저장 데이터가 유효하지 않으면 초기값으로 덮어쓰지 않는다',()=>{
  fixture.docs.set('users/owner/portfolios/pension-v2',{invalid:true});show();expect((screen.getByRole('button',{name:'연금 보유수량·설정 저장'}) as HTMLButtonElement).disabled).toBe(true);
});
it('종가 계산안은 확인 후에만 별도 이력에 저장한다',async()=>{
  show();const save=screen.getByRole('button',{name:'연금 계산안 저장'}) as HTMLButtonElement;expect(save.disabled).toBe(true);
  fireEvent.click(screen.getByRole('checkbox'));fireEvent.click(save);
  await waitFor(()=>expect(fixture.add).toHaveBeenCalledTimes(1));expect(fixture.add.mock.calls[0][0]).toBe('users/owner/portfolios/pension-v2/rebalances');
  expect(fixture.add.mock.calls[0][1]).toMatchObject({currency:'KRW',month:'2026-10',equity:1000000});expect(fixture.add.mock.calls[0][1].lines.slice(0,3)).toEqual([0,10000,12]);expect(fixture.write).not.toHaveBeenCalled();
});
it('비중 합계가 틀리면 계산안을 저장하지 못한다',()=>{
  show();fireEvent.change(screen.getByLabelText('연금 379800 목표 비중 (%)'),{target:{value:'25'}});expect(screen.queryByRole('button',{name:'연금 계산안 저장'})).toBeNull();expect(screen.getByRole('alert').textContent).toContain('100%');
});
it('인증된 수동 갱신에서 fallback 안내를 표시한다',async()=>{
  vi.mocked(fetch).mockResolvedValue({ok:true,headers:new Headers({'content-type':'application/json'}),json:async()=>({market,source:'fallback',message:'직전 성공값입니다.'})} as Response);
  show();fireEvent.click(screen.getByRole('button',{name:'연금 시세 갱신'}));await screen.findByText('직전 성공값입니다.');
  expect(fetch).toHaveBeenCalledWith('/api/pension/refresh',expect.objectContaining({method:'POST',headers:expect.objectContaining({Authorization:'Bearer fixture'})}));
});
it('화면을 나가면 요청을 중단하고 구독을 해제한다',async()=>{
  let signal:AbortSignal|undefined;vi.mocked(fetch).mockImplementation(async(_url,opts)=>{signal=opts?.signal as AbortSignal;return new Promise(()=>{});});
  const view=show();fireEvent.click(screen.getByRole('button',{name:'연금 시세 갱신'}));await act(async()=>{});view.unmount();expect(signal?.aborted).toBe(true);expect(fixture.callbacks.size).toBe(0);
});

it('저장된 36개 숫자 이력을 종목별 보유·가격·목표로 복원한다',()=>{
  show();act(()=>fixture.callbacks.get('users/owner/portfolios/pension-v2/rebalances')?.({docs:[{id:'saved',data:()=>({month:'2026-10',equity:1000000,kind:'계산안',weights:defaultPensionProfile().weights,lines:PENSION_CODES.flatMap(()=>[5,10000,12]),priceUpdatedAt:market.updatedAt,oldestPriceAt:market.updatedAt})}]}));
  fireEvent.click(screen.getByRole('button',{name:'연금 저장 이력'}));
  expect(screen.getAllByText('5 → 12')).toHaveLength(12);
});
