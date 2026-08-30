import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { User } from 'firebase/auth';
const fixture=vi.hoisted(()=>({docs:new Map<string,unknown>(),callbacks:new Map<string,(s:unknown)=>void>(),writes:vi.fn(),adds:vi.fn()}));
vi.mock('./firebase',()=>({demo:false,db:{},apiHeaders:async()=>({Authorization:'Bearer fixture'})}));
vi.mock('firebase/firestore',()=>({
  doc:(_db:unknown,...parts:string[])=>parts.join('/'),collection:(_db:unknown,...parts:string[])=>parts.join('/'),
  query:(ref:string)=>ref,orderBy:vi.fn(),limit:vi.fn(),serverTimestamp:()=>({seconds:1}),
  onSnapshot:(ref:string,callback:(s:unknown)=>void)=>{
    fixture.callbacks.set(ref,callback);
    callback({exists:()=>fixture.docs.has(ref),data:()=>fixture.docs.get(ref),docs:[]});
    return ()=>fixture.callbacks.delete(ref);
  },
  setDoc:async(ref:string,data:unknown)=>{fixture.writes(ref,data);fixture.docs.set(ref,data);},
  addDoc:async(ref:string,data:unknown)=>{fixture.adds(ref,data);return{id:'saved'};},
}));
import { SujinPortfolio } from './SujinPortfolio';
import { defaultSujinProfile, SUJIN_TICKERS } from './sujinModel';
const user={uid:'owner'} as User;
const now=Date.parse('2026-08-28T15:00:00Z');
const market={prices:Object.fromEntries(SUJIN_TICKERS.map(t=>[t,100])),priceTimes:Object.fromEntries(SUJIN_TICKERS.map(t=>[t,new Date(now).toISOString()])),updatedAt:new Date(now).toISOString()};
const show=()=>render(<SujinPortfolio user={user} now={now} renderBacktest={r=><div>검증 결과 {r.actualStart}</div>}/>);
beforeEach(()=>{
  fixture.docs.clear();fixture.callbacks.clear();fixture.writes.mockClear();fixture.adds.mockClear();
  fixture.docs.set('users/owner/portfolios/sujin',{...defaultSujinProfile(),totalBalance:10000});fixture.docs.set('market/sujin',market);
  vi.stubGlobal('fetch',vi.fn());
});
afterEach(()=>{cleanup();vi.unstubAllGlobals();});
describe('수진 저장·갱신 화면',()=>{
  it('아내 계정도 서버가 승인한 현근 소유 경로에 저장한다',async()=>{
    render(<SujinPortfolio user={{uid:'wife'} as User} ownerUid="owner" now={now} renderBacktest={()=>null}/>);
    fireEvent.change(screen.getByLabelText('수진 QQQM 보유수량'),{target:{value:'8'}});
    fireEvent.click(screen.getByRole('button',{name:'수진 보유수량·설정 저장'}));
    await waitFor(()=>expect(fixture.writes).toHaveBeenCalledTimes(1));
    expect(fixture.writes.mock.calls[0][0]).toBe('users/owner/portfolios/sujin');
    expect(fixture.callbacks.has('users/wife/portfolios/sujin')).toBe(false);
  });
  it('탭을 열기만 하면 Yahoo 서버를 호출하지 않는다',()=>{show();expect(fetch).not.toHaveBeenCalled();});
  it('저장 위치가 현근 문서와 분리되며 다시 열면 수량을 복원한다',async()=>{
    const view=show();fireEvent.change(screen.getByLabelText('수진 QQQM 보유수량'),{target:{value:'7'}});
    fireEvent.click(screen.getByRole('button',{name:'수진 보유수량·설정 저장'}));
    await waitFor(()=>expect(fixture.writes).toHaveBeenCalledTimes(1));
    expect(fixture.writes.mock.calls[0][0]).toBe('users/owner/portfolios/sujin');
    view.unmount();show();expect((screen.getByLabelText('수진 QQQM 보유수량') as HTMLInputElement).value).toBe('7');
  });
  it('오래된 시세는 확인 전 계산안 저장을 막고 가격 시각을 기록한다',async()=>{
    fixture.docs.set('status/sujinQuotes',{ok:false});show();
    const button=screen.getByRole('button',{name:'수진 계산안 저장 · 주문 아님'}) as HTMLButtonElement;
    expect(button.disabled).toBe(true);
    fireEvent.click(screen.getByRole('checkbox'));fireEvent.click(button);
    await waitFor(()=>expect(fixture.adds).toHaveBeenCalledTimes(1));
    expect(fixture.adds.mock.calls[0][0]).toBe('users/owner/portfolios/sujin/rebalances');
    expect(fixture.adds.mock.calls[0][1]).toMatchObject({kind:'계산안',priceUpdatedAt:market.updatedAt,month:'2026-08'});
    expect(Object.keys(fixture.adds.mock.calls[0][1].lines[0]).sort()).toEqual(['held','price','target','ticker','trade']);
    expect(fixture.writes).not.toHaveBeenCalled();
  });
  it('수동 갱신은 인증된 POST를 한 번 보내고 fallback 안내를 표시한다',async()=>{
    vi.mocked(fetch).mockResolvedValue({ok:true,headers:new Headers({'content-type':'application/json'}),json:async()=>({market,source:'fallback',message:'직전 성공값을 표시합니다.'})} as Response);
    show();fireEvent.click(screen.getByRole('button',{name:'시세 수동 갱신'}));
    await screen.findByText('직전 성공값을 표시합니다.');
    expect(fetch).toHaveBeenCalledWith('/api/sujin/refresh',expect.objectContaining({method:'POST',headers:expect.objectContaining({Authorization:'Bearer fixture'})}));
  });
  it('서버 오류를 성공 결과로 표시하지 않는다',async()=>{
    vi.mocked(fetch).mockResolvedValue({ok:false,headers:new Headers({'content-type':'application/json'}),json:async()=>({error:'캐시가 없습니다.'})} as Response);
    show();fireEvent.click(screen.getByRole('button',{name:'시세 수동 갱신'}));
    expect((await screen.findByRole('alert')).textContent).toContain('캐시가 없습니다.');
  });
  it('수진 백테스트 경로에 기간을 보내고 현근 경로를 호출하지 않는다',async()=>{
    vi.mocked(fetch).mockResolvedValue({ok:true,headers:new Headers({'content-type':'application/json'}),json:async()=>({actualStart:'2023-01-31'})} as Response);
    show();fireEvent.click(screen.getByRole('button',{name:'장기 백테스트'}));
    fireEvent.change(screen.getByLabelText('시작일'),{target:{value:'2023-01-01'}});
    fireEvent.click(screen.getByRole('button',{name:'수진 백테스트 실행'}));
    await screen.findByText('검증 결과 2023-01-31');
    expect(fetch).toHaveBeenCalledWith('/api/sujin/backtest',expect.objectContaining({body:'{"initial":10000,"start":"2023-01-01"}'}));
  });
  it('저장 데이터 미수신 중에는 덮어쓰기 저장을 허용하지 않는다',()=>{
    fixture.docs.set('users/owner/portfolios/sujin',{invalid:true});show();
    expect((screen.getByRole('button',{name:'수진 보유수량·설정 저장'}) as HTMLButtonElement).disabled).toBe(true);
  });
  it('나간 화면의 요청은 중단하고 구독을 해제한다',async()=>{
    let signal: AbortSignal|undefined;
    vi.mocked(fetch).mockImplementation(async(_url,options)=>{signal=options?.signal as AbortSignal;return new Promise(()=>{});});
    const view=show();fireEvent.click(screen.getByRole('button',{name:'시세 수동 갱신'}));
    await act(async()=>{});view.unmount();expect(signal?.aborted).toBe(true);expect(fixture.callbacks.size).toBe(0);
  });
});
