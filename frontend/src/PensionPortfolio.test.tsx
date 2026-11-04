import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
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
import { defaultPensionProfile, PENSION_ASSETS, PENSION_CODES } from './pensionModel';
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
  fireEvent.click(screen.getByLabelText('연금 계산안이 종가·지연 또는 직전 성공 시세 기준임을 확인했습니다.'));fireEvent.click(save);
  await waitFor(()=>expect(fixture.add).toHaveBeenCalledTimes(1));expect(fixture.add.mock.calls[0][0]).toBe('users/owner/portfolios/pension-v2/rebalances');
  expect(fixture.add.mock.calls[0][1]).toMatchObject({currency:'KRW',month:'2026-10',equity:1000000});expect(fixture.add.mock.calls[0][1].lines.slice(0,3)).toEqual([0,10000,12]);expect(fixture.write).not.toHaveBeenCalled();
});
it('비중 합계가 틀리면 계산안을 저장하지 못한다',()=>{
  show();fireEvent.change(screen.getByLabelText('연금 379800 목표 비중 (%)'),{target:{value:'25'}});expect((screen.getByRole('button',{name:'연금 계산안 저장'}) as HTMLButtonElement).disabled).toBe(true);expect(screen.getByRole('alert').textContent).toContain('100%');
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

it('현재 비중·목표 대비 차이와 금·예수금을 매매안 오류와 별도로 보여준다',()=>{
  show();
  fireEvent.change(screen.getByLabelText('연금 379800 보유수량'),{target:{value:'20'}});
  fireEvent.change(screen.getByLabelText('연금 0072R0 보유수량'),{target:{value:'3'}});
  fireEvent.change(screen.getByLabelText('연금 411060 보유수량'),{target:{value:'7'}});
  const table=within(screen.getByRole('region',{name:'연금 현재 보유 비중'}));
  expect(within(table.getByRole('row',{name:/KODEX 미국S&P500/})).getByText('20.00%')).toBeTruthy();
  expect(within(table.getByRole('row',{name:/KODEX 미국S&P500/})).getByText('+8.00%p')).toBeTruthy();
  expect(within(table.getByRole('row',{name:/금 ETF 합계/})).getByText('-10.00%p')).toBeTruthy();
  expect(within(table.getByRole('row',{name:/현재 예수금/})).getByText('70.00%')).toBeTruthy();
  fireEvent.change(screen.getByLabelText('연금 379800 목표 보유수량'),{target:{value:'100'}});
  expect(screen.getByRole('alert').textContent).toContain('총자산을 초과');
  expect(within(table.getByRole('row',{name:/금 ETF 합계/})).getByText('-10.00%p')).toBeTruthy();
});

it('기존 총자산보다 평가액이 큰 손실 상태에서도 조절안과 0원 가정을 표시하고 예수금을 저장한다',async()=>{
  const p={...defaultPensionProfile(),totalBalance:500000,principal:1500000};p.holdings['379800']=100;
  fixture.docs.set('users/owner/portfolios/pension-v2',p);const view=show();
  expect(screen.getByText(/예수금을 0원으로 가정/)).toBeTruthy();
  expect(screen.getByText('-₩500,000')).toBeTruthy();
  expect(screen.getByRole('button',{name:'연금 계산안 저장'})).toBeTruthy();
  fireEvent.change(screen.getByLabelText('연금 예수금 (원)'),{target:{value:'200000'}});
  expect(screen.queryByText(/예수금을 0원으로 가정/)).toBeNull();expect(screen.getByText('-₩300,000')).toBeTruthy();
  fireEvent.click(screen.getByRole('button',{name:'연금 보유수량·설정 저장'}));
  await waitFor(()=>expect(fixture.write).toHaveBeenCalledTimes(1));
  expect(fixture.write.mock.calls[0][1].cashBalance).toBe(200000);
  expect(fixture.write.mock.calls[0][1].totalBalance).toBe(500000);
  view.unmount();show();expect((screen.getByLabelText('연금 예수금 (원)') as HTMLInputElement).value).toBe('200000');
});

it('리밸런싱 계산안에 12개 ETF의 이름과 코드가 함께 표시된다',()=>{
  show();
  const section=screen.getByRole('heading',{name:'연금 리밸런싱 계산안'}).closest('section')!;
  for(const asset of PENSION_ASSETS){
    const row=within(section).getByRole('row',{name:new RegExp(asset.code)});
    expect(row.textContent).toContain(asset.name);
    expect(row.textContent).toContain(asset.code);
  }
});

it('정수와 소수점 보기를 전환하면 목표·매매수량과 잔액이 일관되게 바뀌고 입력값은 보존된다',()=>{
  show();
  const section=screen.getByRole('heading',{name:'연금 리밸런싱 계산안'}).closest('section')!;
  const select=within(section).getByLabelText('연금 수량 표시 방식');
  const gold=()=>within(section).getByRole('row',{name:/0072R0/});
  expect(gold().textContent).toContain('0 → 7');
  expect(within(section).getByText('리밸런싱 후 예상 예수금').parentElement!.textContent).toContain('₩10,000');
  fireEvent.change(select,{target:{value:'fractional'}});
  expect(gold().textContent).toContain('0 → 7.69');expect(gold().textContent).toContain('+7.69');
  expect(within(section).getByText('소수점 계산상 잔액').parentElement!.textContent).toContain('₩100');
  expect(within(section).queryByRole('button',{name:/버림|올림/})).toBeNull();
  expect((within(section).getByRole('button',{name:'연금 계산안 저장'}) as HTMLButtonElement).disabled).toBe(true);
  expect((within(section).getByRole('button',{name:'연금 CSV 다운로드'}) as HTMLButtonElement).disabled).toBe(true);
  fireEvent.change(screen.getByLabelText('연금 0072R0 목표 보유수량'),{target:{value:'6'}});
  expect(gold().textContent).toContain('0 → 6.00');
  fireEvent.change(select,{target:{value:'whole'}});
  expect(gold().textContent).toContain('0 → 6');
  expect((screen.getByLabelText('연금 0072R0 목표 보유수량') as HTMLInputElement).value).toBe('6');
  expect(fixture.write).not.toHaveBeenCalled();expect(fixture.add).not.toHaveBeenCalled();
});

it('한 표의 같은 종목 행에서 수량 편집과 현재·조정 후 결과를 함께 확인한다',()=>{
  show();const section=screen.getByRole('heading',{name:'연금 리밸런싱 계산안'}).closest('section')!;
  const row=()=>within(section).getByRole('row',{name:/379800/});
  fireEvent.change(within(row()).getByLabelText('연금 379800 보유수량'),{target:{value:'5'}});
  expect(row().textContent).toContain('5.00%');expect(row().textContent).toContain('+7');
  fireEvent.change(within(row()).getByLabelText('연금 379800 목표 보유수량'),{target:{value:'7'}});
  expect(row().textContent).toContain('+2');expect(row().textContent).toContain('₩70,000');expect(row().textContent).toContain('7.00%');
  expect(within(row()).getByLabelText('연금 379800 목표 비중 (%)')).toBeTruthy();
});
it('목표수량이 총자산을 넘으면 오류를 표시하면서 표에서 바로 수정할 수 있다',()=>{
  show();const field=screen.getByLabelText('연금 379800 목표 보유수량');
  fireEvent.change(field,{target:{value:'101'}});
  expect(screen.getByRole('alert').textContent).toContain('총자산을 초과');
  expect(screen.getByLabelText('연금 379800 목표 보유수량')).toBeTruthy();
  fireEvent.change(field,{target:{value:'0'}});
  expect(screen.queryByRole('alert')).toBeNull();
});

it('휴대폰 간편 보기와 전체 계산을 전환해도 입력한 수량과 목표를 유지한다',()=>{
  show();
  const row=within(screen.getByRole('row',{name:/379800/}));
  fireEvent.change(row.getByLabelText('연금 379800 보유수량'),{target:{value:'5'}});
  fireEvent.change(row.getByLabelText('연금 379800 목표 보유수량'),{target:{value:'7'}});
  const view=within(screen.getByRole('group',{name:'연금 표 보기'}));
  fireEvent.click(view.getByRole('button',{name:'전체 계산'}));
  expect(view.getByRole('button',{name:'전체 계산'}).getAttribute('aria-pressed')).toBe('true');
  expect(screen.getByRole('region',{name:'연금 현재 보유 비중'}).getAttribute('data-mobile-view')).toBe('all');
  expect((row.getByLabelText('연금 379800 보유수량') as HTMLInputElement).value).toBe('5');
  expect(row.getByText('+2')).toBeTruthy();
  fireEvent.click(view.getByRole('button',{name:'간편 보기'}));
  expect(view.getByRole('button',{name:'간편 보기'}).getAttribute('aria-pressed')).toBe('true');
  expect(screen.getByRole('region',{name:'연금 현재 보유 비중'}).getAttribute('data-mobile-view')).toBe('simple');
  expect((row.getByLabelText('연금 379800 목표 보유수량') as HTMLInputElement).value).toBe('7');
  expect(fixture.write).not.toHaveBeenCalled();
});
