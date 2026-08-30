import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { Curve } from './Curve';
import type { Backtest } from './types';

const data: Backtest['curve'] = Array.from({length:5},(_,i)=>({
  date: new Date(Date.UTC(2026,i+1,0)).toISOString().slice(0,10), value:10000+i*100, drawdown:-i/100,
}));
let width=600;
let resized: () => void;
beforeEach(()=>{
  width=600;
  vi.spyOn(Element.prototype,'getBoundingClientRect').mockImplementation(()=>({x:0,y:0,left:0,top:0,right:width,bottom:300,width,height:300,toJSON:()=>({})}));
  vi.stubGlobal('ResizeObserver',class { constructor(cb:()=>void){resized=cb;} observe(){} disconnect(){} });
  vi.stubGlobal('PointerEvent',class extends MouseEvent { pointerType: string; pointerId: number;
    constructor(type:string,init:PointerEventInit={}) {super(type,init);this.pointerType=init.pointerType??'mouse';this.pointerId=init.pointerId??1;}
  });
});
afterEach(()=>{cleanup();vi.restoreAllMocks();vi.unstubAllGlobals();});

describe('대화형 월말 차트',()=>{
  it('마우스를 올린 월의 정확한 날짜·금액·낙폭을 표시한다',()=>{
    render(<Curve data={data}/>);
    fireEvent.pointerMove(screen.getByRole('img'),{clientX:322,pointerType:'mouse'});
    const tip=screen.getByRole('tooltip');
    expect(tip.textContent).toContain('2026년 3월 31일'); expect(tip.textContent).toContain('10,200.00'); expect(tip.textContent).toContain('-2.00%');
    fireEvent.pointerLeave(screen.getByRole('img'),{pointerType:'mouse'});
    expect(screen.queryByRole('tooltip')).toBeNull();
  });
  it('화면을 축소해도 포인터 좌표를 올바르게 환산한다',()=>{
    render(<Curve data={data}/>);
    // SVG가 컨테이너 좌표의 절반 크기로 표시된 상황을 재현한다.
    vi.spyOn(screen.getByRole('img'),'getBoundingClientRect').mockReturnValue({x:0,y:0,left:0,top:0,right:300,bottom:300,width:300,height:300,toJSON:()=>({})});
    fireEvent.pointerMove(screen.getByRole('img'),{clientX:161});
    expect(screen.getByRole('tooltip').textContent).toContain('2026년 3월 31일');
  });
  it('터치로 선택하고 좌우 이동한 월을 손을 뗀 뒤에도 유지한다',()=>{
    render(<Curve data={data}/>);
    const chart=screen.getByRole('img');
    fireEvent.pointerDown(chart,{clientX:62,pointerType:'touch'});
    fireEvent.pointerMove(chart,{clientX:322,pointerType:'touch'});
    fireEvent.pointerUp(chart,{clientX:322,pointerType:'touch'});
    expect((screen.getByRole('slider') as HTMLInputElement).value).toBe('2');
    expect(screen.getByRole('slider').getAttribute('aria-valuetext')).toContain('2026년 3월 31일');
  });
  it('탐색 막대와 버튼으로 각 월을 빠짐없이 탐색한다',()=>{
    render(<Curve data={data}/>);
    fireEvent.change(screen.getByRole('slider'),{target:{value:'0'}});
    expect((screen.getByRole('button',{name:'포트폴리오 가치 이전 월'}) as HTMLButtonElement).disabled).toBe(true);
    fireEvent.click(screen.getByRole('button',{name:'포트폴리오 가치 다음 월'}));
    expect(screen.getByRole('slider').getAttribute('aria-valuetext')).toContain('2026년 2월 28일');
    fireEvent.click(screen.getByRole('button',{name:'포트폴리오 가치 이전 월'}));
    expect((screen.getByRole('slider') as HTMLInputElement).value).toBe('0');
  });
  it('낙폭 차트도 음수 퍼센트와 정확한 평가금액을 표시한다',()=>{
    render(<Curve data={data} drawdown/>);
    fireEvent.focus(screen.getByRole('slider'));
    fireEvent.change(screen.getByRole('slider'),{target:{value:'4'}});
    expect(screen.getByRole('tooltip').textContent).toContain('-4.00%'); expect(screen.getByRole('tooltip').textContent).toContain('10,400.00');
  });
  it('키보드의 처음·끝·화살표 키로 월을 선택한다',()=>{
    render(<Curve data={data}/>);
    const slider=screen.getByRole('slider') as HTMLInputElement;
    fireEvent.keyDown(slider,{key:'Home'}); expect(slider.value).toBe('0');
    fireEvent.keyDown(slider,{key:'ArrowRight'}); expect(slider.value).toBe('1');
    fireEvent.keyDown(slider,{key:'End'}); expect(slider.value).toBe('4');
    fireEvent.keyDown(slider,{key:'ArrowLeft'}); expect(slider.value).toBe('3');
  });
  it('좁은 화면으로 변경하면 날짜 개수를 줄이고 그래프 높이는 유지한다',()=>{
    const longData=Array.from({length:217},(_,i)=>({...data[0],date:new Date(Date.UTC(2008,i+1,0)).toISOString().slice(0,10)}));
    const {container}=render(<Curve data={longData}/>);
    const before=container.querySelectorAll('.curve-month-tick').length;
    width=320; fireEvent(window,new Event('resize'));
    expect(container.querySelectorAll('.curve-month-tick').length).toBeLessThan(before);
    expect(screen.getByRole('img').getAttribute('viewBox')).toBe('0 0 320 252');
    expect(resized).toBeTypeOf('function');
  });
  it('새 기간을 계산하면 이전 선택 월을 초기화한다',()=>{
    const {rerender}=render(<Curve data={data}/>);
    fireEvent.change(screen.getByRole('slider'),{target:{value:'0'}});
    rerender(<Curve data={data.slice(2)}/>);
    expect((screen.getByRole('slider') as HTMLInputElement).value).toBe('2');
    expect(screen.getByRole('slider').getAttribute('aria-valuetext')).toContain('2026년 5월 31일');
  });
  it('빈 데이터·잘못된 값에는 그래프를 꾸며내지 않는다',()=>{
    const {rerender}=render(<Curve data={[]}/>); expect(screen.queryByRole('img')).toBeNull();
    rerender(<Curve data={[{...data[0],value:NaN}]}/>); expect(screen.queryByRole('img')).toBeNull();
  });
  it('단일 관측값과 낙폭 0에도 NaN 좌표를 만들지 않는다',()=>{
    const {container}=render(<Curve data={[data[0]]} drawdown/>);
    expect(container.querySelector('polyline')?.getAttribute('points')).not.toMatch(/NaN|Infinity/);
    expect((screen.getByRole('slider') as HTMLInputElement).disabled).toBe(true);
  });
});
