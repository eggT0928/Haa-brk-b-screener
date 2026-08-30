import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
vi.mock('./firebase', () => ({ demo: true, configured: false, auth: null, db: null, login: vi.fn(), logout: vi.fn(), apiHeaders: vi.fn() }));
import App from './App';

afterEach(cleanup);
describe('한국어 웹앱', () => {
  it('샘플임을 명시하고 확정과 예상을 분리한다', () => {
    render(<App />);
    expect(screen.getByText(/샘플 데이터입니다/)).toBeTruthy();
    expect(screen.getByRole('heading',{name:'확정 신호'})).toBeTruthy();
    expect(screen.getByRole('heading',{name:'예상 신호'})).toBeTruthy();
    expect((screen.getByRole('button',{name:'보유수량·설정 저장'}) as HTMLButtonElement).disabled).toBe(true);
  });
  it('실전 ETF 변경과 보유수량 입력이 계산에 반영된다', () => {
    render(<App />);
    fireEvent.change(screen.getByLabelText('실전 S&P500 ETF'),{target:{value:'SPYM'}});
    expect(screen.getByText('0 → 28')).toBeTruthy();
    fireEvent.change(screen.getByLabelText('SPYM 보유수량'),{target:{value:'10'}});
    expect(screen.getByText('10 → 30')).toBeTruthy();
  });
  it('백테스트 기간과 이력 화면을 전환한다', () => {
    render(<App />);
    fireEvent.click(screen.getByRole('button',{name:'장기 백테스트'}));
    const input = screen.getByLabelText('시작일') as HTMLInputElement;
    fireEvent.change(input,{target:{value:'2020-01-01'}});
    expect(input.value).toBe('2020-01-01');
    fireEvent.click(screen.getByRole('button',{name:'최대 기간'}));
    expect(input.value).toBe('');
    fireEvent.click(screen.getByRole('button',{name:'저장 이력'}));
    expect(screen.getByRole('heading',{name:'월말 확정 신호'})).toBeTruthy();
  });
});
