import { describe, expect, it } from 'vitest';
import { axisMonth, closestMonth, fullDate, monthTickIndices, valueDomain } from './chartModel';

describe('월별 차트 눈금과 좌표', () => {
  it('짧은 기간은 모든 월을 표시한다', () => expect(monthTickIndices(6, 600)).toEqual([0,1,2,3,4,5]));
  it('장기 기간은 첫·끝·중간 월을 간격에 맞춰 표시한다', () => {
    const ticks = monthTickIndices(217, 900);
    expect(ticks[0]).toBe(0); expect(ticks.at(-1)).toBe(216);
    expect(ticks.length).toBeGreaterThan(2); expect(ticks.length).toBeLessThan(14);
    expect(new Set(ticks).size).toBe(ticks.length);
  });
  it('모바일에서는 겹치지 않도록 날짜 개수를 더 줄인다', () => {
    const ticks = monthTickIndices(217, 240);
    expect(ticks.length).toBe(4);
    expect(ticks.length).toBeLessThan(monthTickIndices(217, 900).length);
    for (let i=1; i<ticks.length; i++) expect((ticks[i]-ticks[i-1])/216*240).toBeGreaterThanOrEqual(76);
  });
  it('한 지점·빈 데이터·극소 폭을 처리한다', () => {
    expect(monthTickIndices(0, 200)).toEqual([]); expect(monthTickIndices(1, 200)).toEqual([0]);
    expect(monthTickIndices(3, 0)).toEqual([0,2]);
  });
  it('축 눈금에서 생략된 월도 정확히 선택할 수 있다', () => {
    for(let i=0;i<217;i++) expect(closestMonth(62+i/216*600,62,600,217)).toBe(i);
    expect(closestMonth(-100,62,600,217)).toBe(0); expect(closestMonth(900,62,600,217)).toBe(216);
    expect(closestMonth(200,62,600,1)).toBe(0);
  });
  it('날짜는 현지 시간대의 전날로 바뀌지 않는다', () => {
    expect(axisMonth('2026-07-31')).toBe('2026.07'); expect(fullDate('2026-07-31')).toBe('2026년 7월 31일');
  });
  it('평탄하거나 한 점인 그래프도 유한한 범위를 유지한다', () => {
    for (const values of [[10000],[0,0],[10000,10000]]) {
      const [min,max]=valueDomain(values,false); expect(max).toBeGreaterThan(min);
      expect(min).toBeLessThanOrEqual(values[0]); expect(max).toBeGreaterThanOrEqual(values[0]);
    }
    expect(valueDomain([0,0],true)).toEqual([-1,0]); expect(valueDomain([-20,-10,0],true)[1]).toBe(0);
  });
});
