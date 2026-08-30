// 날짜 눈금만 줄이고 원본 월별 관측값은 모두 보존한다.
export function monthTickIndices(length: number, plotWidth: number): number[] {
  if (length < 1) return [];
  if (length === 1) return [0];
  const count = Math.min(length, Math.max(2, Math.floor(plotWidth / 76) + 1));
  return Array.from({ length: count }, (_, i) => Math.round(i * (length - 1) / (count - 1)));
}

export function closestMonth(x: number, left: number, plotWidth: number, length: number): number {
  if (length <= 1 || plotWidth <= 0) return 0;
  const ratio = Math.max(0, Math.min(1, (x - left) / plotWidth));
  return Math.round(ratio * (length - 1));
}

export function valueDomain(values: number[], drawdown: boolean): [number, number] {
  if (!values.length) return [0, 1];
  const min = Math.min(...values), max = Math.max(...values);
  // 낙폭은 항상 0%를 위쪽 기준으로 삼고, 전부 0%여도 축이 무너지지 않는다.
  if (drawdown) return [Math.min(-1, min * 1.08), 0];
  const padding = Math.max((max - min) * .08, Math.abs(max) * .01, 1);
  return [Math.max(0, min - padding), max + padding];
}

export const axisMonth = (date: string) => date.slice(0, 7).replace('-', '.');
export function fullDate(date: string): string {
  // 시간대 변환 없이 서버가 반환한 월말 날짜를 그대로 표시한다.
  const [year, month, day] = date.split('-').map(Number);
  return `${year}년 ${month}월 ${day}일`;
}
