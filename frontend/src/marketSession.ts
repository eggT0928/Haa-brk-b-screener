import { sessions, calendarStart, calendarEnd } from './nyseSessions';

export function marketSession(now: number) {
  const date = new Intl.DateTimeFormat('en-CA', { timeZone: 'America/New_York', year: 'numeric', month: '2-digit', day: '2-digit' }).format(now);
  if (date < calendarStart || date > calendarEnd) return { message: '거래소 달력 확인이 필요합니다. 표시된 시세 기준 시각을 확인하세요.', closed: false };
  const session = sessions[date];
  if (!session) {
    const weekend = [0, 6].includes(new Date(`${date}T12:00:00Z`).getUTCDay());
    return { closed: true, message: `미국 증시 휴장 (${weekend ? '주말' : '미국 공휴일·거래소 휴장일'}): 새로운 시세 조회가 불가능하여 마지막 거래일의 저장된 시세·신호를 표시합니다.` };
  }
  if (now < Date.parse(session[0]) || now >= Date.parse(session[1])) return { closed: true, message: '미국 증시 정규장 시간 외입니다. 마지막으로 확인된 시세·신호를 표시하며 다음 거래일 정규장에 자동 갱신합니다.' };
  return { closed: false, message: '미국 증시 정규장 운영 중 · 시세는 약 15분 간격으로 갱신합니다.' };
}
