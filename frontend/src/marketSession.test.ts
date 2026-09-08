import { describe, expect, it } from 'vitest';
import { marketSession } from './marketSession';
describe('NYSE market status', () => {
  it('identifies Labor Day instead of a quote failure', () => expect(marketSession(Date.parse('2026-09-07T15:00:00Z')).message).toContain('공휴일'));
  it('uses US date during Korean Tuesday morning', () => expect(marketSession(Date.parse('2026-09-08T02:00:00Z')).message).toContain('휴장'));
  it('identifies weekends', () => expect(marketSession(Date.parse('2026-09-06T16:00:00Z')).message).toContain('주말'));
  it('recognizes the next open session', () => expect(marketSession(Date.parse('2026-09-08T14:00:00Z')).closed).toBe(false));
  it('respects early closing', () => expect(marketSession(Date.parse('2026-11-27T18:30:00Z')).closed).toBe(true));
  it('does not invent holidays beyond calendar coverage', () => expect(marketSession(Date.parse('2040-09-07T15:00:00Z')).message).toContain('달력 확인'));
});
