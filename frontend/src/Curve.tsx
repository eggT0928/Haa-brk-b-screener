import { useEffect, useId, useRef, useState, type PointerEvent } from 'react';
import { axisMonth, closestMonth, fullDate, monthTickIndices, valueDomain } from './chartModel';
import { money, percent } from './portfolio';
import type { Backtest } from './types';
import './curve.css';

const compactUsd = new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 1 });

export function Curve({ data, drawdown = false }: { data: Backtest['curve']; drawdown?: boolean }) {
  const frame = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(720);
  const [selected, setSelected] = useState<number | null>(null);
  const [active, setActive] = useState(false);
  const dragging = useRef(false);
  const id = useId();

  useEffect(() => {
    const element = frame.current;
    if (!element) return;
    const measure = () => {
      const next = Math.round(element.getBoundingClientRect().width);
      if (next > 0) setWidth(next);
    };
    measure();
    // 접힌 차트를 열거나 기기를 회전해도 다시 측정한다. 축 글자는 축소하지 않는다.
    const observer = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(measure);
    observer?.observe(element);
    window.addEventListener('resize', measure);
    return () => { observer?.disconnect(); window.removeEventListener('resize', measure); };
  }, []);

  useEffect(() => { setSelected(null); setActive(false); dragging.current = false; }, [data]);

  const valid = data.length > 0 && data.every(p => Number.isFinite(p.value) && Number.isFinite(p.drawdown));
  const title = drawdown ? '월말 낙폭' : '포트폴리오 가치';
  const height = width < 600 ? 252 : 300;
  const left = 62, right = 18, top = 20, bottom = height - 38;
  const plotWidth = Math.max(1, width - left - right);
  const values = valid ? data.map(p => drawdown ? p.drawdown * 100 : p.value) : [];
  const [min, max] = valueDomain(values, drawdown);
  const x = (i: number) => left + (data.length === 1 ? .5 : i / (data.length - 1)) * plotWidth;
  const y = (value: number) => bottom - (value - min) / (max - min) * (bottom - top);
  const index = Math.max(0, Math.min(selected ?? data.length - 1, data.length - 1));
  const point = valid ? data[index] : null;
  const points = values.map((value, i) => `${x(i)},${y(value)}`).join(' ');
  const ticks = monthTickIndices(data.length, plotWidth);
  const stroke = drawdown ? '#b66844' : '#21664d';
  const pointText = point ? `${fullDate(point.date)}, 평가금액 ${money(point.value)}, 낙폭 ${percent(point.drawdown)}` : '';

  function choose(next: number) {
    setSelected(Math.max(0, Math.min(data.length - 1, next)));
    setActive(true);
  }
  function pick(event: PointerEvent<SVGSVGElement>) {
    const bounds = event.currentTarget.getBoundingClientRect();
    if (bounds.width > 0) choose(closestMonth((event.clientX - bounds.left) * width / bounds.width, left, plotWidth, data.length));
  }

  return <section className={`curve ${drawdown ? 'curve-drawdown' : ''}`} aria-label={`${title} 상세 차트`}>
    <div className="curve-heading"><h3>{title}{!drawdown && <span> USD</span>}</h3><span>월말 기준 · {data.length.toLocaleString()}개 기록</span></div>
    <div ref={frame} className="curve-frame">
      {!point ? <p className="empty">표시할 유효한 월별 데이터가 없습니다.</p> : <>
        <div className="curve-readout">
          <div className="curve-date"><span>{selected === null ? '마지막 월말' : '선택한 월말'}</span><strong>{fullDate(point.date)}</strong></div>
          <div><span>평가금액</span><strong>{money(point.value)}</strong></div>
          <div><span>낙폭</span><strong className={point.drawdown < 0 ? 'negative' : ''}>{percent(point.drawdown)}</strong></div>
        </div>
        <div className="curve-plot">
          <svg viewBox={`0 0 ${width} ${height}`} style={{ height }} role="img"
            aria-label={drawdown ? '월말 기준 낙폭 그래프' : '월말 포트폴리오 가치 그래프'} aria-describedby={`${id}-help`}
            onPointerDown={event => {
              dragging.current = true; pick(event);
              if (event.pointerType === 'touch') event.currentTarget.setPointerCapture?.(event.pointerId);
            }}
            onPointerMove={event => { if (event.pointerType !== 'touch' || dragging.current) pick(event); }}
            onPointerUp={event => {
              dragging.current = false;
              if (event.currentTarget.hasPointerCapture?.(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
            }}
            onPointerCancel={() => { dragging.current = false; setActive(false); }}
            onPointerLeave={event => { if (event.pointerType !== 'touch') setActive(false); }}>
            <defs><linearGradient id={`${id}-fill`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={stroke} stopOpacity=".13"/><stop offset="100%" stopColor={stroke} stopOpacity=".01"/>
            </linearGradient></defs>
            {[0, 1, 2, 3].map(i => {
              const value = max - (max - min) * i / 3, row = y(value);
              return <g key={i}><line x1={left} x2={width - right} y1={row} y2={row} stroke="#e0e6dd" strokeDasharray="3 5"/>
                <text x={left - 10} y={row + 4} textAnchor="end">{drawdown ? `${value.toFixed(1)}%` : `$${compactUsd.format(value)}`}</text></g>;
            })}
            {!drawdown && data.length > 1 && <polygon points={`${x(0)},${bottom} ${points} ${x(data.length - 1)},${bottom}`} fill={`url(#${id}-fill)`}/>}
            <polyline points={points} fill="none" stroke={stroke} strokeWidth="2.5" strokeLinejoin="round" strokeLinecap="round"/>
            {data.length === 1 && <circle cx={x(0)} cy={y(values[0])} r="4" fill={stroke}/>}
            {ticks.map((i, position) => <g key={i} className="curve-month-tick">
              <line x1={x(i)} x2={x(i)} y1={bottom} y2={bottom + 5} stroke="#cbd6c6"/>
              <text x={x(i)} y={bottom + 25} textAnchor={ticks.length === 1 ? 'middle' : position === 0 ? 'start' : position === ticks.length - 1 ? 'end' : 'middle'}>{axisMonth(data[i].date)}</text>
            </g>)}
            {(active || selected !== null) && <g className="curve-cursor" aria-hidden="true">
              <line x1={x(index)} x2={x(index)} y1={top} y2={bottom} stroke="#89978b" strokeDasharray="4 4"/>
              <circle cx={x(index)} cy={y(values[index])} r="5" fill={stroke} stroke="white" strokeWidth="2"/>
            </g>}
          </svg>
          {active && <div className="curve-tooltip" role="tooltip" style={{ left: Math.max(4, Math.min(width - 224, x(index) + (x(index) > width / 2 ? -224 : 14))) }}>
            <strong>{fullDate(point.date)}</strong><div><span>평가금액</span><b>{money(point.value)}</b></div><div><span>낙폭</span><b>{percent(point.drawdown)}</b></div>
          </div>}
        </div>
        <div className="curve-explorer">
          <button aria-label={`${title} 이전 월`} disabled={index === 0} onClick={() => choose(index - 1)}>‹</button>
          <label><span>월별 탐색 <b>{axisMonth(point.date)}</b></span><input type="range" min="0" max={data.length - 1} step="1" value={index}
            disabled={data.length < 2} aria-label={`${title} 월별 탐색`} aria-valuetext={pointText} aria-describedby={`${id}-help`}
            onKeyDown={event => {
              // 브라우저별 range 기본 동작과 무관하게 한 달 단위 키보드 탐색을 보장한다.
              const next = { ArrowLeft: index - 1, ArrowDown: index - 1, ArrowRight: index + 1, ArrowUp: index + 1,
                Home: 0, End: data.length - 1, PageUp: index + 12, PageDown: index - 12 }[event.key];
              if (next !== undefined) { event.preventDefault(); choose(next); }
            }}
            onFocus={() => setActive(true)} onBlur={() => setActive(false)} onChange={event => choose(Number(event.target.value))}/></label>
          <button aria-label={`${title} 다음 월`} disabled={index === data.length - 1} onClick={() => choose(index + 1)}>›</button>
        </div>
        <p className="curve-help" id={`${id}-help`}>마우스를 올리거나 그래프를 터치해 수치를 확인하세요. 아래 막대·화살표로 모든 월을 선택할 수 있습니다. 날짜 눈금은 화면 폭에 맞춰 생략됩니다.</p>
      </>}
    </div>
  </section>;
}
