import { useMemo } from 'react'
import { fmt } from '../theme'

export interface Series {
  name: string
  color: string
  data: number[]
  dashed?: boolean
  width?: number
}

interface Props {
  series: Series[]
  xLabel?: string
  yLabel?: string
  logY?: boolean
  /** 当前时间轴位置，会画一条竖线 */
  marker?: number
  height?: number
  yMin?: number
  yMax?: number
  className?: string
}

const PAD = { l: 52, r: 16, t: 14, b: 30 }

export function LineChart({
  series,
  xLabel,
  yLabel,
  logY,
  marker,
  height = 200,
  yMin,
  yMax,
  className = '',
}: Props) {
  const W = 640
  const H = height

  const { xs, ys, n } = useMemo(() => {
    const n = Math.max(...series.map((s) => s.data.length), 2)
    const flat = series.flatMap((s) => s.data).filter((x) => Number.isFinite(x))
    const clip = logY ? flat.filter((x) => x > 0) : flat
    let lo = yMin ?? (clip.length ? Math.min(...clip) : 0)
    let hi = yMax ?? (clip.length ? Math.max(...clip) : 1)
    if (logY) {
      lo = Math.max(lo, 1e-12)
      hi = Math.max(hi, lo * 10)
    } else {
      const pad = (hi - lo) * 0.12 || 1
      lo -= pad
      hi += pad
    }
    const tx = (i: number) => PAD.l + (i / (n - 1)) * (W - PAD.l - PAD.r)
    const ty = (y: number) => {
      const t = logY
        ? (Math.log10(Math.max(y, 1e-12)) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo))
        : (y - lo) / (hi - lo)
      return PAD.t + (1 - Math.min(1, Math.max(0, t))) * (H - PAD.t - PAD.b)
    }
    return { xs: tx, ys: ty, n, lo, hi }
  }, [series, logY, yMin, yMax, H])

  const ticks = useMemo(() => {
    const flat = series.flatMap((s) => s.data).filter((x) => Number.isFinite(x))
    const clip = logY ? flat.filter((x) => x > 0) : flat
    const lo = yMin ?? (clip.length ? Math.min(...clip) : 0)
    const hi = yMax ?? (clip.length ? Math.max(...clip) : 1)
    if (logY) {
      const a = Math.floor(Math.log10(Math.max(lo, 1e-12)))
      const b = Math.ceil(Math.log10(Math.max(hi, 1e-11)))
      // 跨度可能有二十几个数量级；按步长抽稀，而不是砍掉尾巴 ——
      // 砍掉尾巴会让网格线和数据完全对不上。
      const stride = Math.max(1, Math.ceil((b - a + 1) / 7))
      const out: number[] = []
      for (let e = a; e <= b; e += stride) out.push(Math.pow(10, e))
      return out
    }
    const pad = (hi - lo) * 0.12 || 1
    const l = lo - pad
    const h = hi + pad
    return Array.from({ length: 5 }, (_, i) => l + ((h - l) * i) / 4)
  }, [series, logY, yMin, yMax])

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className={`w-full ${className}`}>
      {ticks.map((t, i) => (
        <g key={i}>
          <line
            x1={PAD.l}
            x2={W - PAD.r}
            y1={ys(t)}
            y2={ys(t)}
            stroke="var(--line)"
            strokeWidth={1}
          />
          <text
            x={PAD.l - 8}
            y={ys(t) + 3.5}
            textAnchor="end"
            fontSize={10}
            fill="var(--ink-faint)"
            fontFamily="ui-monospace, monospace"
          >
            {logY ? `1e${Math.round(Math.log10(t))}` : fmt(t, 1)}
          </text>
        </g>
      ))}

      <line
        x1={PAD.l}
        x2={W - PAD.r}
        y1={H - PAD.b}
        y2={H - PAD.b}
        stroke="var(--line-strong)"
      />

      {marker !== undefined && marker >= 0 && (
        <line
          x1={xs(marker)}
          x2={xs(marker)}
          y1={PAD.t}
          y2={H - PAD.b}
          stroke="var(--accent)"
          strokeWidth={1.5}
          strokeDasharray="4 3"
        />
      )}

      {series.map((s) => {
        const pts = s.data
          .map((y, i) => (Number.isFinite(y) ? `${xs(i)},${ys(y)}` : null))
          .filter(Boolean)
          .join(' ')
        return (
          <polyline
            key={s.name}
            points={pts}
            fill="none"
            stroke={s.color}
            strokeWidth={s.width ?? 2.2}
            strokeDasharray={s.dashed ? '5 4' : undefined}
            strokeLinejoin="round"
            strokeLinecap="round"
          />
        )
      })}

      {marker !== undefined &&
        series.map((s) =>
          Number.isFinite(s.data[marker]) ? (
            <circle key={`m${s.name}`} cx={xs(marker)} cy={ys(s.data[marker])} r={3.5} fill={s.color} />
          ) : null,
        )}

      {xLabel && (
        <text x={(W + PAD.l) / 2} y={H - 6} textAnchor="middle" fontSize={11} fill="var(--ink-faint)">
          {xLabel}
        </text>
      )}
      {yLabel && (
        <text
          x={12}
          y={H / 2}
          textAnchor="middle"
          fontSize={11}
          fill="var(--ink-faint)"
          transform={`rotate(-90 12 ${H / 2})`}
        >
          {yLabel}
        </text>
      )}

      <g>
        {series.map((s, i) => (
          <g key={`l${s.name}`} transform={`translate(${PAD.l + 8 + i * 150}, ${PAD.t + 4})`}>
            <line x1={0} x2={18} y1={0} y2={0} stroke={s.color} strokeWidth={2.4} strokeDasharray={s.dashed ? '5 4' : undefined} />
            <text x={24} y={3.5} fontSize={11} fill="var(--ink-dim)">
              {s.name}
            </text>
          </g>
        ))}
      </g>

      <text x={W - PAD.r} y={H - 6} textAnchor="end" fontSize={10} fill="var(--ink-faint)">
        n = {n - 1}
      </text>
    </svg>
  )
}
