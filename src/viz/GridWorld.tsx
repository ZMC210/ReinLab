import { useMemo } from 'react'
import { ACTIONS, colOf, rowOf, type MDP } from '../core/mdp'
import type { Policy } from '../core/policy'
import { useBus, useHighlight, type HighlightSets } from '../highlight/bus'
import { fmt, useColors, useSettings, valueColor, type Palette } from '../theme'

const EMPTY_HL: HighlightSets = {
  states: new Set(),
  successors: new Set(),
  actions: new Set(),
  rewards: new Set(),
  gamma: false,
  any: false,
}

export interface GridWorldProps {
  mdp: MDP
  v?: number[]
  q?: number[][]
  policy?: Policy
  /** 并列最优的动作，会画成空心箭头 */
  ties?: number[][]
  showValues?: boolean
  showPolicy?: boolean
  showHeatmap?: boolean
  showQ?: boolean
  /** 显示状态编号 s₁ s₂ … */
  showLabels?: boolean
  cell?: number
  agent?: number
  trail?: number[]
  onCellClick?: (s: number) => void
  className?: string
  /** 关闭与高亮总线的联动（用于并排对比的小图） */
  quiet?: boolean
  /** 高度上限，默认 48vh —— 网格不该霸占整屏 */
  maxH?: string
}

const GAP = 3

export function GridWorld({
  mdp,
  v,
  q,
  policy,
  ties,
  showValues = true,
  showPolicy = true,
  showHeatmap = true,
  showQ = false,
  showLabels = false,
  cell = 58,
  agent,
  trail,
  onCellClick,
  className = '',
  quiet = false,
  maxH = '48vh',
}: GridWorldProps) {
  const { grid } = mdp
  const C = useColors()
  const theme = useSettings((s) => s.theme)
  const live = useHighlight(mdp.nA)
  const setHovered = useBus((s) => s.setHovered)
  const clearHovered = useBus((s) => s.clearHovered)
  const focusState = useBus((s) => s.focus)

  // quiet 的图既不发布也不订阅高亮，用于并排对比时避免互相干扰
  const hl = quiet ? EMPTY_HL : live
  const focus = quiet ? null : focusState

  const W = grid.cols * cell
  const H = grid.rows * cell

  const [lo, hi] = useMemo(() => {
    if (!v) return [0, 1]
    const mn = Math.min(...v)
    const mx = Math.max(...v)
    return mx - mn < 1e-9 ? [mn - 1, mx + 1] : [mn, mx]
  }, [v])

  const trailSet = useMemo(() => new Set(trail ?? []), [trail])

  const emit = quiet ? () => {} : setHovered
  const clear = quiet ? () => {} : clearHovered

  return (
    <svg
      viewBox={`-6 -6 ${W + 12} ${H + 12}`}
      className={`mx-auto block w-full select-none ${className}`}
      style={{ maxHeight: maxH, maxWidth: W + 12 }}
      onMouseLeave={clear}
    >
      <defs>
        <pattern id="hatch" width="7" height="7" patternTransform="rotate(45)" patternUnits="userSpaceOnUse">
          <line x1="0" y1="0" x2="0" y2="7" stroke={C.danger} strokeOpacity="0.35" strokeWidth="3" />
        </pattern>
        <filter id="cellGlow" x="-60%" y="-60%" width="220%" height="220%">
          <feGaussianBlur stdDeviation="4" result="b" />
          <feMerge>
            <feMergeNode in="b" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {Array.from({ length: mdp.nS }, (_, s) => {
        const r = rowOf(grid, s)
        const c = colOf(grid, s)
        const x = c * cell + GAP / 2
        const y = r * cell + GAP / 2
        const w = cell - GAP
        const kind = grid.cells[s]

        const isHot = hl.states.has(s)
        const isSucc = hl.successors.has(s)
        const isFocus = focus === s

        const base =
          showHeatmap && v
            ? valueColor(v[s], lo, hi, theme)
            : kind === 'forbidden'
              ? C.cellForbidden
              : kind === 'target'
                ? C.cellTarget
                : C.cell

        return (
          <g
            key={s}
            onMouseEnter={() => emit([{ kind: 'state', s }])}
            onClick={() => onCellClick?.(s)}
            style={{ cursor: onCellClick ? 'pointer' : 'default' }}
          >
            <rect
              x={x}
              y={y}
              width={w}
              height={w}
              rx={8}
              fill={base}
              style={{ transition: 'fill 420ms cubic-bezier(.4,0,.2,1)' }}
            />
            {kind === 'forbidden' && <rect x={x} y={y} width={w} height={w} rx={8} fill="url(#hatch)" />}
            {kind === 'target' && (
              <rect
                x={x + 3.5}
                y={y + 3.5}
                width={w - 7}
                height={w - 7}
                rx={6}
                fill="none"
                stroke={C.target}
                strokeWidth={2}
                strokeDasharray="5 4"
                opacity={0.9}
              />
            )}
            {trailSet.has(s) && (
              <rect x={x} y={y} width={w} height={w} rx={8} fill={C.accent} opacity={0.12} />
            )}

            <rect
              x={x}
              y={y}
              width={w}
              height={w}
              rx={8}
              fill="none"
              stroke={isFocus || isHot ? C.accent : isSucc ? C.reward : C.cellLine}
              strokeWidth={isFocus ? 3 : isHot || isSucc ? 2.5 : 1}
              strokeDasharray={isSucc && !isHot ? '6 4' : undefined}
              filter={isHot || isFocus ? 'url(#cellGlow)' : undefined}
              style={{ transition: 'stroke 180ms, stroke-width 180ms' }}
            />

            {showLabels && (
              <text
                x={x + 5}
                y={y + 13}
                fontSize={9.5}
                fill={C.cellText}
                opacity={0.5}
                fontFamily="ui-monospace, monospace"
              >
                s{s + 1}
              </text>
            )}

            {showValues && v && (
              <text
                x={x + w / 2}
                y={y + w / 2 + (showPolicy ? 17 : 5)}
                textAnchor="middle"
                fontSize={showPolicy ? 12 : 17}
                fontWeight={600}
                fill={C.cellText}
                fontFamily="ui-monospace, monospace"
                style={{ paintOrder: 'stroke', stroke: C.cellTextHalo, strokeWidth: 3 }}
              >
                {fmt(v[s], 1)}
              </text>
            )}
          </g>
        )
      })}

      {showPolicy &&
        policy &&
        Array.from({ length: mdp.nS }, (_, s) => (
          <PolicyGlyphs
            key={`p${s}`}
            s={s}
            cell={cell}
            grid={mdp.grid}
            probs={policy[s]}
            tie={ties?.[s]}
            hot={hl.actions}
            C={C}
            onHover={(a) => emit([{ kind: 'action', s, a }])}
          />
        ))}

      {showQ && q && (
        <g>
          {Array.from({ length: mdp.nS }, (_, s) =>
            q[s].map((val, a) => {
              const r = rowOf(mdp.grid, s)
              const c = colOf(mdp.grid, s)
              const cx = c * cell + cell / 2
              const cy = r * cell + cell / 2
              const d = ACTIONS[a]
              const off = cell * 0.34
              return (
                <text
                  key={`q${s}-${a}`}
                  x={cx + d.dc * off}
                  y={cy + d.dr * off + 3.5}
                  textAnchor="middle"
                  fontSize={9}
                  fontFamily="ui-monospace, monospace"
                  fill={C.qvalue}
                  opacity={hl.actions.has(`${s}:${a}`) ? 1 : 0.62}
                >
                  {fmt(val, 1)}
                </text>
              )
            }),
          )}
        </g>
      )}

      {agent !== undefined && (
        <g
          style={{ transition: 'transform 260ms cubic-bezier(.34,1.4,.5,1)' }}
          transform={`translate(${colOf(grid, agent) * cell + cell / 2}, ${
            rowOf(grid, agent) * cell + cell / 2
          })`}
        >
          <circle r={cell * 0.19} fill={C.accent} opacity={0.24} />
          <circle r={cell * 0.11} fill={C.accent} stroke={C.cellTextHalo} strokeWidth={2} />
        </g>
      )}
    </svg>
  )
}

function PolicyGlyphs({
  s,
  cell,
  grid,
  probs,
  tie,
  hot,
  onHover,
  C,
}: {
  s: number
  cell: number
  grid: MDP['grid']
  probs: number[]
  tie?: number[]
  hot: Set<string>
  onHover: (a: number) => void
  C: Palette
}) {
  const cx = colOf(grid, s) * cell + cell / 2
  const cy = rowOf(grid, s) * cell + cell / 2

  return (
    <g>
      {probs.map((p, a) => {
        if (p < 0.005) return null
        const isHot = hot.has(`${s}:${a}`)
        const isTie = tie ? tie.includes(a) && tie.length > 1 : false
        const d = ACTIONS[a]
        const opacity = 0.4 + 0.6 * p
        const stroke = isHot ? C.accent : C.policy

        if (d.dr === 0 && d.dc === 0) {
          return (
            <circle
              key={a}
              cx={cx}
              cy={cy - cell * 0.12}
              r={cell * 0.06 * (0.6 + 0.8 * p)}
              fill="none"
              stroke={stroke}
              strokeWidth={isHot ? 2.6 : 1.9}
              strokeDasharray={isTie ? '3 3' : undefined}
              opacity={opacity}
              onMouseEnter={() => onHover(a)}
              style={{ transition: 'all 160ms', cursor: 'crosshair' }}
            />
          )
        }

        const len = cell * 0.31 * (0.45 + 0.55 * p)
        const ex = cx + d.dc * len
        const ey = cy + d.dr * len
        const head = cell * 0.08
        const px = -d.dr
        const py = d.dc
        const tri = `${ex + d.dc * head},${ey + d.dr * head} ${ex + px * head * 0.6},${
          ey + py * head * 0.6
        } ${ex - px * head * 0.6},${ey - py * head * 0.6}`

        return (
          <g
            key={a}
            opacity={opacity}
            onMouseEnter={() => onHover(a)}
            style={{ transition: 'opacity 160ms', cursor: 'crosshair' }}
          >
            <line
              x1={cx}
              y1={cy}
              x2={ex}
              y2={ey}
              stroke={stroke}
              strokeWidth={isHot ? 2.8 : 2}
              strokeLinecap="round"
            />
            <polygon points={tri} fill={isTie ? 'none' : stroke} stroke={stroke} strokeWidth={1.3} />
            <line x1={cx} y1={cy} x2={ex} y2={ey} stroke="transparent" strokeWidth={14} />
          </g>
        )
      })}
    </g>
  )
}
