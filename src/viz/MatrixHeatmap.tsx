import { useBus, useHighlight } from '../highlight/bus'
import { useColors } from '../theme'

interface Props {
  /** 方阵 */
  m: number[][]
  size?: number
  label?: string
  className?: string
}

/**
 * P_π 的热力图。
 * 关键教学点：矩阵的第 s 行，就是网格上「从格子 s 出发」的那一步 ——
 * 悬停一行，网格上对应的格子和它的后继会同时亮起来。
 */
export function MatrixHeatmap({ m, size = 15, label, className = '' }: Props) {
  const n = m.length
  const C = useColors()
  const hl = useHighlight()
  const setHovered = useBus((s) => s.setHovered)
  const clearHovered = useBus((s) => s.clearHovered)

  return (
    <div className={className}>
      {label && <div className="mb-2 text-[11px] tracking-wide text-faint">{label}</div>}
      <svg
        viewBox={`0 0 ${n * size + 1} ${n * size + 1}`}
        style={{ width: Math.min(n * size + 1, 320) }}
        onMouseLeave={clearHovered}
      >
        {m.map((row, i) =>
          row.map((val, j) => {
            const rowHot = hl.states.has(i)
            const colHot = hl.successors.has(j) || hl.states.has(j)
            return (
              <rect
                key={`${i}-${j}`}
                x={j * size + 0.5}
                y={i * size + 0.5}
                width={size - 1}
                height={size - 1}
                fill={
                  val > 0
                    ? rowHot
                      ? C.accent
                      : C.prob
                    : rowHot || colHot
                      ? 'var(--surface-3)'
                      : 'var(--surface-2)'
                }
                opacity={val > 0 ? 0.35 + 0.65 * val : 1}
                onMouseEnter={() => {
                  const ents = row
                    .map((p, sp) => (p > 0 ? { kind: 'transition' as const, s: i, a: 0, sp } : null))
                    .filter(Boolean) as { kind: 'transition'; s: number; a: number; sp: number }[]
                  setHovered([{ kind: 'state', s: i }, ...ents])
                }}
              />
            )
          }),
        )}
      </svg>
    </div>
  )
}
