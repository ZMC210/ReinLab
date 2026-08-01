import { useEffect, useRef, useState } from 'react'

/**
 * 时间轴。
 *
 * 因为环境小到可以把每一步的完整状态都存下来，所以这里能做到
 * 「任意回看第 k 次迭代的全貌」—— 这是让算法从黑盒变成可观察对象的关键。
 */
export function Scrubber({
  k,
  setK,
  max,
  label = '迭代步 k',
  fps = 6,
}: {
  k: number
  setK: (k: number) => void
  max: number
  label?: string
  fps?: number
}) {
  const [playing, setPlaying] = useState(false)
  const kRef = useRef(k)
  kRef.current = k

  useEffect(() => {
    if (!playing) return
    const id = setInterval(() => {
      const next = kRef.current + 1
      if (next > max) {
        setPlaying(false)
        return
      }
      setK(next)
    }, 1000 / fps)
    return () => clearInterval(id)
  }, [playing, max, setK, fps])

  const btn =
    'flex h-8 w-8 items-center justify-center rounded-lg border border-line text-[12px] text-dim transition-colors hover:border-brand/40 hover:text-brand disabled:opacity-30'

  return (
    <div className="flex items-center gap-2.5">
      <button className={btn} onClick={() => setK(0)} disabled={k === 0} title="回到起点">
        ⏮
      </button>
      <button className={btn} onClick={() => setK(Math.max(0, k - 1))} disabled={k === 0} title="上一步">
        ◀
      </button>
      <button
        className={`${btn} border-brand/40 text-brand`}
        onClick={() => {
          if (k >= max) setK(0)
          setPlaying(!playing)
        }}
        title="播放"
      >
        {playing ? '❚❚' : '▶'}
      </button>
      <button
        className={btn}
        onClick={() => setK(Math.min(max, k + 1))}
        disabled={k >= max}
        title="下一步"
      >
        ▶
      </button>

      <input
        type="range"
        min={0}
        max={max}
        step={1}
        value={k}
        onChange={(e) => {
          setPlaying(false)
          setK(Number(e.target.value))
        }}
        className="ui-range mx-1 min-w-0 flex-1"
        style={{ '--pct': `${(k / max) * 100}%` } as React.CSSProperties}
      />

      <span className="shrink-0 font-mono text-[12px] whitespace-nowrap text-brand tabular-nums">
        {label} = {k}
      </span>
    </div>
  )
}
