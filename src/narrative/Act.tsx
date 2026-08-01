import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react'
import { useSettings } from '../theme'
import { KeyPoints } from '../ui/prims'

interface ActCtx {
  active: string
  setActive: (id: string) => void
}
const Ctx = createContext<ActCtx>({ active: '', setActive: () => {} })

/**
 * 一「幕」= 8~12 分钟的注意力预算。
 *
 * 注意力在连续投入 10~15 分钟后会显著衰减，所以内容按幕切分。
 * 每一幕开头写清「这一幕要搞懂什么」，结尾用要点卡收口 ——
 * 叙事负责理解，要点负责带走，两者缺一不可。
 */
export function Act({
  no,
  title,
  goal,
  minutes,
  stage,
  children,
  id,
  points,
}: {
  id: string
  no: string
  title: string
  goal: string
  minutes: number
  stage: (beat: string) => ReactNode
  children: ReactNode
  /** 本幕要点。速读模式下正文收起，只剩它和舞台。 */
  points?: ReactNode[]
}) {
  const [active, setActive] = useState('')
  const value = useMemo(() => ({ active, setActive }), [active])
  const skim = useSettings((s) => s.skim)

  return (
    <section id={id} className="scroll-mt-24 border-t border-line py-14 first:border-0">
      <Ctx.Provider value={value}>
        <header className="mx-auto mb-8 max-w-6xl px-6">
          <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
            <span className="font-mono text-[12px] tracking-[.2em] text-brand">{no}</span>
            <h2 className="font-serif text-[28px] leading-tight font-bold text-ink md:text-[34px]">
              {title}
            </h2>
            <span className="ml-auto rounded-full border border-line px-2.5 py-0.5 font-mono text-[11px] text-faint">
              约 {minutes} 分钟
            </span>
          </div>
          <p className="mt-3 max-w-3xl text-[14.5px] leading-relaxed text-dim">
            <span className="text-faint">这一幕你要搞懂的是：</span>
            {goal}
          </p>
        </header>

        <div className="mx-auto grid max-w-6xl gap-10 px-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,500px)]">
          <div className="order-2 min-w-0 lg:order-1">
            {points && skim && <div className="pt-1">{<KeyPoints items={points} />}</div>}
            {children}
            {points && !skim && (
              <div className="pl-6">
                <KeyPoints items={points} />
              </div>
            )}
          </div>
          <div className="order-1 min-w-0 lg:order-2">
            {/* 舞台比视口高时让它自己内部滚动，否则 sticky 会把底部截掉 */}
            <div className="lg:sticky lg:top-20 lg:max-h-[calc(100vh-6rem)] lg:overflow-y-auto lg:pr-1">
              {stage(active)}
            </div>
          </div>
        </div>
      </Ctx.Provider>
    </section>
  )
}

/**
 * 一个叙事节拍。滚到屏幕中间时它会成为「当前节拍」，舞台随之演化。
 *
 * `keep` 标记的节拍在速读模式下保留 —— 通常是承载公式、下注题、
 * 关键定义的那些，其余纯叙述段落会被收起。
 */
export function Beat({
  id,
  children,
  className = '',
  keep = false,
}: {
  id: string
  children: ReactNode
  className?: string
  keep?: boolean
}) {
  const { active, setActive } = useContext(Ctx)
  const ref = useRef<HTMLDivElement>(null)
  const skim = useSettings((s) => s.skim)

  useEffect(() => {
    const el = ref.current
    if (!el) return
    const io = new IntersectionObserver(
      ([e]) => {
        if (e.isIntersecting) setActive(id)
      },
      { rootMargin: '-45% 0px -45% 0px', threshold: 0 },
    )
    io.observe(el)
    return () => io.disconnect()
  }, [id, setActive])

  if (skim && !keep) return null

  const isActive = active === id
  return (
    <div
      ref={ref}
      className={`beat relative py-7 pl-6 ${className}`}
      data-active={isActive || undefined}
    >
      <span
        className="absolute top-8 left-0 h-[calc(100%-2rem)] w-px rounded transition-all duration-500"
        style={{
          background: isActive
            ? 'linear-gradient(180deg, var(--accent), transparent)'
            : 'var(--line)',
        }}
      />
      <div
        className="prose-body transition-opacity duration-500"
        style={{ opacity: active === '' || isActive || skim ? 1 : 0.62 }}
      >
        {children}
      </div>
    </div>
  )
}

export function useActiveBeat() {
  return useContext(Ctx).active
}
