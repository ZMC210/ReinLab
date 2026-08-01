import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import katex from 'katex'
import type { Entity } from '../highlight/bus'
import { useBus, useHighlight, type HighlightSets } from '../highlight/bus'
import { fmt } from '../theme'
import { nodeDesc, renderFormula, type FCtx, type FMode, type FNode } from './core'

let counter = 0
const nextUid = () => `f${++counter}_`

function isLit(e: Entity, hl: HighlightSets): boolean {
  switch (e.kind) {
    case 'state':
    case 'value':
      return hl.states.has(e.s) || hl.successors.has(e.s)
    case 'policy':
      return hl.states.has(e.s)
    case 'action':
    case 'qvalue':
      return hl.actions.has(`${e.s}:${e.a}`)
    case 'reward':
      return hl.rewards.has(`${e.s}:${e.a}`) || hl.actions.has(`${e.s}:${e.a}`)
    case 'transition':
      return hl.actions.has(`${e.s}:${e.a}`) && hl.successors.has(e.sp)
    case 'gamma':
      return hl.gamma
  }
}

interface Tip {
  x: number
  y: number
  /** 卡片朝上还是朝下展开 —— 靠近视口顶部时必须朝下，否则会被裁掉 */
  below: boolean
  title: string
  desc?: string
  val?: number
}

/**
 * 悬停卡片挂在 body 上而不是公式容器里。
 * 因为舞台是 sticky + overflow-y-auto 的，任何绝对定位的浮层
 * 都会被它裁掉 —— 这正是之前文字显示不全的原因。
 */
function TipCard({ tip }: { tip: Tip }) {
  return createPortal(
    <div
      className="pointer-events-none fixed z-[100] -translate-x-1/2"
      style={{
        left: tip.x,
        top: tip.y,
        transform: `translateX(-50%) translateY(${tip.below ? '10px' : 'calc(-100% - 10px)'})`,
      }}
    >
      <div className="w-[min(20rem,calc(100vw-2rem))] rounded-xl border border-line bg-surface px-3.5 py-2.5 shadow-xl backdrop-blur-sm">
        {tip.title && <div className="text-[13px] font-semibold text-ink">{tip.title}</div>}
        {tip.desc && (
          <div className="mt-1 text-[12.5px] leading-relaxed text-dim">{tip.desc}</div>
        )}
        {tip.val !== undefined && Number.isFinite(tip.val) && (
          <div className="mt-1.5 font-mono text-[12.5px] text-brand">
            当前值 = {fmt(tip.val, 3)}
          </div>
        )}
      </div>
    </div>,
    document.body,
  )
}

interface Props {
  node: FNode
  ctx: FCtx
  mode?: FMode
  className?: string
  /** 关掉交互，用于纯展示的小公式 */
  inert?: boolean
}

export function LiveFormula({ node, ctx, mode = 'symbolic', className = '', inert }: Props) {
  const uid = useMemo(nextUid, [])
  const hostRef = useRef<HTMLDivElement>(null)
  const [tip, setTip] = useState<Tip | null>(null)
  const setHovered = useBus((s) => s.setHovered)
  const clearHovered = useBus((s) => s.clearHovered)
  const hl = useHighlight(ctx.mdp.nA)

  const { tex, registry } = useMemo(
    () => renderFormula(node, uid, mode, ctx),
    [node, uid, mode, ctx],
  )

  useLayoutEffect(() => {
    const host = hostRef.current
    if (!host) return
    host.innerHTML = katex.renderToString(tex, {
      displayMode: true,
      throwOnError: false,
      trust: true,
      strict: false,
      output: 'html',
    })
  }, [tex])

  // 反向绑定：世界那边点亮了什么，公式这边对应的片段跟着亮
  useEffect(() => {
    const host = hostRef.current
    if (!host || inert) return
    for (const [path, n] of registry) {
      const el = host.querySelector<HTMLElement>(`[id="${CSS.escape(path)}"]`)
      if (!el) continue
      const ents = n.entities?.(ctx) ?? []
      const lit = hl.any && ents.length > 0 && ents.some((e) => isLit(e, hl))
      el.classList.toggle('f-lit', lit)
    }
  }, [hl, registry, ctx, tex, inert])

  // 滚动时立刻收起卡片，避免它停在错误的位置上
  useEffect(() => {
    if (!tip) return
    const off = () => setTip(null)
    addEventListener('scroll', off, { passive: true, capture: true })
    return () => removeEventListener('scroll', off, { capture: true })
  }, [tip])

  if (inert) {
    return <div ref={hostRef} className={`live-formula inert ${className}`} />
  }

  const clearHot = () =>
    hostRef.current?.querySelectorAll('.f-hot').forEach((n) => n.classList.remove('f-hot'))

  const onMove = (ev: React.MouseEvent) => {
    const host = hostRef.current
    if (!host) return
    const el = (ev.target as HTMLElement).closest<HTMLElement>(`[id^="${uid}"]`)
    if (!el) {
      setTip(null)
      clearHovered()
      clearHot()
      return
    }
    const n = registry.get(el.id)
    if (!n) return

    clearHot()
    el.classList.add('f-hot')
    setHovered(n.entities?.(ctx) ?? [])

    const b = el.getBoundingClientRect()
    setTip({
      x: b.left + b.width / 2,
      // 顶部空间不足 160px 时改为向下展开
      below: b.top < 170,
      y: b.top < 170 ? b.bottom : b.top,
      title: n.name ?? '',
      desc: nodeDesc(n, ctx),
      val: n.value?.(ctx),
    })
  }

  const onLeave = () => {
    setTip(null)
    clearHovered()
    clearHot()
  }

  return (
    <div className={className}>
      <div ref={hostRef} className="live-formula" onMouseMove={onMove} onMouseLeave={onLeave} />
      {tip && (tip.title || tip.desc || tip.val !== undefined) && <TipCard tip={tip} />}
    </div>
  )
}
