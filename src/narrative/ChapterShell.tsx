import type { ReactNode } from 'react'
import { CHAPTERS } from './Storyline'
import { MB } from '../ui/prims'

export interface Glance {
  /** 这一章的核心公式（LaTeX） */
  formula: string
  /** 一句话说清这个公式在干什么 */
  formulaNote: string
  /** 三到四条硬结论，看完这几条就能跟同学讲清楚这一章 */
  takeaways: ReactNode[]
  /** 最容易被误解的地方 */
  traps: ReactNode[]
}

/**
 * 章首「一页纸速览」。
 *
 * 叙事有个副作用：故事讲得越好，核心越容易被情节盖住。
 * 所以每章开头先把答案摊在桌上 —— 先知道结论再看推导，
 * 阅读时的工作记忆负担会明显下降（这就是「先行组织者」）。
 */
export function ChapterGlance({ g }: { g: Glance }) {
  return (
    <div className="mx-auto max-w-6xl px-6">
      <div className="card overflow-hidden rounded-3xl">
        <div className="flex items-center gap-3 border-b border-line px-6 py-3">
          <span className="h-1.5 w-1.5 rounded-full bg-brand" />
          <span className="text-[11px] font-semibold tracking-[.18em] text-brand uppercase">
            一页纸速览
          </span>
          <span className="ml-auto text-[11.5px] text-faint">
            赶时间就只看这一屏；想学透就往下滚
          </span>
        </div>

        <div className="grid gap-0 md:grid-cols-[minmax(0,1.05fr)_minmax(0,1fr)]">
          <div className="border-line px-6 py-5 md:border-r">
            <div className="text-[11px] tracking-[.14em] text-faint uppercase">本章核心公式</div>
            <MB className="!my-3">{g.formula}</MB>
            <p className="text-[13px] leading-relaxed text-dim">{g.formulaNote}</p>

            <div className="mt-6 text-[11px] tracking-[.14em] text-faint uppercase">
              带得走的结论
            </div>
            <ol className="mt-2.5 space-y-2">
              {g.takeaways.map((t, i) => (
                <li key={i} className="flex gap-2.5 text-[14px] leading-[1.8] text-ink">
                  <span className="mt-[3px] font-mono text-[11px] text-brand">{i + 1}</span>
                  <span>{t}</span>
                </li>
              ))}
            </ol>
          </div>

          <div className="px-6 py-5">
            <div className="text-[11px] tracking-[.14em] text-warn uppercase">最容易搞错的地方</div>
            <ul className="mt-2.5 space-y-3">
              {g.traps.map((t, i) => (
                <li key={i} className="flex gap-2.5 text-[13.5px] leading-[1.8] text-dim">
                  <span className="mt-[7px] h-1 w-1 shrink-0 rounded-full bg-warn" />
                  <span>{t}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}

export function ChapterHero({
  n,
  hook,
  lead,
  gains,
}: {
  n: number
  /** 上一章欠下的那个问题 */
  hook: string
  lead: ReactNode
  gains: string[]
}) {
  const meta = CHAPTERS.find((c) => c.n === n)!
  return (
    <section className="mx-auto max-w-6xl px-6 pt-14 pb-8">
      <div className="grid gap-10 lg:grid-cols-[minmax(0,1fr)_300px]">
        <div>
          <div className="flex items-baseline gap-3">
            <span className="font-mono text-[13px] tracking-[.22em] text-brand">第 {n} 章</span>
          </div>
          <h1 className="mt-2 font-serif text-[38px] leading-tight font-black text-ink md:text-[50px]">
            {meta.title}
          </h1>

          <div className="mt-6 border-l-2 border-warn/50 pl-5">
            <div className="text-[11px] tracking-[.18em] text-warn uppercase">上一章留下的问题</div>
            <p className="mt-1.5 font-serif text-[18px] leading-relaxed text-ink italic">
              「{hook}」
            </p>
          </div>

          <div className="prose-body mt-6 max-w-2xl">{lead}</div>
        </div>

        <aside className="lg:pt-14">
          <div className="card rounded-2xl p-5">
            <div className="text-[11px] tracking-[.16em] text-faint uppercase">读完这一章，你会</div>
            <ul className="mt-3 space-y-2.5">
              {gains.map((g) => (
                <li key={g} className="flex gap-2.5 text-[13.5px] leading-relaxed text-dim">
                  <span className="mt-[7px] h-1 w-1 shrink-0 rounded-full bg-brand" />
                  <span>{g}</span>
                </li>
              ))}
            </ul>
          </div>
        </aside>
      </div>
    </section>
  )
}
