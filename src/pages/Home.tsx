import { useMemo, useState } from 'react'
import { buildGridMDP, classicGrid } from '../core/mdp'
import { valueIterationSolve } from '../core/solvers'
import { bellmanElementwise } from '../formula/bellman'
import { LiveFormula } from '../formula/LiveFormula'
import { useFormulaCtx } from '../formula/core'
import { GridWorld } from '../viz/GridWorld'
import { StorylineMap } from '../narrative/Storyline'
import { FrameworkMap } from '../narrative/FrameworkMap'
import { Slider } from '../ui/prims'
import { useBus } from '../highlight/bus'

/** 公式树是无状态的，放在模块级即可，避免每次渲染重建 */
const BELLMAN = bellmanElementwise('compact')

function HeroDemo() {
  const [gamma, setGamma] = useState(0.9)
  const mdp = useMemo(() => buildGridMDP(classicGrid()), [])
  const focus = useBus((s) => s.focus)
  const setFocus = useBus((s) => s.setFocus)

  const { v, policy, ties } = useMemo(() => valueIterationSolve(mdp, gamma), [mdp, gamma])

  const s = focus ?? 8
  const ctx = useFormulaCtx(mdp, policy, gamma, v, s)

  return (
    <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_minmax(0,420px)]">
      <div className="rounded-3xl border border-line bg-surface p-6">
        <div className="mb-4 flex items-center justify-between gap-4">
          <span className="text-[11px] tracking-[.18em] text-faint uppercase">
            贝尔曼公式 · 把鼠标放到任意一段上
          </span>
        </div>
        <LiveFormula node={BELLMAN} ctx={ctx} />
        <div className="mt-5 border-t border-line pt-4">
          <Slider
            label={
              <>
                折扣因子 <span className="font-mono text-brand">γ</span>
              </>
            }
            value={gamma}
            min={0.1}
            max={0.99}
            step={0.01}
            onChange={setGamma}
            hint="拖动它。看着整个价值场重新流动，然后注意某些格子上的箭头会突然翻向另一边。"
            accent="var(--gamma)"
          />
        </div>
      </div>

      <div className="rounded-3xl border border-line bg-surface p-5">
        <GridWorld
          mdp={mdp}
          v={v}
          policy={policy}
          ties={ties}
          cell={62}
          onCellClick={(x) => setFocus(x === focus ? null : x)}
        />
        <p className="mt-3 text-center text-[11.5px] text-faint">
          点一个格子把它钉住，公式就会围绕它展开
        </p>
      </div>
    </div>
  )
}

const PRINCIPLES = [
  {
    k: '公式与世界，同屏且相连',
    d: '认知负荷理论里的「注意力分散效应」说：当图、公式、文字彼此分离时，工作记忆会被消耗在建立对应关系上，真正的思考反而没有余量。所以这里的每一个符号都和世界里的元素双向绑定 —— 对应关系不需要你去记。',
  },
  {
    k: '先下注，再揭晓',
    d: '被动读懂和主动答对是两件事。每个核心概念之前都有一道必须先做出承诺的题。猜错不是失败，猜错时被纠正的那一刻，记得最牢。',
  },
  {
    k: '按认知顺序，不按逻辑顺序',
    d: '教材的顺序是「定义→定理→证明」，可人脑的顺序是「困惑→尝试→失败→修正」。这里刻意保留了那些走不通的笨办法，因为你自己想的时候，多半就会想到它们。',
  },
  {
    k: '直觉与数学，一个都不能少',
    d: '每个结论都配一句人话的直觉，也配一份完整的推导。想快的人可以只看直觉，想扎实的人可以逐行展开证明 —— 深度由你控制，但严谨性从不打折。',
  },
]

export function Home() {
  return (
    <div>
      <section className="mx-auto max-w-6xl px-6 pt-20 pb-16">
        <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-line bg-surface px-3 py-1">
          <span className="h-1.5 w-1.5 rounded-full bg-brand" />
          <span className="text-[11.5px] tracking-wide text-dim">
            ReinLab · 交互式强化学习实验室
          </span>
        </div>
        <h1 className="font-serif text-[42px] leading-[1.15] font-black text-ink md:text-[62px]">
          ReinLab
        </h1>
        <p className="mt-3 font-serif text-[22px] leading-snug text-dim md:text-[28px]">
          看得见的
          <span className="bg-gradient-to-r from-brand to-alt bg-clip-text text-transparent">
            强化学习
          </span>
        </p>
        <p className="mt-5 max-w-2xl text-[17px] leading-[1.9] text-dim">
          从贝尔曼公式一路走到 Actor-Critic。
          公式不再是纸上的符号 —— 把鼠标放上去，它指向的那一格就会亮起来；
          拖动一个参数，整个世界会当着你的面重新排列。
        </p>
        <div className="mt-8 flex flex-wrap items-center gap-3">
          <a
            href="#/ch1"
            className="rounded-xl bg-brand px-5 py-2.5 text-[14px] font-medium text-white shadow-sm transition-transform hover:scale-[1.02]"
          >
            从第 1 章开始
          </a>
          <a
            href="#/ch2"
            className="rounded-xl border border-line px-5 py-2.5 text-[14px] text-dim transition-colors hover:border-brand/40 hover:text-brand"
          >
            直接看旗舰章：贝尔曼公式
          </a>
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-6 pb-20">
        <HeroDemo />
      </section>

      <section className="border-t border-line py-20">
        <div className="mx-auto max-w-6xl px-6">
          <h2 className="font-serif text-[28px] font-bold text-ink">全景地图：整个强化学习长什么样</h2>
          <p className="mt-2 mb-8 max-w-2xl text-[14.5px] leading-relaxed text-faint">
            在钻进任何一个公式之前，先把这张图看懂。它只回答一个问题：
            <strong className="text-dim">这么多算法，到底是按什么分岔出来的？</strong>
            后面每一章开头，你都会在这张图上看到自己站在哪。
          </p>
          <FrameworkMap onGo={(id) => (location.hash = `#/${id}`)} />
        </div>
      </section>

      <section className="border-t border-line py-20">
        <div className="mx-auto max-w-6xl px-6">
          <h2 className="font-serif text-[28px] font-bold text-ink">这套教程为什么这样做</h2>
          <p className="mt-2 max-w-2xl text-[14.5px] leading-relaxed text-faint">
            每一个设计决定背后都有一条教育学上的理由，而不是「看起来比较酷」。
          </p>
          <div className="mt-9 grid gap-5 md:grid-cols-2">
            {PRINCIPLES.map((p, i) => (
              <div
                key={p.k}
                className="rounded-2xl border border-line bg-surface p-6 transition-colors hover:border-brand/25"
              >
                <div className="mb-2 flex items-baseline gap-3">
                  <span className="font-mono text-[12px] text-brand/70">0{i + 1}</span>
                  <h3 className="font-serif text-[17px] font-bold text-ink">{p.k}</h3>
                </div>
                <p className="text-[14px] leading-[1.85] text-dim">{p.d}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="border-t border-line py-20">
        <div className="mx-auto max-w-3xl px-6">
          <h2 className="font-serif text-[28px] font-bold text-ink">全书主线</h2>
          <p className="mt-2 mb-8 text-[14.5px] leading-relaxed text-faint">
            十章不是十个并列的知识点。每一章都在偿还上一章欠下的债 ——
            顺着这条链读下去，你会一直知道自己为什么在这里。
          </p>
          <StorylineMap onGo={(c) => (location.hash = `#/${c.id}`)} />
        </div>
      </section>
    </div>
  )
}
