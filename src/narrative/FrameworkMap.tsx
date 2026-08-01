import { useState } from 'react'
import { useColors } from '../theme'

/**
 * 全书全景图。
 *
 * 学生学不进去，往往不是因为某个公式太难，而是因为不知道自己在哪 ——
 * 「这么多算法到底按什么分岔出来的」这个问题一天不解决，
 * 每一章都像是凭空掉下来的。
 *
 * 所以这张图只回答一件事：分岔点在哪里。
 *   · 有没有模型（p 和 r 已知吗）        → 第 4 章 vs 第 5 章之后
 *   · 学价值还是学策略                   → value-based vs policy-based
 *   · 用整条轨迹还是只用一步             → 蒙特卡洛 vs 时序差分
 *   · 表格装得下吗                       → 表格法 vs 函数近似
 *   · 用谁来估 q                         → REINFORCE vs Actor-Critic
 */

type Tone = 'base' | 'value' | 'policy' | 'bridge' | 'ext'

interface Node {
  id: string
  /** 有章节则可点击跳转 */
  ch?: string
  x: number
  y: number
  w: number
  h: number
  title: string
  sub?: string
  tone: Tone
  /** 悬停时的一句话解释 */
  note?: string
}

interface Edge {
  from: string
  to: string
  label?: string
  dashed?: boolean
}

const W = 1060
const H = 560

const NODES: Node[] = [
  // ── 地基：有模型的世界 ──────────────────────────────
  {
    id: 'n1',
    ch: 'ch1',
    x: 16,
    y: 150,
    w: 176,
    h: 44,
    title: '① 基本概念',
    sub: '状态 · 动作 · 回报 · MDP',
    tone: 'base',
    note: '先把游戏规则说清楚：谁在动、动了会怎样、什么算好。',
  },
  {
    id: 'n2',
    ch: 'ch2',
    x: 16,
    y: 208,
    w: 176,
    h: 44,
    title: '② 贝尔曼公式',
    sub: '给定 π，v_π 是多少',
    tone: 'base',
    note: '把「未来」写成递归。整个强化学习的地基。',
  },
  {
    id: 'n3',
    ch: 'ch3',
    x: 16,
    y: 266,
    w: 176,
    h: 44,
    title: '③ 贝尔曼最优公式',
    sub: '最好的 π 长什么样',
    tone: 'base',
    note: '多出一个 max，方程从线性变非线性 —— 也从此有了「最优」。',
  },
  {
    id: 'n4',
    ch: 'ch4',
    x: 16,
    y: 324,
    w: 176,
    h: 44,
    title: '④ 值迭代 / 策略迭代',
    sub: '有模型时怎么解',
    tone: 'base',
    note: '知道 p 和 r 的情况下，把最优方程真的解出来。',
  },

  // ── 分岔口 ────────────────────────────────────────
  {
    id: 'free',
    x: 246,
    y: 248,
    w: 150,
    h: 46,
    title: '不知道 p 和 r',
    sub: 'model-free',
    tone: 'bridge',
    note: '现实里模型几乎从来不给你。只能靠采样 —— 从这里开始整本书换了一个世界。',
  },
  {
    id: 'sa',
    ch: 'ch6',
    x: 246,
    y: 328,
    w: 150,
    h: 46,
    title: '⑥ 随机近似 / SGD',
    sub: '用样本代替期望',
    tone: 'bridge',
    note: '所有 model-free 算法共用的数学引擎：为什么带噪声的更新也能收敛。',
  },

  // ── value-based ──────────────────────────────────
  {
    id: 'vb',
    x: 246,
    y: 96,
    w: 150,
    h: 56,
    title: 'value-based',
    sub: '先估 q(s,a)，再贪心',
    tone: 'value',
    note: '策略不直接学，而是从价值里「读」出来：谁的 q 大就选谁。',
  },
  {
    id: 'mc',
    ch: 'ch5',
    x: 450,
    y: 34,
    w: 158,
    h: 46,
    title: '⑤ 蒙特卡洛',
    sub: '跑完整条轨迹再更新',
    tone: 'value',
    note: '无偏，但必须等到回合结束，方差也大。',
  },
  {
    id: 'td',
    ch: 'ch7',
    x: 450,
    y: 112,
    w: 158,
    h: 46,
    title: '⑦ 时序差分',
    sub: '走一步就更新',
    tone: 'value',
    note: '用估计去更新估计（自举）。有偏，但方差小、可在线学习。',
  },
  {
    id: 'sarsa',
    ch: 'ch7',
    x: 664,
    y: 72,
    w: 140,
    h: 40,
    title: 'Sarsa',
    sub: 'on-policy',
    tone: 'value',
    note: '用「我实际会走的那个动作」来估计 —— 因此它会怕悬崖。',
  },
  {
    id: 'ql',
    ch: 'ch7',
    x: 664,
    y: 130,
    w: 140,
    h: 40,
    title: 'Q-learning',
    sub: 'off-policy',
    tone: 'value',
    note: '用「最好的那个动作」来估计 —— 因此它敢贴着悬崖走。',
  },
  {
    id: 'fa',
    ch: 'ch8',
    x: 858,
    y: 72,
    w: 186,
    h: 46,
    title: '⑧ 值函数近似',
    sub: '表格装不下时',
    tone: 'value',
    note: '状态太多，只能用参数化函数去拟合价值。从这里开始「深度」才有意义。',
  },
  {
    id: 'dqn',
    x: 858,
    y: 132,
    w: 186,
    h: 40,
    title: 'DQN',
    sub: '经验回放 + 目标网络',
    tone: 'ext',
    note: 'Q-learning + 神经网络，再加两个稳定化技巧。',
  },

  // ── policy-based ─────────────────────────────────
  {
    id: 'pb',
    x: 246,
    y: 424,
    w: 150,
    h: 56,
    title: 'policy-based',
    sub: '直接把 π 参数化',
    tone: 'policy',
    note: '不再绕道价值，直接对策略求梯度。天然支持连续动作和随机策略。',
  },
  {
    id: 'pg',
    ch: 'ch9',
    x: 450,
    y: 400,
    w: 158,
    h: 46,
    title: '⑨ 策略梯度',
    sub: '∇J = E[∇lnπ · q]',
    tone: 'policy',
    note: '策略梯度定理。剩下的全部问题只剩一个：q 谁来估？',
  },
  {
    id: 'rf',
    ch: 'ch9',
    x: 664,
    y: 370,
    w: 140,
    h: 40,
    title: 'REINFORCE',
    sub: '用 MC 估 q',
    tone: 'policy',
    note: '最朴素的答案：跑完一整条轨迹，用真实回报当 q。',
  },
  {
    id: 'ac',
    ch: 'ch10',
    x: 664,
    y: 428,
    w: 140,
    h: 40,
    title: '⑩ Actor-Critic',
    sub: '用 TD 估 q',
    tone: 'policy',
    note: '换成 TD 来估 q，于是价值方法和策略方法在这里合流。',
  },
  {
    id: 'a2c',
    ch: 'ch10',
    x: 858,
    y: 398,
    w: 186,
    h: 40,
    title: 'A2C / 带基线',
    sub: '减方差',
    tone: 'policy',
    note: '减去一个基线不改变期望，却能大幅降低方差 —— 优势函数由此而来。',
  },
  {
    id: 'ppo',
    x: 858,
    y: 452,
    w: 186,
    h: 40,
    title: 'TRPO / PPO / GRPO',
    sub: '现代大模型对齐',
    tone: 'ext',
    note: '给策略更新加上「别一步迈太大」的约束。RLHF 用的就是这一支。',
  },
]

const EDGES: Edge[] = [
  { from: 'n1', to: 'n2' },
  { from: 'n2', to: 'n3' },
  { from: 'n3', to: 'n4' },
  { from: 'n4', to: 'free', label: '模型没了' },
  { from: 'free', to: 'sa', label: '靠什么补' },
  { from: 'free', to: 'vb', label: '学价值' },
  { from: 'free', to: 'pb', label: '学策略' },
  { from: 'vb', to: 'mc', label: '整条轨迹' },
  { from: 'vb', to: 'td', label: '只用一步' },
  { from: 'td', to: 'sarsa' },
  { from: 'td', to: 'ql' },
  { from: 'sarsa', to: 'fa', label: '状态太多' },
  { from: 'ql', to: 'fa' },
  { from: 'fa', to: 'dqn' },
  { from: 'pb', to: 'pg' },
  { from: 'pg', to: 'rf', label: 'q 用采样' },
  { from: 'pg', to: 'ac', label: 'q 用网络' },
  { from: 'ac', to: 'a2c' },
  { from: 'a2c', to: 'ppo' },
  { from: 'sa', to: 'td', dashed: true },
  { from: 'sa', to: 'pg', dashed: true },
]

const byId = new Map(NODES.map((n) => [n.id, n]))

function toneColor(t: Tone, C: ReturnType<typeof useColors>) {
  switch (t) {
    case 'value':
      return C.value
    case 'policy':
      return C.policy
    case 'bridge':
      return C.reward
    case 'ext':
      return C.qvalue
    default:
      return C.accent
  }
}

/** 直角折线：先横到中点，再竖，再横过去 */
function elbow(a: Node, b: Node): string {
  const x1 = a.x + a.w
  const y1 = a.y + a.h / 2
  const x2 = b.x
  const y2 = b.y + b.h / 2

  // 同一列上下相连（地基链）
  if (Math.abs(a.x - b.x) < 1 && b.y > a.y) {
    const cx = a.x + a.w / 2
    return `M ${cx} ${a.y + a.h} V ${b.y}`
  }
  const mx = x1 + Math.max(18, (x2 - x1) / 2)
  return `M ${x1} ${y1} H ${mx} V ${y2} H ${x2}`
}

export function FrameworkMap({
  onGo,
  current,
  compact = false,
}: {
  onGo?: (chapterId: string) => void
  /** 当前所在章节，会被高亮 */
  current?: string
  compact?: boolean
}) {
  const C = useColors()
  const [hover, setHover] = useState<string | null>(null)

  const hovered = hover ? byId.get(hover) : null

  // 悬停时把与该节点相连的边和端点一起点亮，其余淡出
  const linked = new Set<string>()
  if (hover) {
    linked.add(hover)
    for (const e of EDGES) {
      if (e.from === hover) linked.add(e.to)
      if (e.to === hover) linked.add(e.from)
    }
  }
  const dim = (id: string) => (hover && !linked.has(id) ? 0.24 : 1)

  return (
    <div className="card overflow-hidden rounded-2xl">
      <div className="flex flex-wrap items-center gap-x-5 gap-y-2 border-b border-line px-5 py-3">
        <span className="text-[11px] font-semibold tracking-[.16em] text-faint uppercase">
          强化学习全景
        </span>
        {(
          [
            ['base', '有模型 · 地基'],
            ['bridge', '过渡：从期望到样本'],
            ['value', 'value-based'],
            ['policy', 'policy-based'],
            ['ext', '延伸（不展开讲）'],
          ] as [Tone, string][]
        ).map(([t, label]) => (
          <span key={t} className="flex items-center gap-1.5 text-[11.5px] text-dim">
            <span
              className="h-2 w-2 rounded-[3px]"
              style={{ background: toneColor(t, C), opacity: 0.85 }}
            />
            {label}
          </span>
        ))}
        <span className="ml-auto text-[11px] text-faint">悬停看解释 · 点击进入该章</span>
      </div>

      <div className="overflow-x-auto p-3">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          style={{ minWidth: compact ? 760 : 880 }}
          className="w-full"
          onMouseLeave={() => setHover(null)}
        >
          <defs>
            <marker id="fm-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">
              <path d="M0,0 L8,4 L0,8 z" fill={C.axis} />
            </marker>
          </defs>

          {/* 两大家族的背景带 */}
          <rect x={228} y={14} width={W - 240} height={172} rx={16} fill={C.value} opacity={0.05} />
          <rect x={228} y={382} width={W - 240} height={128} rx={16} fill={C.policy} opacity={0.05} />
          <text x={W - 22} y={34} textAnchor="end" fontSize={11} fill={C.value} opacity={0.75}>
            估价值 → 再取贪心
          </text>
          <text x={W - 22} y={402} textAnchor="end" fontSize={11} fill={C.policy} opacity={0.75}>
            直接对策略求梯度
          </text>

          {EDGES.map((e, i) => {
            const a = byId.get(e.from)!
            const b = byId.get(e.to)!
            const on = !hover || (linked.has(e.from) && linked.has(e.to))
            const mx = a.x + a.w + Math.max(18, (b.x - (a.x + a.w)) / 2)
            return (
              <g key={i} opacity={on ? 1 : 0.15} style={{ transition: 'opacity 180ms' }}>
                <path
                  d={elbow(a, b)}
                  fill="none"
                  stroke={C.axis}
                  strokeWidth={e.dashed ? 1.2 : 1.6}
                  strokeDasharray={e.dashed ? '4 4' : undefined}
                  markerEnd="url(#fm-arrow)"
                />
                {e.label && (
                  <text
                    x={mx + 4}
                    y={(a.y + a.h / 2 + b.y + b.h / 2) / 2 - 5}
                    fontSize={10.5}
                    fill={C.accent}
                    opacity={0.9}
                  >
                    {e.label}
                  </text>
                )}
              </g>
            )
          })}

          {NODES.map((n) => {
            const col = toneColor(n.tone, C)
            const isCur = current && n.ch === current
            return (
              <g
                key={n.id}
                opacity={dim(n.id)}
                style={{ transition: 'opacity 180ms', cursor: n.ch ? 'pointer' : 'default' }}
                onMouseEnter={() => setHover(n.id)}
                onClick={() => n.ch && onGo?.(n.ch)}
              >
                <rect
                  x={n.x}
                  y={n.y}
                  width={n.w}
                  height={n.h}
                  rx={10}
                  fill={`color-mix(in srgb, ${col} ${hover === n.id ? 20 : 11}%, var(--surface))`}
                  stroke={col}
                  strokeWidth={isCur ? 2.4 : hover === n.id ? 1.8 : 1}
                  strokeOpacity={isCur ? 1 : 0.55}
                  style={{ transition: 'all 180ms' }}
                />
                <text
                  x={n.x + 12}
                  y={n.y + (n.sub ? 20 : n.h / 2 + 4)}
                  fontSize={13}
                  fontWeight={600}
                  fill="var(--ink)"
                >
                  {n.title}
                </text>
                {n.sub && (
                  <text x={n.x + 12} y={n.y + 36} fontSize={10.5} fill="var(--ink-faint)">
                    {n.sub}
                  </text>
                )}
              </g>
            )
          })}
        </svg>
      </div>

      <div className="min-h-[52px] border-t border-line px-5 py-3">
        {hovered?.note ? (
          <p className="text-[13px] leading-relaxed text-dim">
            <span className="font-semibold text-ink">{hovered.title}</span>
            <span className="mx-2 text-faint">·</span>
            {hovered.note}
          </p>
        ) : (
          <p className="text-[13px] leading-relaxed text-faint">
            整张图只有四个分岔点：<strong className="text-dim">模型知不知道</strong>、
            <strong className="text-dim">学价值还是学策略</strong>、
            <strong className="text-dim">用整条轨迹还是只用一步</strong>、
            <strong className="text-dim">表格装不装得下</strong>。
            记住这四个问题，任何一个 RL 算法你都能给它安个位置。
          </p>
        )}
      </div>
    </div>
  )
}
