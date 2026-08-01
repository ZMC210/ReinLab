import { ACTIONS } from '../core/mdp'
import { bellmanBreakdown } from '../core/solvers'
import { fmt } from '../theme'
import type { FCtx, FNode } from './core'

export const stateTex = (s: number) => `s_{${s + 1}}`
const col = (c: string, body: string) => `\\textcolor{${c}}{${body}}`
const numTex = (x: number, c: string) => col(c, fmt(x))

/* ────────────────────────── 通用符号节点 ────────────────────────── */

const gammaNode: FNode = {
  name: '折扣因子 γ',
  desc: 'γ 越接近 1，智能体越有耐心、越看重远期回报；γ 越接近 0，越只顾眼前。它不是一个技术参数，它定义了智能体的「时间偏好」。',
  entities: () => [{ kind: 'gamma' }],
  value: (c) => c.gamma,
  render: ({ ctx }) => col(ctx.colors.gamma, '\\gamma'),
}

const vSymbolic = (sExpr: string, sIdx: (c: FCtx) => number, isNext = false): FNode => ({
  name: isNext ? '后继状态的价值' : '状态价值 v_π',
  desc: isNext
    ? '这就是递归所在：v 的定义里又出现了 v。贝尔曼公式不是在「算出」价值，而是在陈述价值之间必须满足的关系。'
    : '从这个状态出发、一直遵循策略 π 走下去，未来折扣回报的期望。',
  entities: (c) => [{ kind: 'value', s: sIdx(c) }],
  value: (c) => c.v[sIdx(c)],
  render: ({ ctx }) => `${col(ctx.colors.value, 'v_\\pi')}(${sExpr})`,
})

/* ────────────────────────── 具体数值节点 ────────────────────────── */

const piNum = (s: number, a: number): FNode => ({
  name: `π(${ACTIONS[a].tex} | ${stateTex(s)})`,
  desc: `策略在这个状态选择「${ACTIONS[a].name} ${ACTIONS[a].glyph}」的概率。`,
  entities: () => [{ kind: 'action', s, a }],
  value: (c) => c.pi[s][a],
  render: ({ ctx }) => numTex(ctx.pi[s][a], ctx.colors.policy),
})

const rNum = (s: number, a: number): FNode => ({
  name: `r(${stateTex(s)}, ${ACTIONS[a].tex})`,
  desc: '在这里采取这个动作，立刻拿到的奖励。注意它只描述「当下」，与未来无关。',
  entities: () => [{ kind: 'reward', s, a }],
  value: (c) => c.mdp.R[s][a],
  render: ({ ctx }) => numTex(ctx.mdp.R[s][a], ctx.colors.reward),
})

const vNum = (s: number, sp: number, a: number): FNode => ({
  name: `v_π(${stateTex(sp)})`,
  desc: '这个数字就是通往世界的入口 —— 把鼠标停在它上面，看看是网格里的哪一格。',
  entities: () => [
    { kind: 'value', s: sp },
    { kind: 'transition', s, a, sp },
  ],
  value: (c) => c.v[sp],
  render: ({ ctx }) => numTex(ctx.v[sp], ctx.colors.value),
})

const pNum = (s: number, a: number, sp: number): FNode => ({
  name: `p(${stateTex(sp)} | ${stateTex(s)}, ${ACTIONS[a].tex})`,
  desc: '状态转移概率。在这个网格世界里转移是确定性的，所以它非 0 即 1。',
  entities: () => [{ kind: 'transition', s, a, sp }],
  value: (c) => c.mdp.P[s][a][sp],
  render: ({ ctx }) => numTex(ctx.mdp.P[s][a][sp], ctx.colors.prob),
})

/* ────────────────────────── 逐元素贝尔曼公式 ────────────────────────── */

export type BellmanVariant = 'compact' | 'full'

/**
 * v_π(s) = Σ_a π(a|s) [ r(s,a) + γ Σ_{s'} p(s'|s,a) v_π(s') ]
 *
 * symbolic 模式给出教材上的一般形式；numeric 模式把当前聚焦状态的每一项
 * 都代入真实数值展开成多行 —— 而每一个数字仍然是活的，悬停即可回指世界。
 */
export function bellmanElementwise(variant: BellmanVariant = 'compact'): FNode {
  const piG: FNode = {
    name: '策略 π(a|s)',
    desc: '在状态 s 下选择动作 a 的概率。策略就是「每个状态该怎么掷骰子」。',
    entities: (c) => [{ kind: 'policy', s: c.s }],
    render: ({ ctx }) => `${col(ctx.colors.policy, '\\pi')}(a \\mid s)`,
  }

  const rG: FNode = {
    name: '立即奖励的期望 r(s,a)',
    desc: '当下这一步的平均收益。它是贝尔曼公式里唯一「不含未来」的部分。',
    entities: (c) => [{ kind: 'reward', s: c.s, a: 0 }],
    render: ({ ctx }) => `${col(ctx.colors.reward, 'r')}(s, a)`,
  }

  const rFull: FNode = {
    name: 'Σ_r p(r|s,a)·r',
    desc: '奖励本身可以是随机的，所以要对奖励的分布求期望。它等价于紧凑写法里的 r(s,a)。',
    render: ({ ctx }) =>
      `\\sum_{r} ${col(ctx.colors.prob, 'p')}(r \\mid s,a)\\, ${col(ctx.colors.reward, 'r')}`,
  }

  const nextTermG: FNode = {
    name: '未来价值的期望',
    desc: '走一步之后落到各个后继状态的概率，乘上那些状态自己的价值 —— 这一项把「现在」和「以后」缝在了一起。',
    entities: (c) => [{ kind: 'state', s: c.s }],
    render: ({ ctx, child }) =>
      `\\sum_{s' \\in \\mathcal{S}} ${col(ctx.colors.prob, 'p')}(s' \\mid s,a)\\, ${child(
        vSymbolic("s'", (c) => c.s, true),
        'v',
      )}`,
  }

  const bracket: FNode = {
    name: '动作价值 q_π(s,a)',
    desc: '方括号里的整体就是 q_π(s,a)：在 s 处执行 a、此后遵循 π 的期望回报。贝尔曼公式其实是 v = Σ π·q。',
    render: ({ child }) =>
      `\\left[ ${child(variant === 'full' ? rFull : rG, 'r')} + ${child(gammaNode, 'g')} ${child(
        nextTermG,
        'n',
      )} \\right]`,
  }

  const rhsSym: FNode = {
    name: '对所有动作求期望',
    desc: '策略是随机的，所以要把每个动作的结果按概率加权平均。',
    entities: (c) => [{ kind: 'policy', s: c.s }],
    render: ({ child }) => `\\sum_{a \\in \\mathcal{A}} ${child(piG, 'p')} ${child(bracket, 'b')}`,
  }

  return {
    render: (args) => {
      const { mode, ctx, child } = args
      const lhs = child(
        vSymbolic(mode === 'numeric' ? stateTex(ctx.s) : 's', (c) => c.s),
        'lhs',
      )
      if (mode === 'symbolic') return `${lhs} = ${child(rhsSym, 'rhs')}`

      const { actions, total } = bellmanBreakdown(ctx.mdp, ctx.pi, ctx.gamma, ctx.v, ctx.s)
      const live = actions.filter((x) => x.prob > 1e-9)
      const lines: string[] = []

      live.forEach((br, i) => {
        const inner = br.successors
          .map((su) => {
            const vTex = child(vNum(ctx.s, su.sp, br.a), `v${br.a}_${su.sp}`)
            return su.p === 1
              ? vTex
              : `${child(pNum(ctx.s, br.a, su.sp), `p${br.a}_${su.sp}`)} \\cdot ${vTex}`
          })
          .join(' + ')
        const term = `${child(piNum(ctx.s, br.a), `pi${br.a}`)} \\cdot \\left[ ${child(
          rNum(ctx.s, br.a),
          `r${br.a}`,
        )} + ${child(gammaNode, `g${br.a}`)} \\cdot ${inner} \\right]`
        lines.push(`${i === 0 ? '&=' : '&+'}\\; ${term}`)
      })

      const omitted = actions.length - live.length
      const note =
        omitted > 0
          ? `\\\\[2pt] &\\qquad {\\scriptstyle \\text{(概率为 0 的 ${omitted} 个动作已略去)}}`
          : ''

      return `\\begin{aligned} ${lhs} ${lines.join(' \\\\[2pt] ')} \\\\[4pt] &= ${col(
        ctx.colors.value,
        fmt(total, 3),
      )} ${note} \\end{aligned}`
    },
  }
}

/* ────────────────────────── 矩阵-向量形式 ────────────────────────── */

export function bellmanMatrix(): FNode {
  const vVec: FNode = {
    name: '价值向量 v_π',
    desc: '把所有状态的价值摞成一个列向量。n 个状态就是 n 维。',
    render: ({ ctx }) => col(ctx.colors.value, 'v_\\pi'),
  }
  const rVec: FNode = {
    name: '奖励向量 r_π',
    desc: 'r_π[s] = Σ_a π(a|s) r(s,a)，即在策略 π 下每个状态的平均即时奖励。',
    render: ({ ctx }) => col(ctx.colors.reward, 'r_\\pi'),
  }
  const PMat: FNode = {
    name: '状态转移矩阵 P_π',
    desc: 'P_π 的第 s 行，就是从格子 s 出发一步之后会落在哪里的概率分布 —— 矩阵的一行等于网格上的一次「出发」。',
    render: ({ ctx }) => col(ctx.colors.prob, 'P_\\pi'),
  }

  return {
    render: ({ mode, child }) =>
      mode === 'symbolic'
        ? `${child(vVec, 'v')} = ${child(rVec, 'r')} + ${child(gammaNode, 'g')} ${child(
            PMat,
            'P',
          )} ${child(vVec, 'v2')}`
        : `${child(vVec, 'v')} = (I - ${child(gammaNode, 'g')} ${child(PMat, 'P')})^{-1} ${child(
            rVec,
            'r',
          )}`,
  }
}

/* ────────────────────────── 动作价值与回报 ────────────────────────── */

export function qDefinition(): FNode {
  const q: FNode = {
    name: '动作价值 q_π(s,a)',
    desc: '在 s 处「先执行 a」，之后才遵循 π。它比 v 多问了一句：如果我这一步偏离策略呢？',
    entities: (c) => [{ kind: 'qvalue', s: c.s, a: 0 }],
    render: ({ ctx }) => `${col(ctx.colors.qvalue, 'q_\\pi')}(s,a)`,
  }
  const nextTerm: FNode = {
    name: '折扣后的未来',
    desc: '一步之后的期望价值，被 γ 打了折。',
    render: ({ ctx, child }) =>
      `\\sum_{s'} ${col(ctx.colors.prob, 'p')}(s' \\mid s,a)\\, ${child(
        vSymbolic("s'", (c) => c.s, true),
        'v',
      )}`,
  }
  return {
    render: ({ ctx, child }) =>
      `${child(q, 'q')} = ${col(ctx.colors.reward, 'r')}(s,a) + ${child(gammaNode, 'g')} ${child(
        nextTerm,
        'n',
      )}`,
  }
}

export function vFromQ(): FNode {
  return {
    render: ({ ctx, child }) =>
      `${child(
        vSymbolic('s', (c) => c.s),
        'v',
      )} = \\sum_{a \\in \\mathcal{A}} ${col(ctx.colors.policy, '\\pi')}(a \\mid s)\\, ${col(
        ctx.colors.qvalue,
        'q_\\pi',
      )}(s,a)`,
  }
}

export function returnDefinition(): FNode {
  const g: FNode = {
    name: '折扣回报 G_t',
    desc: '从时刻 t 之后拿到的所有奖励，越往后打折越狠，加总起来。强化学习要最大化的是它，不是单步奖励。',
    render: ({ ctx }) => col(ctx.colors.value, 'G_t'),
  }
  return {
    render: ({ child }) =>
      `${child(g, 'g')} = R_{t+1} + ${child(gammaNode, 'g1')} R_{t+2} + ${child(
        gammaNode,
        'g2',
      )}^{2} R_{t+3} + \\cdots = \\sum_{k=0}^{\\infty} ${child(gammaNode, 'g3')}^{k} R_{t+k+1}`,
  }
}

export function valueAsExpectation(): FNode {
  return {
    render: ({ ctx, child }) =>
      `${child(
        vSymbolic('s', (c) => c.s),
        'v',
      )} = \\mathbb{E}\\!\\left[ ${col(ctx.colors.value, 'G_t')} \\mid S_t = s \\right]`,
  }
}

/* ────────────────────────── 贝尔曼最优公式 ────────────────────────── */

export function bellmanOptimality(form: 'pi' | 'max' = 'max'): FNode {
  const vStar = (sExpr: string): FNode => ({
    name: '最优状态价值 v*',
    desc: 'v* 是所有策略里能达到的最大价值。它的存在性与唯一性由压缩映射定理保证。',
    entities: (c) => [{ kind: 'value', s: c.s }],
    value: (c) => c.v[c.s],
    render: ({ ctx }) => `${col(ctx.colors.value, 'v^{*}')}(${sExpr})`,
  })

  const qStar: FNode = {
    name: 'q(s,a)',
    desc: '用当前的 v 估计算出来的动作价值。max 就在这些数之间挑最大的那个。',
    entities: (c) => [{ kind: 'qvalue', s: c.s, a: 0 }],
    render: ({ ctx }) => `${col(ctx.colors.qvalue, 'q')}(s,a)`,
  }

  const maxNode: FNode = {
    name: 'max —— 让方程变成非线性的那一笔',
    desc: '和贝尔曼公式相比，全部差别只在这一个 max 上。可正是它让方程不再是线性方程组，没法一步解出来。',
    render: ({ ctx, child }) =>
      form === 'max'
        ? `\\max_{a \\in \\mathcal{A}} ${child(qStar, 'q')}`
        : `\\max_{\\pi(s) \\in \\Pi} \\sum_{a} ${col(ctx.colors.policy, '\\pi')}(a\\mid s)\\, ${child(
            qStar,
            'q',
          )}`,
  }

  return {
    render: ({ ctx, mode, child }) =>
      `${child(vStar(mode === 'numeric' ? stateTex(ctx.s) : 's'), 'v')} = ${child(maxNode, 'm')}`,
  }
}

export function contractionStatement(): FNode {
  return {
    render: ({ child }) =>
      `\\left\\| f(v_1) - f(v_2) \\right\\|_{\\infty} \\;\\le\\; ${child(
        gammaNode,
        'g',
      )} \\left\\| v_1 - v_2 \\right\\|_{\\infty}`,
  }
}

/* ────────────────────────── 后续章节的公式 ────────────────────────── */

/** 增量式均值：第 6 章的敲门砖，也是 TD 更新的骨架 */
export function incrementalMean(): FNode {
  const w: FNode = {
    name: '当前估计',
    desc: '旧的估计值。整个强化学习的更新式几乎都长成「旧估计 + 步长 × 误差」。',
    render: ({ ctx }) => col(ctx.colors.value, 'w_k'),
  }
  const alpha: FNode = {
    name: '步长 α_k',
    desc: '每次朝误差方向挪多少。太大震荡，太小停滞 —— Robbins-Monro 条件说的就是它该怎么衰减。',
    render: ({ ctx }) => col(ctx.colors.gamma, '\\alpha_k'),
  }
  const err: FNode = {
    name: '误差项',
    desc: '新样本与当前估计的差。它是驱动更新的唯一动力，误差为 0 时估计就不动了。',
    render: ({ ctx }) => `\\left( ${col(ctx.colors.reward, 'x_k')} - ${col(ctx.colors.value, 'w_k')} \\right)`,
  }
  return {
    render: ({ ctx, child }) =>
      `${col(ctx.colors.value, 'w_{k+1}')} = ${child(w, 'w')} + ${child(alpha, 'a')} ${child(
        err,
        'e',
      )}`,
  }
}

/** TD(0) 的更新式，把「TD 目标」和「TD 误差」拆开标注 */
export function tdUpdate(): FNode {
  const target: FNode = {
    name: 'TD 目标',
    desc: '用「一步真实奖励 + 折扣后的下一状态估计」拼出来的目标值。它比蒙特卡洛的目标偏（因为用了估计），但方差小得多。',
    render: ({ ctx }) =>
      `\\underbrace{${col(ctx.colors.reward, 'r_{t+1}')} + ${col(ctx.colors.gamma, '\\gamma')} ${col(
        ctx.colors.value,
        'v(s_{t+1})',
      )}}_{\\text{TD 目标}}`,
  }
  const err: FNode = {
    name: 'TD 误差 δ_t',
    desc: '目标与当前估计之差。它是整个时序差分方法的心脏，第 9、10 章还会以「优势」的名义再见到它。',
    render: ({ ctx, child }) =>
      `\\big[ ${child(target, 't')} - ${col(ctx.colors.value, 'v(s_t)')} \\big]`,
  }
  return {
    render: ({ ctx, child }) =>
      `${col(ctx.colors.value, 'v(s_t)')} \\leftarrow ${col(ctx.colors.value, 'v(s_t)')} + ${col(
        ctx.colors.gamma,
        '\\alpha',
      )} ${child(err, 'e')}`,
  }
}

/** 策略梯度定理 */
export function policyGradient(): FNode {
  const grad: FNode = {
    name: '目标函数的梯度',
    desc: '注意它是一个期望 —— 意味着可以用采样来估计，这正是 REINFORCE 的立足点。',
    render: ({ ctx }) => `\\nabla_\\theta ${col(ctx.colors.value, 'J(\\theta)')}`,
  }
  const score: FNode = {
    name: '得分函数 ∇ln π',
    desc: '「把这个动作的概率往上推」的方向。乘上 q 之后，好动作被推得多，坏动作被推得少甚至往下压。',
    render: ({ ctx }) =>
      `\\nabla_\\theta \\ln ${col(ctx.colors.policy, '\\pi')}(a \\mid s, \\theta)`,
  }
  const q: FNode = {
    name: '动作价值 q_π(s,a)',
    desc: '这个动作到底好不好。它是未知的 —— 谁来估它，决定了你得到的是 REINFORCE 还是 Actor-Critic。',
    render: ({ ctx }) => col(ctx.colors.qvalue, 'q_\\pi(s,a)'),
  }
  return {
    render: ({ child }) =>
      `${child(grad, 'g')} = \\mathbb{E}_{S \\sim \\eta,\\, A \\sim \\pi} \\left[ ${child(
        score,
        's',
      )} \\, ${child(q, 'q')} \\right]`,
  }
}
