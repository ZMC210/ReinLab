import type { MDP } from './mdp'
import type { Policy } from './policy'
import { greedyFromQ, qFromV } from './policy'

export const normInf = (a: number[], b?: number[]) =>
  a.reduce((m, x, i) => Math.max(m, Math.abs(x - (b ? b[i] : 0))), 0)

/** r_π[s] = Σ_a π(a|s) r(s,a) */
export function rPi(mdp: MDP, pi: Policy): number[] {
  const out = new Array<number>(mdp.nS).fill(0)
  for (let s = 0; s < mdp.nS; s++) {
    let acc = 0
    for (let a = 0; a < mdp.nA; a++) {
      if (pi[s][a] !== 0) acc += pi[s][a] * mdp.R[s][a]
    }
    out[s] = acc
  }
  return out
}

/** P_π[s][s'] = Σ_a π(a|s) p(s'|s,a) */
export function PPi(mdp: MDP, pi: Policy): number[][] {
  const out: number[][] = []
  for (let s = 0; s < mdp.nS; s++) {
    const row = new Array<number>(mdp.nS).fill(0)
    for (let a = 0; a < mdp.nA; a++) {
      const w = pi[s][a]
      if (w === 0) continue
      const Psa = mdp.P[s][a]
      for (let sp = 0; sp < mdp.nS; sp++) {
        if (Psa[sp] !== 0) row[sp] += w * Psa[sp]
      }
    }
    out.push(row)
  }
  return out
}

/** 高斯消元（列主元），解 A x = b */
export function solveLinear(A: number[][], b: number[]): number[] {
  const n = b.length
  const M = A.map((row, i) => [...row, b[i]])

  for (let col = 0; col < n; col++) {
    let pivot = col
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(M[r][col]) > Math.abs(M[pivot][col])) pivot = r
    }
    if (Math.abs(M[pivot][col]) < 1e-12) continue
    ;[M[col], M[pivot]] = [M[pivot], M[col]]

    const p = M[col][col]
    for (let c = col; c <= n; c++) M[col][c] /= p

    for (let r = 0; r < n; r++) {
      if (r === col) continue
      const f = M[r][col]
      if (f === 0) continue
      for (let c = col; c <= n; c++) M[r][c] -= f * M[col][c]
    }
  }
  return M.map((row) => row[n])
}

/** 解析解：v_π = (I − γP_π)^{-1} r_π */
export function policyEvaluationDirect(mdp: MDP, pi: Policy, gamma: number): number[] {
  const n = mdp.nS
  const P = PPi(mdp, pi)
  const r = rPi(mdp, pi)
  const A = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0) - gamma * P[i][j]),
  )
  return solveLinear(A, r)
}

/** 迭代解：v_{k+1} = r_π + γ P_π v_k。返回完整轨迹，供时间轴回看 */
export function policyEvaluationTrace(
  mdp: MDP,
  pi: Policy,
  gamma: number,
  steps = 60,
  v0?: number[],
): number[][] {
  const P = PPi(mdp, pi)
  const r = rPi(mdp, pi)
  let v = v0 ? [...v0] : new Array<number>(mdp.nS).fill(0)
  const trace: number[][] = [v]
  for (let k = 0; k < steps; k++) {
    const next = new Array<number>(mdp.nS).fill(0)
    for (let s = 0; s < mdp.nS; s++) {
      let acc = 0
      for (let sp = 0; sp < mdp.nS; sp++) {
        if (P[s][sp] !== 0) acc += P[s][sp] * v[sp]
      }
      next[s] = r[s] + gamma * acc
    }
    v = next
    trace.push(v)
  }
  return trace
}

export interface VIStep {
  v: number[]
  q: number[][]
  policy: Policy
  /** 每个状态并列最优的动作 */
  ties: number[][]
  /** ‖v_k − v_{k−1}‖_∞ */
  delta: number
}

/** 值迭代 —— 同时也是求解贝尔曼最优公式的算法 */
export function valueIterationTrace(mdp: MDP, gamma: number, steps = 80, v0?: number[]): VIStep[] {
  let v = v0 ? [...v0] : new Array<number>(mdp.nS).fill(0)
  const out: VIStep[] = []
  let prev = v
  for (let k = 0; k <= steps; k++) {
    const q = qFromV(mdp, gamma, v)
    const { policy, ties } = greedyFromQ(q)
    out.push({ v, q, policy, ties, delta: k === 0 ? Infinity : normInf(v, prev) })
    prev = v
    v = q.map((row) => Math.max(...row))
  }
  return out
}

/**
 * 只要最终结果的值迭代。
 * 和 valueIterationTrace 的区别是它不保留每一步的快照 ——
 * 需要扫一大批 γ 时（比如画相变曲线）必须用这个，否则内存会白白涨几十兆。
 */
export function valueIterationSolve(
  mdp: MDP,
  gamma: number,
  tol = 1e-10,
  maxIter = 4000,
): { v: number[]; q: number[][]; policy: Policy; ties: number[][]; iters: number } {
  let v = new Array<number>(mdp.nS).fill(0)
  let iters = maxIter
  for (let k = 0; k < maxIter; k++) {
    const next = qFromV(mdp, gamma, v).map((row) => Math.max(...row))
    const d = normInf(next, v)
    v = next
    if (d < tol) {
      iters = k + 1
      break
    }
  }
  const q = qFromV(mdp, gamma, v)
  const { policy, ties } = greedyFromQ(q)
  return { v, q, policy, ties, iters }
}

export interface PIStep {
  /** 策略评估用了几步截断（Infinity 表示精确求解） */
  evalSteps: number
  vEval: number[]
  policy: Policy
  ties: number[][]
}

/**
 * 截断策略迭代。
 * evalSteps = 1 时退化为值迭代，evalSteps = ∞ 时是标准策略迭代 ——
 * 这两个「不同的算法」其实是同一条谱系上的两端。
 */
export function truncatedPolicyIterationTrace(
  mdp: MDP,
  gamma: number,
  evalSteps: number,
  rounds = 40,
  pi0?: Policy,
): PIStep[] {
  let pi: Policy =
    pi0 ?? Array.from({ length: mdp.nS }, () => [1, 0, 0, 0, 0].slice(0, mdp.nA) as number[])
  let v = new Array<number>(mdp.nS).fill(0)
  const out: PIStep[] = []

  for (let k = 0; k < rounds; k++) {
    v = Number.isFinite(evalSteps)
      ? policyEvaluationTrace(mdp, pi, gamma, evalSteps, v).at(-1)!
      : policyEvaluationDirect(mdp, pi, gamma)

    const q = qFromV(mdp, gamma, v)
    const { policy, ties } = greedyFromQ(q)
    out.push({ evalSteps, vEval: v, policy, ties })
    pi = policy
  }
  return out
}

/**
 * 在「所有 ε-贪心策略」这个受限集合里找最优的那个。
 *
 * 第 5 章需要它来回答一个尖锐的问题：ε 不为 0 时，
 * 学到的策略还是真正的最优策略吗？（答案：ε 一大就不是了。）
 */
export function epsilonGreedyOptimal(
  mdp: MDP,
  gamma: number,
  eps: number,
  rounds = 200,
): { v: number[]; policy: Policy; greedyActions: number[] } {
  let pi: Policy = Array.from({ length: mdp.nS }, () =>
    new Array<number>(mdp.nA).fill(1 / mdp.nA),
  )
  let v = new Array<number>(mdp.nS).fill(0)
  let greedy = new Array<number>(mdp.nS).fill(0)

  for (let k = 0; k < rounds; k++) {
    v = policyEvaluationDirect(mdp, pi, gamma)
    const q = qFromV(mdp, gamma, v)
    const next: Policy = []
    const g: number[] = []
    for (let s = 0; s < mdp.nS; s++) {
      const best = q[s].indexOf(Math.max(...q[s]))
      g.push(best)
      const row = new Array<number>(mdp.nA).fill(eps / mdp.nA)
      row[best] += 1 - eps
      next.push(row)
    }
    const same = g.every((a, i) => a === greedy[i])
    greedy = g
    pi = next
    if (same && k > 0) break
  }
  return { v, policy: pi, greedyActions: greedy }
}

/**
 * 压缩映射的数值见证：从两个任意初值出发，
 * 两条轨迹之间的距离必须以不超过 γ^k 的速度收缩。
 */
export function contractionWitness(
  mdp: MDP,
  pi: Policy,
  gamma: number,
  v0a: number[],
  v0b: number[],
  steps = 40,
): { distances: number[]; bound: number[] } {
  const A = policyEvaluationTrace(mdp, pi, gamma, steps, v0a)
  const B = policyEvaluationTrace(mdp, pi, gamma, steps, v0b)
  const d0 = normInf(v0a, v0b)
  return {
    distances: A.map((va, k) => normInf(va, B[k])),
    bound: A.map((_, k) => d0 * Math.pow(gamma, k)),
  }
}

export interface ActionBreakdown {
  a: number
  /** π(a|s) */
  prob: number
  /** r(s,a) */
  reward: number
  /** 有非零转移概率的后继状态 */
  successors: { sp: number; p: number; v: number }[]
  /** Σ_{s'} p(s'|s,a) v(s') */
  expectedNext: number
  /** q(s,a) = r(s,a) + γ·expectedNext */
  q: number
  /** π(a|s)·q(s,a)，即这个动作对 v(s) 的贡献 */
  contribution: number
}

/** 把 v(s) 的贝尔曼展开拆成可以逐项展示的结构 —— 这是「活的公式」的数据源 */
export function bellmanBreakdown(
  mdp: MDP,
  pi: Policy,
  gamma: number,
  v: number[],
  s: number,
): { actions: ActionBreakdown[]; total: number } {
  const actions: ActionBreakdown[] = []
  let total = 0
  for (let a = 0; a < mdp.nA; a++) {
    const Psa = mdp.P[s][a]
    const successors: { sp: number; p: number; v: number }[] = []
    let expectedNext = 0
    for (let sp = 0; sp < mdp.nS; sp++) {
      if (Psa[sp] !== 0) {
        successors.push({ sp, p: Psa[sp], v: v[sp] })
        expectedNext += Psa[sp] * v[sp]
      }
    }
    const q = mdp.R[s][a] + gamma * expectedNext
    const prob = pi[s][a]
    const contribution = prob * q
    total += contribution
    actions.push({ a, prob, reward: mdp.R[s][a], successors, expectedNext, q, contribution })
  }
  return { actions, total }
}
