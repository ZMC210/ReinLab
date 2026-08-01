import type { Env } from './env'
import { mulberry32 } from './sample'
import { greedyFromQ, type Policy } from './policy'

/**
 * 无模型学习算法族。
 *
 * 这里所有算法都只通过 Env.step 与世界打交道 —— 它们看不到 P，也看不到 R。
 * 之所以把 MC / Sarsa / Q-learning / n-step 写成同一个签名，
 * 是为了让第 7 章能把它们塞进同一张对比图里：
 * 差别只在「用什么当 TD 目标」这一行。
 */

export interface LearnOpts {
  gamma: number
  alpha: number
  eps: number
  episodes: number
  seed?: number
  /** 采样多少个点画曲线 */
  probes?: number
  /** ε 与 α 是否随时间衰减 */
  decay?: boolean
}

export interface LearnResult {
  q: number[][]
  /** 贪心策略 */
  policy: Policy
  /** 每回合的累计奖励（未折扣），用来画学习曲线 */
  episodeReturn: number[]
  /** 探针横坐标 */
  xs: number[]
  /** 探针处的 q 表快照，用于时间轴回看 */
  snaps: { ep: number; q: number[][]; policy: Policy }[]
}

const zeros = (n: number, m: number) => Array.from({ length: n }, () => new Array<number>(m).fill(0))

function epsGreedyPick(q: number[], eps: number, rnd: () => number): number {
  if (rnd() < eps) return Math.floor(rnd() * q.length) % q.length
  let best = 0
  for (let a = 1; a < q.length; a++) if (q[a] > q[best]) best = a
  return best
}

function makeProbes(episodes: number, probes: number) {
  const step = Math.max(1, Math.floor(episodes / probes))
  return { step }
}

export type TDVariant = 'sarsa' | 'qlearning' | 'expected-sarsa'

/**
 * 时序差分控制的统一实现。
 *
 * 三个变体唯一的区别是下一行怎么写：
 *   Sarsa          : target = r + γ·q(s', a')     a' 是真的会走的那个
 *   Q-learning     : target = r + γ·max_a q(s',a) 不管实际走哪个
 *   Expected Sarsa : target = r + γ·Σ_a π(a|s')q(s',a)
 */
export function tdControl(env: Env, variant: TDVariant, o: LearnOpts): LearnResult {
  const rnd = mulberry32(o.seed ?? 7)
  const q = zeros(env.nS, env.nA)
  const episodeReturn: number[] = []
  const snaps: LearnResult['snaps'] = []
  const { step: probeStep } = makeProbes(o.episodes, o.probes ?? 40)
  const xs: number[] = []

  for (let ep = 0; ep < o.episodes; ep++) {
    const frac = ep / Math.max(1, o.episodes)
    const eps = o.decay ? Math.max(0.01, o.eps * (1 - frac)) : o.eps
    const alpha = o.decay ? Math.max(0.02, o.alpha * (1 - 0.7 * frac)) : o.alpha

    let s = env.start(rnd)
    let a = epsGreedyPick(q[s], eps, rnd)
    let total = 0

    for (let t = 0; t < env.horizon; t++) {
      const { sp, r, done } = env.step(s, a, rnd)
      total += r
      const ap = epsGreedyPick(q[sp], eps, rnd)

      let target: number
      if (done) {
        target = r
      } else if (variant === 'sarsa') {
        target = r + o.gamma * q[sp][ap]
      } else if (variant === 'qlearning') {
        target = r + o.gamma * Math.max(...q[sp])
      } else {
        const best = q[sp].indexOf(Math.max(...q[sp]))
        let exp = 0
        for (let k = 0; k < env.nA; k++) {
          const p = eps / env.nA + (k === best ? 1 - eps : 0)
          exp += p * q[sp][k]
        }
        target = r + o.gamma * exp
      }

      q[s][a] += alpha * (target - q[s][a])
      if (done) break
      s = sp
      a = ap
    }

    episodeReturn.push(total)
    if (ep % probeStep === 0 || ep === o.episodes - 1) {
      xs.push(ep)
      snaps.push({ ep, q: q.map((r) => [...r]), policy: greedyFromQ(q).policy })
    }
  }

  return { q, policy: greedyFromQ(q).policy, episodeReturn, xs, snaps }
}

/** n 步 Sarsa：n=1 是 Sarsa，n→∞ 是蒙特卡洛。整条谱系由一个滑块贯穿。 */
export function nStepSarsa(env: Env, n: number, o: LearnOpts): LearnResult {
  const rnd = mulberry32(o.seed ?? 7)
  const q = zeros(env.nS, env.nA)
  const episodeReturn: number[] = []
  const snaps: LearnResult['snaps'] = []
  const { step: probeStep } = makeProbes(o.episodes, o.probes ?? 40)
  const xs: number[] = []

  for (let ep = 0; ep < o.episodes; ep++) {
    const frac = ep / Math.max(1, o.episodes)
    const eps = o.decay ? Math.max(0.01, o.eps * (1 - frac)) : o.eps
    const alpha = o.decay ? Math.max(0.02, o.alpha * (1 - 0.7 * frac)) : o.alpha

    const S: number[] = [env.start(rnd)]
    const A: number[] = [epsGreedyPick(q[S[0]], eps, rnd)]
    const R: number[] = [0]
    let T = Infinity
    let total = 0

    for (let t = 0; t < env.horizon + n; t++) {
      if (t < T) {
        const { sp, r, done } = env.step(S[t], A[t], rnd)
        S.push(sp)
        R.push(r)
        total += r
        if (done || t + 1 >= env.horizon) {
          T = t + 1
        } else {
          A.push(epsGreedyPick(q[sp], eps, rnd))
        }
      }
      const tau = t - n + 1
      if (tau >= 0) {
        let G = 0
        for (let i = tau + 1; i <= Math.min(tau + n, T); i++) {
          G += Math.pow(o.gamma, i - tau - 1) * R[i]
        }
        if (tau + n < T) G += Math.pow(o.gamma, n) * q[S[tau + n]][A[tau + n]]
        q[S[tau]][A[tau]] += alpha * (G - q[S[tau]][A[tau]])
      }
      if (tau >= T - 1) break
    }

    episodeReturn.push(total)
    if (ep % probeStep === 0 || ep === o.episodes - 1) {
      xs.push(ep)
      snaps.push({ ep, q: q.map((r) => [...r]), policy: greedyFromQ(q).policy })
    }
  }
  return { q, policy: greedyFromQ(q).policy, episodeReturn, xs, snaps }
}

/**
 * 每次访问型蒙特卡洛控制（ε-贪心版）。
 * 与 TD 的唯一结构差别：必须等整条轨迹跑完，再从后往前算回报。
 */
export function mcControl(env: Env, o: LearnOpts & { firstVisit?: boolean }): LearnResult {
  const rnd = mulberry32(o.seed ?? 7)
  const q = zeros(env.nS, env.nA)
  const cnt = zeros(env.nS, env.nA)
  const episodeReturn: number[] = []
  const snaps: LearnResult['snaps'] = []
  const { step: probeStep } = makeProbes(o.episodes, o.probes ?? 40)
  const xs: number[] = []

  for (let ep = 0; ep < o.episodes; ep++) {
    const frac = ep / Math.max(1, o.episodes)
    const eps = o.decay ? Math.max(0.05, o.eps * (1 - frac)) : o.eps

    const S: number[] = []
    const A: number[] = []
    const R: number[] = []
    let s = env.start(rnd)
    let total = 0
    for (let t = 0; t < env.horizon; t++) {
      const a = epsGreedyPick(q[s], eps, rnd)
      const { sp, r, done } = env.step(s, a, rnd)
      S.push(s)
      A.push(a)
      R.push(r)
      total += r
      s = sp
      if (done) break
    }

    const seen = new Set<number>()
    let G = 0
    for (let t = S.length - 1; t >= 0; t--) {
      G = o.gamma * G + R[t]
      const key = S[t] * env.nA + A[t]
      if (o.firstVisit && seen.has(key)) continue
      seen.add(key)
      cnt[S[t]][A[t]] += 1
      q[S[t]][A[t]] += (G - q[S[t]][A[t]]) / cnt[S[t]][A[t]]
    }

    episodeReturn.push(total)
    if (ep % probeStep === 0 || ep === o.episodes - 1) {
      xs.push(ep)
      snaps.push({ ep, q: q.map((r) => [...r]), policy: greedyFromQ(q).policy })
    }
  }
  return { q, policy: greedyFromQ(q).policy, episodeReturn, xs, snaps }
}

/**
 * TD(0) 估计状态价值（只做预测，不做控制）。
 * 返回每一步之后的误差，用来和蒙特卡洛的方差做对比。
 */
export function tdPrediction(
  env: Env,
  pi: Policy,
  o: { gamma: number; alpha: number; episodes: number; seed?: number },
  vTrue?: number[],
): { v: number[]; err: number[]; xs: number[] } {
  const rnd = mulberry32(o.seed ?? 11)
  const v = new Array<number>(env.nS).fill(0)
  const err: number[] = []
  const xs: number[] = []

  const pick = (probs: number[]) => {
    let u = rnd()
    for (let i = 0; i < probs.length; i++) {
      u -= probs[i]
      if (u <= 0) return i
    }
    return probs.length - 1
  }

  for (let ep = 0; ep < o.episodes; ep++) {
    let s = env.start(rnd)
    for (let t = 0; t < env.horizon; t++) {
      const a = pick(pi[s])
      const { sp, r, done } = env.step(s, a, rnd)
      v[s] += o.alpha * (r + (done ? 0 : o.gamma * v[sp]) - v[s])
      if (done) break
      s = sp
    }
    if (vTrue) {
      xs.push(ep)
      err.push(Math.max(...v.map((x, i) => Math.abs(x - vTrue[i]))))
    }
  }
  return { v, err, xs }
}

/** 蒙特卡洛估计状态价值，与 tdPrediction 对照 */
export function mcPrediction(
  env: Env,
  pi: Policy,
  o: { gamma: number; episodes: number; seed?: number },
  vTrue?: number[],
): { v: number[]; err: number[]; xs: number[] } {
  const rnd = mulberry32(o.seed ?? 11)
  const v = new Array<number>(env.nS).fill(0)
  const cnt = new Array<number>(env.nS).fill(0)
  const err: number[] = []
  const xs: number[] = []

  const pick = (probs: number[]) => {
    let u = rnd()
    for (let i = 0; i < probs.length; i++) {
      u -= probs[i]
      if (u <= 0) return i
    }
    return probs.length - 1
  }

  for (let ep = 0; ep < o.episodes; ep++) {
    const S: number[] = []
    const R: number[] = []
    let s = env.start(rnd)
    for (let t = 0; t < env.horizon; t++) {
      const a = pick(pi[s])
      const { sp, r, done } = env.step(s, a, rnd)
      S.push(s)
      R.push(r)
      s = sp
      if (done) break
    }
    let G = 0
    for (let t = S.length - 1; t >= 0; t--) {
      G = o.gamma * G + R[t]
      cnt[S[t]] += 1
      v[S[t]] += (G - v[S[t]]) / cnt[S[t]]
    }
    if (vTrue) {
      xs.push(ep)
      err.push(Math.max(...v.map((x, i) => Math.abs(x - vTrue[i]))))
    }
  }
  return { v, err, xs }
}

/** 把 q 表变成从某点出发的贪心轨迹，用来画「学到的路线」 */
export function greedyPath(env: Env, q: number[][], maxLen = 60): number[] {
  const rnd = () => 0.5
  let s = env.start(rnd)
  const path = [s]
  const seen = new Set<number>([s])
  for (let t = 0; t < maxLen; t++) {
    const a = q[s].indexOf(Math.max(...q[s]))
    const { sp, done } = env.step(s, a, rnd)
    path.push(sp)
    if (done || seen.has(sp)) break
    seen.add(sp)
    s = sp
  }
  return path
}

/** 平滑曲线，让学习曲线的趋势看得见 */
export function smooth(xs: number[], w = 10): number[] {
  const out: number[] = []
  let acc = 0
  for (let i = 0; i < xs.length; i++) {
    acc += xs[i]
    if (i >= w) acc -= xs[i - w]
    out.push(acc / Math.min(i + 1, w))
  }
  return out
}
