import type { Env } from './env'
import { mulberry32 } from './sample'
import type { Policy } from './policy'

/**
 * 第 9、10 章：策略梯度与 Actor-Critic。
 *
 * 前八章都在绕道：先估价值，再从价值里读出策略。
 * 这里换一条路 —— 把策略本身参数化，对目标函数直接求梯度。
 */

/** 表格化 softmax 策略：θ[s][a]，π(a|s) ∝ exp(θ[s][a]) */
export function softmax(theta: number[]): number[] {
  const m = Math.max(...theta)
  const e = theta.map((x) => Math.exp(x - m))
  const z = e.reduce((a, b) => a + b, 0)
  return e.map((x) => x / z)
}

export function policyFromTheta(theta: number[][]): Policy {
  return theta.map(softmax)
}

function pick(probs: number[], rnd: () => number): number {
  let u = rnd()
  for (let i = 0; i < probs.length; i++) {
    u -= probs[i]
    if (u <= 0) return i
  }
  return probs.length - 1
}

export interface PGOpts {
  gamma: number
  /** 策略的学习率 */
  alphaTheta: number
  /** 评论家的学习率（Actor-Critic 才用） */
  alphaW?: number
  episodes: number
  seed?: number
  probes?: number
}

export interface PGResult {
  theta: number[][]
  policy: Policy
  /** 每回合的折扣回报 */
  episodeReturn: number[]
  /** 每回合更新量的模长，用来看方差 */
  gradNorm: number[]
  snaps: { ep: number; policy: Policy; v?: number[] }[]
  xs: number[]
  v?: number[]
}

export type PGVariant = 'reinforce' | 'reinforce-baseline' | 'qac' | 'a2c'

export const PG_LABEL: Record<PGVariant, string> = {
  reinforce: 'REINFORCE（用整条轨迹的真实回报当 q）',
  'reinforce-baseline': 'REINFORCE + 基线（减去 v(s)）',
  qac: 'QAC（评论家估 q，单步更新）',
  a2c: 'A2C（评论家估 v，用 TD 误差当优势）',
}

/**
 * 四个变体共用一套骨架。
 *
 * 差别只有一处：θ 的更新量前面乘的那个系数是谁。
 *   REINFORCE      : G_t                 —— 无偏，方差大
 *   带基线          : G_t − v(s)          —— 仍无偏，方差小
 *   QAC            : q̂(s,a)              —— 有偏，可在线
 *   A2C            : r + γv̂(s′) − v̂(s)   —— TD 误差就是优势的估计
 */
export function policyGradient(env: Env, variant: PGVariant, o: PGOpts): PGResult {
  const rnd = mulberry32(o.seed ?? 17)
  const theta = Array.from({ length: env.nS }, () => new Array<number>(env.nA).fill(0))
  const v = new Array<number>(env.nS).fill(0)
  const qw = Array.from({ length: env.nS }, () => new Array<number>(env.nA).fill(0))
  const alphaW = o.alphaW ?? 0.1

  const episodeReturn: number[] = []
  const gradNorm: number[] = []
  const snaps: PGResult['snaps'] = []
  const xs: number[] = []
  const probeStep = Math.max(1, Math.floor(o.episodes / (o.probes ?? 40)))

  const bumpTheta = (s: number, a: number, coef: number, disc: number) => {
    const p = softmax(theta[s])
    let mag = 0
    for (let k = 0; k < env.nA; k++) {
      // ∇_θ ln π(a|s) 对 softmax 而言就是 1{k=a} − π(k|s)
      const g = (k === a ? 1 : 0) - p[k]
      const step = o.alphaTheta * disc * coef * g
      theta[s][k] += step
      mag += step * step
    }
    return Math.sqrt(mag)
  }

  for (let ep = 0; ep < o.episodes; ep++) {
    const S: number[] = []
    const A: number[] = []
    const R: number[] = []
    let s = env.start(rnd)
    let gsum = 0

    if (variant === 'qac' || variant === 'a2c') {
      // 在线：走一步更新一步
      let disc = 1
      for (let t = 0; t < env.horizon; t++) {
        const a = pick(softmax(theta[s]), rnd)
        const { sp, r, done } = env.step(s, a, rnd)
        R.push(r)

        let coef: number
        if (variant === 'a2c') {
          const td = r + (done ? 0 : o.gamma * v[sp]) - v[s]
          v[s] += alphaW * td
          coef = td
        } else {
          const ap = pick(softmax(theta[sp]), rnd)
          const td = r + (done ? 0 : o.gamma * qw[sp][ap]) - qw[s][a]
          qw[s][a] += alphaW * td
          coef = qw[s][a]
        }
        gsum += bumpTheta(s, a, coef, disc)
        disc *= o.gamma
        if (done) break
        s = sp
      }
    } else {
      // 蒙特卡洛：先跑完，再从后往前更新
      for (let t = 0; t < env.horizon; t++) {
        const a = pick(softmax(theta[s]), rnd)
        const { sp, r, done } = env.step(s, a, rnd)
        S.push(s)
        A.push(a)
        R.push(r)
        s = sp
        if (done) break
      }
      let G = 0
      const Gs: number[] = new Array(S.length).fill(0)
      for (let t = S.length - 1; t >= 0; t--) {
        G = o.gamma * G + R[t]
        Gs[t] = G
      }
      for (let t = 0; t < S.length; t++) {
        let coef = Gs[t]
        if (variant === 'reinforce-baseline') {
          coef = Gs[t] - v[S[t]]
          v[S[t]] += alphaW * (Gs[t] - v[S[t]])
        }
        gsum += bumpTheta(S[t], A[t], coef, Math.pow(o.gamma, t))
      }
    }

    let disc = 1
    let ret = 0
    for (const r of R) {
      ret += disc * r
      disc *= o.gamma
    }
    episodeReturn.push(ret)
    gradNorm.push(gsum)

    if (ep % probeStep === 0 || ep === o.episodes - 1) {
      xs.push(ep)
      snaps.push({ ep, policy: policyFromTheta(theta), v: [...v] })
    }
  }

  return { theta, policy: policyFromTheta(theta), episodeReturn, gradNorm, snaps, xs, v }
}
