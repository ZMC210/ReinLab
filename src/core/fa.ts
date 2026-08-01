import { colOf, rowOf, type MDP } from './mdp'
import type { Env } from './env'
import type { Policy } from './policy'
import { mulberry32 } from './sample'

/**
 * 第 8 章：值函数近似。
 *
 * 表格法把 v 当成一个 |S| 维向量；函数近似把它当成一条曲线，
 * 只存曲线的几个参数。省内存只是副产品 ——
 * 真正的收获是「没见过的状态也能给出估计」。
 */

export type FeatureKind = 'poly1' | 'poly2' | 'poly3' | 'fourier' | 'tabular'

export const FEATURE_LABEL: Record<FeatureKind, string> = {
  poly1: '一次多项式（平面）',
  poly2: '二次多项式（曲面）',
  poly3: '三次多项式',
  fourier: '傅里叶基（低频）',
  tabular: '表格（每个状态一个参数）',
}

/** 把状态映射成特征向量 φ(s)。归一化到 [-1,1] 是为了让不同阶的项量级接近。 */
export function featureMap(mdp: MDP, kind: FeatureKind): (s: number) => number[] {
  const { grid } = mdp
  const norm = (s: number) => {
    const x = (colOf(grid, s) / Math.max(1, grid.cols - 1)) * 2 - 1
    const y = (rowOf(grid, s) / Math.max(1, grid.rows - 1)) * 2 - 1
    return [x, y] as const
  }

  switch (kind) {
    case 'tabular':
      return (s) => {
        const v = new Array<number>(mdp.nS).fill(0)
        v[s] = 1
        return v
      }
    case 'poly1':
      return (s) => {
        const [x, y] = norm(s)
        return [1, x, y]
      }
    case 'poly2':
      return (s) => {
        const [x, y] = norm(s)
        return [1, x, y, x * x, y * y, x * y]
      }
    case 'poly3':
      return (s) => {
        const [x, y] = norm(s)
        return [1, x, y, x * x, y * y, x * y, x ** 3, y ** 3, x * x * y, x * y * y]
      }
    case 'fourier':
      return (s) => {
        const [x, y] = norm(s)
        const out = [1]
        for (let i = 0; i <= 2; i++) {
          for (let j = 0; j <= 2; j++) {
            if (i === 0 && j === 0) continue
            out.push(Math.cos(Math.PI * (i * (x + 1) * 0.5 + j * (y + 1) * 0.5)))
          }
        }
        return out
      }
  }
}

export const featureDim = (mdp: MDP, kind: FeatureKind) => featureMap(mdp, kind)(0).length

/** 最小二乘拟合真实的 v_π —— 「近似能力的上限」，与 TD 学出来的做对照 */
export function leastSquaresFit(mdp: MDP, kind: FeatureKind, vTrue: number[]): number[] {
  const phi = featureMap(mdp, kind)
  const d = featureDim(mdp, kind)
  const A = Array.from({ length: d }, () => new Array<number>(d).fill(0))
  const b = new Array<number>(d).fill(0)

  for (let s = 0; s < mdp.nS; s++) {
    const f = phi(s)
    for (let i = 0; i < d; i++) {
      b[i] += f[i] * vTrue[s]
      for (let j = 0; j < d; j++) A[i][j] += f[i] * f[j]
    }
  }
  for (let i = 0; i < d; i++) A[i][i] += 1e-8 // 正则化，避免奇异
  return solve(A, b)
}

function solve(A: number[][], b: number[]): number[] {
  const n = b.length
  const M = A.map((row, i) => [...row, b[i]])
  for (let c = 0; c < n; c++) {
    let p = c
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r
    if (Math.abs(M[p][c]) < 1e-14) continue
    ;[M[c], M[p]] = [M[p], M[c]]
    const d = M[c][c]
    for (let k = c; k <= n; k++) M[c][k] /= d
    for (let r = 0; r < n; r++) {
      if (r === c) continue
      const f = M[r][c]
      if (!f) continue
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]
    }
  }
  return M.map((row) => row[n])
}

export const evalW = (phi: (s: number) => number[], w: number[], s: number) =>
  phi(s).reduce((acc, f, i) => acc + f * w[i], 0)

export function valuesFromW(mdp: MDP, kind: FeatureKind, w: number[]): number[] {
  const phi = featureMap(mdp, kind)
  return Array.from({ length: mdp.nS }, (_, s) => evalW(phi, w, s))
}

/**
 * 半梯度 TD(0) 做价值近似。
 *
 * 「半」在哪：TD 目标 r + γ·v̂(s′,w) 里也含 w，但求梯度时我们假装它是常数。
 * 这一步不严谨，却是全部麻烦（和全部实用性）的来源。
 */
export function semiGradientTD(
  env: Env,
  mdp: MDP,
  pi: Policy,
  kind: FeatureKind,
  o: { gamma: number; alpha: number; episodes: number; seed?: number },
  vTrue?: number[],
): { w: number[]; err: number[]; xs: number[] } {
  const phi = featureMap(mdp, kind)
  const d = featureDim(mdp, kind)
  const w = new Array<number>(d).fill(0)
  const rnd = mulberry32(o.seed ?? 13)
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
    let s = Math.floor(rnd() * env.nS) % env.nS
    for (let t = 0; t < 40; t++) {
      const a = pick(pi[s])
      const { sp, r, done } = env.step(s, a, rnd)
      const f = phi(s)
      const delta = r + (done ? 0 : o.gamma * evalW(phi, w, sp)) - evalW(phi, w, s)
      for (let i = 0; i < d; i++) w[i] += o.alpha * delta * f[i]
      if (done) break
      s = sp
    }
    if (vTrue) {
      xs.push(ep)
      const v = valuesFromW(mdp, kind, w)
      err.push(
        Math.sqrt(v.reduce((acc, x, i) => acc + (x - vTrue[i]) ** 2, 0) / v.length),
      )
    }
  }
  return { w, err, xs }
}

/* ────────────────────────── 致命三位一体 ────────────────────────── */

/**
 * Baird 反例（7 个状态的简化版）。
 *
 * 它的唯一用途，就是证明「函数近似 + 自举 + 离策略」这三样凑齐时，
 * 参数可以指数发散 —— 哪怕真实价值函数明明就在近似空间里。
 */
export interface TriadOpts {
  /** 自举：用 v̂(s′) 当目标（关掉就是蒙特卡洛式的真回报） */
  bootstrap: boolean
  /** 离策略：行为策略和目标策略不同 */
  offPolicy: boolean
  /** 函数近似：关掉就是表格 */
  approx: boolean
  alpha: number
  gamma: number
  steps: number
  seed?: number
}

const BAIRD_N = 7

/** 上面 6 个状态的特征都带一个共享分量，这正是发散的温床 */
function bairdFeature(s: number, approx: boolean): number[] {
  if (!approx) {
    const v = new Array<number>(BAIRD_N).fill(0)
    v[s] = 1
    return v
  }
  const v = new Array<number>(8).fill(0)
  if (s < 6) {
    v[s] = 2
    v[7] = 1
  } else {
    v[6] = 1
    v[7] = 2
  }
  return v
}

export function bairdCounterexample(o: TriadOpts): { norm: number[]; w: number[] } {
  const dim = o.approx ? 8 : BAIRD_N
  const w = new Array<number>(dim).fill(1)
  if (o.approx) w[7] = 10
  const rnd = mulberry32(o.seed ?? 4)
  const norm: number[] = []

  const val = (s: number) => bairdFeature(s, o.approx).reduce((a, f, i) => a + f * w[i], 0)

  for (let k = 0; k < o.steps; k++) {
    const s = Math.floor(rnd() * 6) % 6 // 行为策略均匀访问上面 6 个状态
    // 目标策略永远选「去第 7 个状态」；行为策略以 6/7 的概率往上面走
    const goLower = o.offPolicy ? true : rnd() < 1 / 7
    const sp = goLower ? 6 : Math.floor(rnd() * 6) % 6
    const rho = o.offPolicy ? (goLower ? 7 : 0) : 1

    const target = o.bootstrap ? o.gamma * val(sp) : 0
    const delta = target - val(s)
    const f = bairdFeature(s, o.approx)
    for (let i = 0; i < dim; i++) w[i] += o.alpha * rho * delta * f[i]

    norm.push(Math.min(1e12, Math.sqrt(w.reduce((a, x) => a + x * x, 0))))
  }
  return { norm, w }
}
