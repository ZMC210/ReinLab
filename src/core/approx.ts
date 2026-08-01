import { mulberry32 } from './sample'

/**
 * 第 6 章：随机近似与随机梯度下降。
 *
 * 这一章不碰 MDP。它要回答的是一个更底层的问题：
 * 当你只能看到带噪声的样本时，凭什么相信「一点一点挪」最终能挪到正确答案上？
 */

export type StepRule = 'inv-k' | 'const' | 'inv-sqrt' | 'inv-k2'

export const STEP_LABEL: Record<StepRule, string> = {
  'inv-k': 'α = 1/k',
  const: 'α = 0.1（常数）',
  'inv-sqrt': 'α = 1/√k',
  'inv-k2': 'α = 1/k²',
}

/** Robbins-Monro 两个条件：Σα = ∞（走得到）与 Σα² < ∞（噪声压得住） */
export const STEP_CHECK: Record<StepRule, { sum: boolean; sq: boolean; verdict: string }> = {
  'inv-k': { sum: true, sq: true, verdict: '两个条件都满足 —— 收敛' },
  const: { sum: true, sq: false, verdict: 'Σα²发散：会一直被噪声推着抖，只能收敛到一个邻域' },
  'inv-sqrt': { sum: true, sq: false, verdict: 'Σα²发散：抖得比 1/k 厉害，但仍在慢慢逼近' },
  'inv-k2': { sum: false, sq: true, verdict: 'Σα收敛：步子加起来是有限的，走不到真值就停住了' },
}

export function stepSize(rule: StepRule, k: number): number {
  switch (rule) {
    case 'inv-k':
      return 1 / k
    case 'const':
      return 0.1
    case 'inv-sqrt':
      return 1 / Math.sqrt(k)
    case 'inv-k2':
      return 1 / (k * k)
  }
}

/**
 * 均值估计：w_{k+1} = w_k + α_k (x_k − w_k)。
 * 这是整本书里最重要的一行代码 —— 后面所有的 TD 更新都是它的变体。
 */
export function meanEstimation(
  rule: StepRule,
  n: number,
  o: { mean?: number; noise?: number; w0?: number; seed?: number } = {},
): { w: number[]; batch: number[]; samples: number[] } {
  const { mean = 3, noise = 1, w0 = 10, seed = 5 } = o
  const rnd = mulberry32(seed)
  const gauss = () => {
    const u = Math.max(1e-9, rnd())
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rnd())
  }

  const samples: number[] = []
  const w: number[] = [w0]
  const batch: number[] = [w0]
  let cur = w0
  let acc = 0

  for (let k = 1; k <= n; k++) {
    const x = mean + noise * gauss()
    samples.push(x)
    cur = cur + stepSize(rule, k) * (x - cur)
    w.push(cur)
    acc += x
    batch.push(acc / k)
  }
  return { w, batch, samples }
}

/**
 * Robbins-Monro：只能观测到 g(w) 的带噪值，求 g(w)=0 的根。
 * 这里取 g(w) = w³ − 5，真值是 5^(1/3) ≈ 1.7100。
 */
export const RM_ROOT = Math.cbrt(5)

export function robbinsMonro(
  rule: StepRule,
  n: number,
  o: { w0?: number; noise?: number; seed?: number } = {},
): { w: number[]; g: number[] } {
  const { w0 = 3, noise = 1, seed = 9 } = o
  const rnd = mulberry32(seed)
  const w: number[] = [w0]
  const g: number[] = []
  let cur = w0
  for (let k = 1; k <= n; k++) {
    const eta = noise * (rnd() * 2 - 1)
    const obs = cur ** 3 - 5 + eta
    g.push(obs)
    cur = cur - stepSize(rule, k) * obs
    // 数值上防跑飞：常数步长遇到三次函数会炸
    cur = Math.max(-6, Math.min(6, cur))
    w.push(cur)
  }
  return { w, g }
}

/* ────────────────────────── 梯度下降三兄弟 ────────────────────────── */

export type GDKind = 'bgd' | 'mbgd' | 'sgd'

/**
 * 最小化 J(w) = E[‖w − X‖²/2]，最优解就是 E[X]。
 * 三种方法的差别只是每一步用了多少个样本去估那个期望。
 */
export function gradientDescentDemo(
  kind: GDKind,
  o: {
    /** 样本池 */
    data: { x: number; y: number }[]
    steps: number
    alpha: number
    batch?: number
    w0?: [number, number]
    seed?: number
  },
): { path: [number, number][] } {
  const { data, steps, alpha, batch = 8, w0 = [-8, 8], seed = 3 } = o
  const rnd = mulberry32(seed)
  let w = [...w0] as [number, number]
  const path: [number, number][] = [[...w] as [number, number]]

  for (let k = 1; k <= steps; k++) {
    let gx = 0
    let gy = 0
    const m = kind === 'bgd' ? data.length : kind === 'sgd' ? 1 : batch
    for (let i = 0; i < m; i++) {
      const d = kind === 'bgd' ? data[i] : data[Math.floor(rnd() * data.length) % data.length]
      gx += w[0] - d.x
      gy += w[1] - d.y
    }
    w = [w[0] - (alpha * gx) / m, w[1] - (alpha * gy) / m]
    path.push([...w] as [number, number])
  }
  return { path }
}

export function sampleCloud(n: number, seed = 21): { x: number; y: number }[] {
  const rnd = mulberry32(seed)
  const gauss = () => {
    const u = Math.max(1e-9, rnd())
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rnd())
  }
  return Array.from({ length: n }, () => ({ x: 2 * gauss(), y: 2 * gauss() }))
}
