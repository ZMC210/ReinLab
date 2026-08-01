import type { MDP } from './mdp'
import type { Policy } from './policy'

function pick(probs: number[], rnd: () => number): number {
  let u = rnd()
  for (let i = 0; i < probs.length; i++) {
    u -= probs[i]
    if (u <= 0) return i
  }
  return probs.length - 1
}

/** 一个可复现的伪随机数发生器，保证同样的参数每次画出同样的图 */
export function mulberry32(seed: number) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/**
 * 从状态 s 出发，按策略 π 采样若干条轨迹，返回每条轨迹的折扣回报。
 * 这是「最笨的打分办法」—— 第 2 章用它来制造对贝尔曼公式的渴望，
 * 第 5 章会回来给它起个正式的名字：蒙特卡洛。
 */
export function sampleReturns(
  mdp: MDP,
  pi: Policy,
  gamma: number,
  s: number,
  episodes: number,
  horizon = 200,
  seed = 42,
): number[] {
  const rnd = mulberry32(seed)
  const out: number[] = []
  for (let e = 0; e < episodes; e++) {
    let cur = s
    let g = 0
    let disc = 1
    for (let t = 0; t < horizon; t++) {
      const a = pick(pi[cur], rnd)
      g += disc * mdp.R[cur][a]
      disc *= gamma
      cur = pick(mdp.P[cur][a], rnd)
      if (disc < 1e-6) break
    }
    out.push(g)
  }
  return out
}

/**
 * 从 (s, a) 出发采样：第一步强制执行 a，之后遵循 π。
 * 这正是蒙特卡洛估计 q_π(s,a) 的定义式，也是 MC Basic 的取数方式。
 */
export function sampleActionReturns(
  mdp: MDP,
  pi: Policy,
  gamma: number,
  s: number,
  a0: number,
  episodes: number,
  horizon = 100,
  seed = 42,
): number[] {
  const rnd = mulberry32(seed)
  const out: number[] = []
  for (let e = 0; e < episodes; e++) {
    let cur = s
    let a = a0
    let g = 0
    let disc = 1
    for (let t = 0; t < horizon; t++) {
      g += disc * mdp.R[cur][a]
      disc *= gamma
      cur = pick(mdp.P[cur][a], rnd)
      a = pick(pi[cur], rnd)
      if (disc < 1e-6) break
    }
    out.push(g)
  }
  return out
}

export function runningMean(xs: number[]): number[] {
  let acc = 0
  return xs.map((x, i) => {
    acc += x
    return acc / (i + 1)
  })
}
