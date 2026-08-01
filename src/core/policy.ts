import type { MDP } from './mdp'
import { N_ACTIONS } from './mdp'

/** π[s][a] = 在状态 s 选择动作 a 的概率 */
export type Policy = number[][]

export function uniformPolicy(nS: number, nA = N_ACTIONS): Policy {
  return Array.from({ length: nS }, () => new Array<number>(nA).fill(1 / nA))
}

export function deterministicPolicy(actions: number[], nA = N_ACTIONS): Policy {
  return actions.map((a) => {
    const row = new Array<number>(nA).fill(0)
    row[a] = 1
    return row
  })
}

export function clonePolicy(pi: Policy): Policy {
  return pi.map((row) => [...row])
}

/** 策略是否处处确定（每个状态只有一个动作概率为 1） */
export function isDeterministic(pi: Policy): boolean {
  return pi.every((row) => row.some((p) => p > 0.999))
}

/** 取出确定性策略在每个状态选的动作；随机策略下返回概率最大的那个 */
export function argmaxActions(pi: Policy): number[] {
  return pi.map((row) => row.indexOf(Math.max(...row)))
}

/** q[s][a] = r(s,a) + γ Σ_{s'} p(s'|s,a) v(s') */
export function qFromV(mdp: MDP, gamma: number, v: number[]): number[][] {
  const q: number[][] = []
  for (let s = 0; s < mdp.nS; s++) {
    const row: number[] = []
    for (let a = 0; a < mdp.nA; a++) {
      let acc = 0
      const Psa = mdp.P[s][a]
      for (let sp = 0; sp < mdp.nS; sp++) {
        if (Psa[sp] !== 0) acc += Psa[sp] * v[sp]
      }
      row.push(mdp.R[s][a] + gamma * acc)
    }
    q.push(row)
  }
  return q
}

const TIE_EPS = 1e-9

/** 对 q 贪心。ties 记录每个状态并列最优的动作集合——这是「最优策略不唯一」的证据 */
export function greedyFromQ(q: number[][]): { policy: Policy; ties: number[][] } {
  const policy: Policy = []
  const ties: number[][] = []
  for (const row of q) {
    const best = Math.max(...row)
    const tied = row.map((x, a) => (x > best - TIE_EPS ? a : -1)).filter((a) => a >= 0)
    const p = new Array<number>(row.length).fill(0)
    p[tied[0]] = 1
    policy.push(p)
    ties.push(tied)
  }
  return { policy, ties }
}

/** ε-贪心策略，第 5 章之后要用 */
export function epsilonGreedyFromQ(q: number[][], eps: number): Policy {
  return q.map((row) => {
    const nA = row.length
    const best = row.indexOf(Math.max(...row))
    const p = new Array<number>(nA).fill(eps / nA)
    p[best] += 1 - eps
    return p
  })
}
