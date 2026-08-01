import { ACTIONS, buildGridMDP, colOf, idx, rowOf, type CellKind, type GridSpec, type MDP } from './mdp'

/**
 * 采样式环境。
 *
 * 第 1~4 章的世界是「模型已知」的：我们直接读 P 和 R。
 * 从第 5 章起，算法只被允许做一件事 —— 调用 step()。
 * 这个接口存在的意义就是把这条界线画出来：
 * 凡是只依赖 Env 的算法，就是 model-free 的。
 */
export interface Env {
  nS: number
  nA: number
  /** 回合起点。给随机数是为了支持 exploring starts */
  start: (rnd: () => number) => number
  step: (s: number, a: number, rnd: () => number) => { sp: number; r: number; done: boolean }
  /** 用于渲染；没有网格的环境（如 Baird）留空 */
  grid?: GridSpec
  /** 单回合最长步数 */
  horizon: number
}

/** 把已知模型的 MDP 包成只能采样的环境 —— 算法这边看不到 P 和 R */
export function mdpToEnv(mdp: MDP, opts: { start?: number; horizon?: number } = {}): Env {
  const { start = 0, horizon = 100 } = opts
  return {
    nS: mdp.nS,
    nA: mdp.nA,
    grid: mdp.grid,
    horizon,
    start: () => start,
    step: (s, a, rnd) => {
      const probs = mdp.P[s][a]
      let u = rnd()
      let sp = mdp.nS - 1
      for (let i = 0; i < probs.length; i++) {
        u -= probs[i]
        if (u <= 0) {
          sp = i
          break
        }
      }
      return { sp, r: mdp.R[s][a], done: false }
    },
  }
}

/** exploring starts：每回合从任意 (s,a) 出发 */
export function withExploringStarts(env: Env): Env {
  return { ...env, start: (rnd) => Math.floor(rnd() * env.nS) % env.nS }
}

/* ────────────────────────── 悬崖行走 ────────────────────────── */

/**
 * Sutton & Barto 的经典反例，4×12。
 *
 * 它之所以是 Sarsa 与 Q-learning 之争的最佳舞台：
 * 最优路径贴着悬崖走，但只要还在探索，贴边走就会时不时掉下去。
 * 于是「学到的最优路径」和「实际走得最好的路径」第一次分道扬镳。
 */
export function cliffGrid(rows = 4, cols = 12): GridSpec {
  const cells: CellKind[] = new Array(rows * cols).fill('normal')
  for (let c = 1; c < cols - 1; c++) cells[(rows - 1) * cols + c] = 'forbidden'
  cells[(rows - 1) * cols + (cols - 1)] = 'target'
  return { rows, cols, cells, rewards: { step: -1, boundary: -1, forbidden: -100, target: 0 } }
}

export function cliffEnv(rows = 4, cols = 12): Env {
  const grid = cliffGrid(rows, cols)
  const startS = (rows - 1) * cols
  const goal = (rows - 1) * cols + (cols - 1)

  return {
    nS: rows * cols,
    nA: 4, // 悬崖行走里没有「原地不动」
    grid,
    horizon: 400,
    start: () => startS,
    step: (s, a) => {
      const act = ACTIONS[a]
      const r0 = rowOf(grid, s)
      const c0 = colOf(grid, s)
      const nr = Math.min(rows - 1, Math.max(0, r0 + act.dr))
      const nc = Math.min(cols - 1, Math.max(0, c0 + act.dc))
      const sp = idx(grid, nr, nc)
      if (grid.cells[sp] === 'forbidden') return { sp: startS, r: -100, done: false }
      // 踏进终点的这一步同样计 −1（Sutton & Barto 的约定），于是最优回报正好 −13
      if (sp === goal) return { sp, r: -1, done: true }
      return { sp, r: -1, done: false }
    },
  }
}

/** 悬崖世界的真实模型，用来算「理论最优」作为对照 */
export function cliffMDP(rows = 4, cols = 12): MDP {
  const grid = cliffGrid(rows, cols)
  const nS = rows * cols
  const nA = 4
  const startS = (rows - 1) * cols
  const goal = nS - 1

  const P: number[][][] = []
  const R: number[][] = []
  for (let s = 0; s < nS; s++) {
    const Ps: number[][] = []
    const Rs: number[] = []
    for (let a = 0; a < nA; a++) {
      const row = new Array<number>(nS).fill(0)
      if (s === goal) {
        row[goal] = 1
        Ps.push(row)
        Rs.push(0)
        continue
      }
      const act = ACTIONS[a]
      const nr = Math.min(rows - 1, Math.max(0, rowOf(grid, s) + act.dr))
      const nc = Math.min(cols - 1, Math.max(0, colOf(grid, s) + act.dc))
      const sp = idx(grid, nr, nc)
      if (grid.cells[sp] === 'forbidden') {
        row[startS] = 1
        Rs.push(-100)
      } else if (sp === goal) {
        row[goal] = 1
        Rs.push(-1)
      } else {
        row[sp] = 1
        Rs.push(-1)
      }
      Ps.push(row)
    }
    P.push(Ps)
    R.push(Rs)
  }
  const rDist = R.map((row) => row.map((r) => [{ r, p: 1 }]))
  return { nS, nA, P, R, rDist, grid }
}

/** 一个小一点的网格，用来做「表格法够用」的对照组 */
export function smallGridMDP(): MDP {
  const cells: CellKind[] = new Array(16).fill('normal')
  cells[5] = 'forbidden'
  cells[9] = 'forbidden'
  cells[15] = 'target'
  return buildGridMDP({
    rows: 4,
    cols: 4,
    cells,
    rewards: { step: 0, boundary: -1, forbidden: -1, target: 1 },
  })
}
