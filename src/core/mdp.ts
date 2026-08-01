/**
 * 有限 MDP 与网格世界。
 *
 * 约定与《强化学习的数学原理》（赵世钰）保持一致：
 * 动作集合为 {上, 右, 下, 左, 原地}，转移是确定性的，
 * 立即奖励由「进入了什么样的格子」决定。
 */

export type CellKind = 'normal' | 'forbidden' | 'target'

export interface RewardSpec {
  /** 走到普通格子 */
  step: number
  /** 试图走出边界（智能体停在原地） */
  boundary: number
  /** 进入禁区 */
  forbidden: number
  /** 进入目标格 */
  target: number
}

export interface GridSpec {
  rows: number
  cols: number
  /** 长度为 rows*cols，行优先 */
  cells: CellKind[]
  rewards: RewardSpec
}

export interface ActionDef {
  id: number
  name: string
  /** 教材里的记号 a_1 … a_5 */
  tex: string
  glyph: string
  dr: number
  dc: number
}

export const ACTIONS: readonly ActionDef[] = [
  { id: 0, name: '上', tex: 'a_1', glyph: '↑', dr: -1, dc: 0 },
  { id: 1, name: '右', tex: 'a_2', glyph: '→', dr: 0, dc: 1 },
  { id: 2, name: '下', tex: 'a_3', glyph: '↓', dr: 1, dc: 0 },
  { id: 3, name: '左', tex: 'a_4', glyph: '←', dr: 0, dc: -1 },
  { id: 4, name: '原地', tex: 'a_5', glyph: '↻', dr: 0, dc: 0 },
] as const

export const N_ACTIONS = ACTIONS.length

export interface MDP {
  nS: number
  nA: number
  /** P[s][a][s'] —— 状态转移概率 */
  P: number[][][]
  /** R[s][a] —— 立即奖励的期望值 r(s,a) = Σ_r p(r|s,a)·r */
  R: number[][]
  /** rDist[s][a] —— 奖励的分布，用于在公式里展开 Σ_r p(r|s,a)·r */
  rDist: { r: number; p: number }[][][]
  grid: GridSpec
}

export const idx = (grid: GridSpec, row: number, col: number) => row * grid.cols + col
export const rowOf = (grid: GridSpec, s: number) => Math.floor(s / grid.cols)
export const colOf = (grid: GridSpec, s: number) => s % grid.cols

/** 在状态 s 采取动作 a 之后的确定性结果 */
export function stepResult(grid: GridSpec, s: number, a: number): { next: number; reward: number } {
  const act = ACTIONS[a]
  const r = rowOf(grid, s)
  const c = colOf(grid, s)
  const nr = r + act.dr
  const nc = c + act.dc

  const outOfBounds = nr < 0 || nr >= grid.rows || nc < 0 || nc >= grid.cols
  if (outOfBounds) {
    // 撞墙：留在原地，吃一个边界惩罚
    return { next: s, reward: grid.rewards.boundary }
  }

  const next = idx(grid, nr, nc)
  switch (grid.cells[next]) {
    case 'target':
      return { next, reward: grid.rewards.target }
    case 'forbidden':
      return { next, reward: grid.rewards.forbidden }
    default:
      return { next, reward: grid.rewards.step }
  }
}

export function buildGridMDP(grid: GridSpec): MDP {
  const nS = grid.rows * grid.cols
  const nA = N_ACTIONS

  const P: number[][][] = []
  const R: number[][] = []
  const rDist: { r: number; p: number }[][][] = []

  for (let s = 0; s < nS; s++) {
    const Ps: number[][] = []
    const Rs: number[] = []
    const Ds: { r: number; p: number }[][] = []
    for (let a = 0; a < nA; a++) {
      const row = new Array<number>(nS).fill(0)
      const { next, reward } = stepResult(grid, s, a)
      row[next] = 1
      Ps.push(row)
      Rs.push(reward)
      Ds.push([{ r: reward, p: 1 }])
    }
    P.push(Ps)
    R.push(Rs)
    rDist.push(Ds)
  }

  return { nS, nA, P, R, rDist, grid }
}

/** 教材第 1~3 章反复使用的 5×5 经典网格 */
export function classicGrid(): GridSpec {
  const rows = 5
  const cols = 5
  const cells: CellKind[] = new Array(rows * cols).fill('normal')
  const forbidden: [number, number][] = [
    [1, 1],
    [1, 2],
    [2, 2],
    [3, 1],
    [3, 3],
    [4, 1],
  ]
  for (const [r, c] of forbidden) cells[r * cols + c] = 'forbidden'
  cells[3 * cols + 2] = 'target'
  return {
    rows,
    cols,
    cells,
    rewards: { step: 0, boundary: -1, forbidden: -1, target: 1 },
  }
}

/**
 * 蛇形走廊。
 *
 * 第 4 章比较「外层轮数 vs 内层评估步数」时需要它：
 * 5×5 的经典网格太通透，价值信息几步就传遍全场，
 * 于是内层评估准不准几乎不影响外层轮数，曲线是平的。
 * 走廊把唯一通路拉长到几十格，信息必须一跳一跳爬过去，
 * 截断带来的差别才看得见。
 */
export function corridorGrid(rows = 9, cols = 9): GridSpec {
  const cells: CellKind[] = new Array(rows * cols).fill('forbidden')
  // 偶数行整行打通
  for (let r = 0; r < rows; r += 2) {
    for (let c = 0; c < cols; c++) cells[r * cols + c] = 'normal'
  }
  // 奇数行只在一端留一个口，左右交替，把整条路折成蛇形
  for (let r = 1; r < rows; r += 2) {
    cells[r * cols + ((r - 1) % 4 === 0 ? cols - 1 : 0)] = 'normal'
  }
  cells[rows * cols - 1] = 'target'
  return {
    rows,
    cols,
    cells,
    rewards: { step: -0.1, boundary: -1, forbidden: -10, target: 1 },
  }
}

/** 完全对称的 3×3，目标在正中央 —— 四个角上必然出现并列最优动作 */
export function symmetricGrid(): GridSpec {
  const cells: CellKind[] = new Array(9).fill('normal')
  cells[4] = 'target'
  return {
    rows: 3,
    cols: 3,
    cells,
    rewards: { step: 0, boundary: -1, forbidden: -1, target: 1 },
  }
}

/** 对所有奖励做仿射变换 r → αr + β */
export function affineRewards(g: GridSpec, alpha: number, beta: number): GridSpec {
  const t = (x: number) => alpha * x + beta
  return {
    ...g,
    cells: [...g.cells],
    rewards: {
      step: t(g.rewards.step),
      boundary: t(g.rewards.boundary),
      forbidden: t(g.rewards.forbidden),
      target: t(g.rewards.target),
    },
  }
}

export function cloneGrid(g: GridSpec): GridSpec {
  return { ...g, cells: [...g.cells], rewards: { ...g.rewards } }
}
