// 一次性核对正文里的数值断言。node scripts/verify.mjs
const ACT = [
  [-1, 0],
  [0, 1],
  [1, 0],
  [0, -1],
  [0, 0],
]

function build(rows, cols, cells, rw) {
  const nS = rows * cols
  const nA = 5
  const P = []
  const R = []
  for (let s = 0; s < nS; s++) {
    const Ps = []
    const Rs = []
    const r = Math.floor(s / cols)
    const c = s % cols
    for (let a = 0; a < nA; a++) {
      const row = new Array(nS).fill(0)
      const nr = r + ACT[a][0]
      const nc = c + ACT[a][1]
      if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) {
        row[s] = 1
        Ps.push(row)
        Rs.push(rw.boundary)
        continue
      }
      const sp = nr * cols + nc
      row[sp] = 1
      Ps.push(row)
      Rs.push(cells[sp] === 'target' ? rw.target : cells[sp] === 'forbidden' ? rw.forbidden : rw.step)
    }
    P.push(Ps)
    R.push(Rs)
  }
  return { nS, nA, P, R }
}

function classic() {
  const cells = new Array(25).fill('normal')
  for (const [r, c] of [
    [1, 1],
    [1, 2],
    [2, 2],
    [3, 1],
    [3, 3],
    [4, 1],
  ])
    cells[r * 5 + c] = 'forbidden'
  cells[3 * 5 + 2] = 'target'
  return cells
}

const RW = { step: 0, boundary: -1, forbidden: -1, target: 1 }

function qFromV(m, gamma, v) {
  return Array.from({ length: m.nS }, (_, s) =>
    Array.from({ length: m.nA }, (_, a) => {
      let acc = 0
      for (let sp = 0; sp < m.nS; sp++) if (m.P[s][a][sp]) acc += m.P[s][a][sp] * v[sp]
      return m.R[s][a] + gamma * acc
    }),
  )
}

function vi(m, gamma, steps = 400) {
  let v = new Array(m.nS).fill(0)
  for (let k = 0; k < steps; k++) v = qFromV(m, gamma, v).map((r) => Math.max(...r))
  return v
}

function greedy(q) {
  return q.map((row) => {
    const best = Math.max(...row)
    const tied = row.map((x, a) => (x > best - 1e-9 ? a : -1)).filter((a) => a >= 0)
    return tied
  })
}

const m = build(5, 5, classic(), RW)

// 1. γ=0.9 下目标格的价值是不是 ≈10
const v09 = vi(m, 0.9)
console.log('γ=0.9  v*(target s18) =', v09[17].toFixed(4))
console.log('γ=0.9  v* 范围 =', Math.min(...v09).toFixed(2), '~', Math.max(...v09).toFixed(2))

// 2. γ 翻转
const p09 = greedy(qFromV(m, 0.9, v09)).map((t) => t[0])
for (const g of [0.1, 0.3, 0.5, 0.7, 0.95, 0.99]) {
  const p = greedy(qFromV(m, g, vi(m, g))).map((t) => t[0])
  const flips = p.reduce((a, x, i) => a + (x === p09[i] ? 0 : 1), 0)
  console.log(`γ=${g}  与 γ=0.9 相比翻转格子数 = ${flips}`)
}

// 2b. 「大 γ 敢穿禁区抄近道 / 小 γ 绕远路」这个机理说法
const GLYPH = ['↑', '→', '↓', '←', '·']
const cells = classic()
for (const g of [0.99, 0.9, 0.7, 0.5, 0.1]) {
  const p = greedy(qFromV(m, g, vi(m, g))).map((t) => t[0])
  let into = 0
  for (let s = 0; s < 25; s++) {
    const nr = Math.floor(s / 5) + ACT[p[s]][0]
    const nc = (s % 5) + ACT[p[s]][1]
    if (nr < 0 || nr > 4 || nc < 0 || nc > 4) continue
    if (cells[nr * 5 + nc] === 'forbidden') into++
  }
  console.log(`\nγ=${g}  主动踏入禁区的格子数 = ${into}`)
  for (let r = 0; r < 5; r++) {
    console.log(
      '   ' +
        Array.from({ length: 5 }, (_, c) => {
          const s = r * 5 + c
          const mark = cells[s] === 'forbidden' ? 'x' : cells[s] === 'target' ? 'T' : ' '
          return GLYPH[p[s]] + mark
        }).join(' '),
    )
  }
}

// 3. 对称 3×3 的并列最优
const symCells = new Array(9).fill('normal')
symCells[4] = 'target'
const sm = build(3, 3, symCells, RW)
const vs = vi(sm, 0.9)
const ties = greedy(qFromV(sm, 0.9, vs))
console.log('对称 3×3 并列格子数 =', ties.filter((t) => t.length > 1).length, JSON.stringify(ties))

// 4. 仿射不变性
for (const [al, be] of [
  [1, 100],
  [3, 0],
  [0.1, -50],
  [5, 100],
]) {
  const cells = classic()
  const t = (x) => al * x + be
  const m2 = build(5, 5, cells, {
    step: t(RW.step),
    boundary: t(RW.boundary),
    forbidden: t(RW.forbidden),
    target: t(RW.target),
  })
  const p = greedy(qFromV(m2, 0.9, vi(m2, 0.9))).map((x) => x[0])
  const diff = p.reduce((a, x, i) => a + (x === p09[i] ? 0 : 1), 0)
  console.log(`α=${al} β=${be}  策略差异 = ${diff}`)
}

// 5. 「一路向右」策略在 γ=0.9 下 s18 的价值
function solve(A, b) {
  const n = b.length
  const M = A.map((r, i) => [...r, b[i]])
  for (let c = 0; c < n; c++) {
    let p = c
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r
    ;[M[c], M[p]] = [M[p], M[c]]
    const d = M[c][c]
    for (let j = c; j <= n; j++) M[c][j] /= d
    for (let r = 0; r < n; r++) {
      if (r === c) continue
      const f = M[r][c]
      for (let j = c; j <= n; j++) M[r][j] -= f * M[c][j]
    }
  }
  return M.map((r) => r[n])
}
const naive = Array.from({ length: 25 }, (_, s) => (s % 5 === 4 ? 2 : 1))
const Pn = Array.from({ length: 25 }, (_, s) => m.P[s][naive[s]])
const rn = Array.from({ length: 25 }, (_, s) => m.R[s][naive[s]])
const A = Array.from({ length: 25 }, (_, i) =>
  Array.from({ length: 25 }, (_, j) => (i === j ? 1 : 0) - 0.9 * Pn[i][j]),
)
const vn = solve(A, rn)
console.log('一路向右  v(s18) =', vn[17].toFixed(4), ' v(s7) =', vn[6].toFixed(4))
