/**
 * 第 4~10 章的数值断言核对。
 *
 * 和 verify.mjs（独立重写一遍算法、交叉验证第 1~3 章）分工不同：
 * 这个脚本直接调用 src/core 里正文用的那份实现，超参和种子也照抄章节里写死的值。
 * 它保证的是「正文里那句定性结论，在这组种子下真的成立」——
 * 这类结论最容易在调参时悄悄失效，而页面上又看不出来。
 *
 * 用法：npm run verify:core
 */
import { buildGridMDP, classicGrid, corridorGrid } from '../src/core/mdp'
import { uniformPolicy } from '../src/core/policy'
import {
  epsilonGreedyOptimal,
  normInf,
  policyEvaluationDirect,
  truncatedPolicyIterationTrace,
  valueIterationSolve,
  valueIterationTrace,
} from '../src/core/solvers'
import { cliffEnv, cliffMDP, mdpToEnv, smallGridMDP, withExploringStarts } from '../src/core/env'
import { greedyPath, mcControl, mcPrediction, nStepSarsa, tdControl, tdPrediction } from '../src/core/learn'
import { RM_ROOT, gradientDescentDemo, meanEstimation, robbinsMonro, sampleCloud } from '../src/core/approx'
import { bairdCounterexample, featureMap, leastSquaresFit, semiGradientTD } from '../src/core/fa'
import { policyGradient } from '../src/core/pg'

let failed = 0
let passed = 0

function check(name: string, ok: boolean, detail: string) {
  if (ok) {
    passed++
    console.log(`  \x1b[32m✓\x1b[0m ${name}  \x1b[2m${detail}\x1b[0m`)
  } else {
    failed++
    console.log(`  \x1b[31m✗ ${name}  ${detail}\x1b[0m`)
  }
}

const head = (t: string) => console.log(`\n\x1b[1m${t}\x1b[0m`)
const note = (t: string) => console.log(`    \x1b[2m${t}\x1b[0m`)

const mdp = buildGridMDP(classicGrid())
const GAMMA = 0.9
const star = valueIterationSolve(mdp, GAMMA)

/* ─────────────────── 第 4 章 值迭代 / 策略迭代 ─────────────────── */
head('第 4 章 · 值迭代与策略迭代')
{
  // 谱系实验用的是蛇形走廊（见 Chapter4 的注释）：经典网格上曲线是平的，做不出效果
  const cor = buildGridMDP(corridorGrid(9, 9))
  const corStar = valueIterationSolve(cor, GAMMA)
  const corTarget = JSON.stringify(corStar.policy)
  const roundsFor = (j: number) => {
    const tr = truncatedPolicyIterationTrace(cor, GAMMA, j, 120, uniformPolicy(cor.nS, cor.nA))
    for (let k = 0; k < tr.length; k++) if (JSON.stringify(tr[k].policy) === corTarget) return k + 1
    return Infinity
  }
  const SPECTRUM = [1, 2, 3, 5, 8, 13, 21, 34]
  const spec = SPECTRUM.map(roundsFor)
  const rInf = roundsFor(Infinity)
  note(`蛇形走廊的截断谱系 j=${SPECTRUM.join(',')} → ${spec.join(', ')}   （j=∞ 时 ${rInf}）`)

  check('j=1（值迭代）能达到最优策略', Number.isFinite(spec[0]), `${spec[0]} 轮`)
  check('j=∞（策略迭代）轮数最少', rInf <= spec[0], `${rInf} ≤ ${spec[0]}`)
  check('轮数随内层步数单调不增', spec.every((x, i) => i === 0 || x <= spec[i - 1]), '谱系单调')
  check(
    '谱系「先陡后平」而不是一条直线（这一幕的全部意义）',
    spec[0] - spec[2] >= 3 && spec.at(-1)! === rInf,
    `前段掉了 ${spec[0] - spec[2]} 轮，末段贴住地板 ${rInf}`,
  )
  check(
    '内层再精确也不会低于策略迭代（正文原话）',
    Math.min(...spec) >= rInf,
    `谱系最小 ${Math.min(...spec)} ≥ ${rInf}`,
  )

  // 对照：经典网格上这个实验确实做不出效果，正文的换世界理由成立
  const flat = [1, 2, 3, 5, 8].map((j) => {
    const t = JSON.stringify(star.policy)
    const tr = truncatedPolicyIterationTrace(mdp, GAMMA, j, 120, uniformPolicy(mdp.nS, mdp.nA))
    for (let k = 0; k < tr.length; k++) if (JSON.stringify(tr[k].policy) === t) return k + 1
    return Infinity
  })
  note(`（对照）5×5 经典网格上的谱系 = ${flat.join(', ')} —— 几乎是平的，所以换了走廊`)
  check('经典网格上谱系确实平坦，换世界的理由成立', flat[0] - flat.at(-1)! <= 2, `落差只有 ${flat[0] - flat.at(-1)!} 轮`)

  const piV = truncatedPolicyIterationTrace(mdp, GAMMA, Infinity, 60).at(-1)!.vEval
  check('两条路殊途同归到同一个 v*', normInf(piV, star.v) < 1e-8, `‖差‖∞ = ${normInf(piV, star.v).toExponential(2)}`)

  const e = valueIterationTrace(mdp, GAMMA, 60).map((s) => normInf(s.v, star.v))
  check(
    '值迭代误差被 γ^k 压住',
    e.every((x, k) => x <= e[0] * GAMMA ** k + 1e-9),
    `e₀=${e[0].toFixed(3)} → e₂₀=${e[20].toExponential(2)}（上界 ${(e[0] * GAMMA ** 20).toExponential(2)}）`,
  )
}

/* ─────────────────── 第 5 章 蒙特卡洛 ─────────────────── */
head('第 5 章 · 蒙特卡洛')
{
  const optActions = star.policy.map((row) => row.indexOf(Math.max(...row)))
  const rows = [0, 0.1, 0.2, 0.5, 1].map((eps) => {
    const r = epsilonGreedyOptimal(mdp, GAMMA, eps)
    return {
      eps,
      wrong: r.greedyActions.filter((a, i) => a !== optActions[i]).length,
      v0: r.v[0],
    }
  })
  for (const r of rows) note(`ε=${r.eps}  与最优不一致的格子 ${r.wrong}/25  v(s1)=${r.v0.toFixed(3)}`)

  check('ε=0 时受限最优就是真最优', rows[0].wrong === 0, '0 个格子不一致')
  check('ε 足够大时受限最优不再是真最优（正文断言）', rows.at(-1)!.wrong > 0, `ε=1 时 ${rows.at(-1)!.wrong} 个格子不一致`)
  check(
    'ε 越大 v(s1) 越低（探索是有代价的）',
    rows.every((r, i) => i === 0 || r.v0 <= rows[i - 1].v0 + 1e-9),
    rows.map((r) => r.v0.toFixed(2)).join(' ≥ '),
  )

  // MC 控制：和章节里完全一致的参数（exploring starts + ε=0.2 + 600 回合）
  const env = mdpToEnv(mdp, { start: 0, horizon: 30 })
  const mc = mcControl(withExploringStarts(env), {
    gamma: GAMMA,
    alpha: 0,
    eps: 0.2,
    episodes: 600,
    seed: 3,
    probes: 40,
    decay: false,
  })
  const hit = mc.policy.reduce(
    (a, row, s) => a + (star.ties[s].includes(row.indexOf(Math.max(...row))) ? 1 : 0),
    0,
  )
  note(`exploring starts + 600 回合后，${hit}/25 个格子命中最优动作`)
  check('MC 控制确实学出了接近最优的策略', hit >= 20, `${hit}/25`)

  // 固定起点 vs exploring starts —— 正文说前者学不全
  const mcFixed = mcControl(env, { gamma: GAMMA, alpha: 0, eps: 0.2, episodes: 600, seed: 3, decay: false })
  const hitFixed = mcFixed.policy.reduce(
    (a, row, s) => a + (star.ties[s].includes(row.indexOf(Math.max(...row))) ? 1 : 0),
    0,
  )
  note(`固定起点 600 回合后，${hitFixed}/25 个格子命中最优动作`)
  check('固定起点覆盖不如 exploring starts（正文断言）', hitFixed <= hit, `${hitFixed} ≤ ${hit}`)
}

/* ─────────────────── 第 6 章 随机近似与 SGD ─────────────────── */
head('第 6 章 · 随机近似与随机梯度下降')
{
  const N = 3000
  const rk = meanEstimation('inv-k', N, { mean: 3, noise: 1, seed: 5 })
  const ck = meanEstimation('const', N, { mean: 3, noise: 1, seed: 5 })
  const k2 = meanEstimation('inv-k2', N, { mean: 3, noise: 1, seed: 5 })

  check('α=1/k 收敛到真值', Math.abs(rk.w.at(-1)! - 3) < 0.1, `w_N = ${rk.w.at(-1)!.toFixed(4)}`)
  check(
    'α=1/k 的增量式结果 = 批量平均（RM 就是在求平均）',
    Math.abs(rk.w.at(-1)! - rk.batch.at(-1)!) < 1e-9,
    `${rk.w.at(-1)!.toFixed(8)} vs ${rk.batch.at(-1)!.toFixed(8)}`,
  )

  const tail = ck.w.slice(-300)
  const spread = Math.max(...tail) - Math.min(...tail)
  check(
    '常数步长不收敛，只在真值邻域抖动（Σα² 发散）',
    spread > 0.05,
    `末段 300 步振幅 = ${spread.toFixed(4)}`,
  )
  check(
    'α=1/k² 走不到真值就停住（Σα 收敛）',
    Math.abs(k2.w.at(-1)! - 3) > Math.abs(rk.w.at(-1)! - 3),
    `残差 ${Math.abs(k2.w.at(-1)! - 3).toFixed(3)} > ${Math.abs(rk.w.at(-1)! - 3).toFixed(3)}`,
  )

  const rm = robbinsMonro('inv-k', 4000, { w0: 3, noise: 1, seed: 9 })
  note(`Robbins-Monro 求 w³=5 的根：真值 ${RM_ROOT.toFixed(4)}，迭代末值 ${rm.w.at(-1)!.toFixed(4)}`)
  check('RM 在只看得到带噪观测时仍能求根', Math.abs(rm.w.at(-1)! - RM_ROOT) < 0.15, `|误差| = ${Math.abs(rm.w.at(-1)! - RM_ROOT).toFixed(4)}`)

  const data = sampleCloud(200, 21)
  const mean: [number, number] = [
    data.reduce((a, d) => a + d.x, 0) / data.length,
    data.reduce((a, d) => a + d.y, 0) / data.length,
  ]
  const ends = (['bgd', 'mbgd', 'sgd'] as const).map((kind) => ({
    kind,
    end: gradientDescentDemo(kind, { data, steps: 300, alpha: 0.05, batch: 8, seed: 3 }).path.at(-1)!,
  }))
  for (const e of ends) note(`${e.kind.padEnd(4)} 终点 [${e.end.map((x) => x.toFixed(3)).join(', ')}]，样本均值 [${mean.map((x) => x.toFixed(3)).join(', ')}]`)
  check(
    '三种梯度下降都收敛到样本均值附近（正文断言：终点一样，路径不同）',
    ends.every((e) => Math.hypot(e.end[0] - mean[0], e.end[1] - mean[1]) < 0.8),
    '与样本均值距离均 < 0.8',
  )
}

/* ─────────────────── 第 7 章 时序差分 ─────────────────── */
head('第 7 章 · 时序差分')
{
  const env = mdpToEnv(mdp, { start: 0, horizon: 40 })
  const pi = uniformPolicy(mdp.nS, mdp.nA)
  const vTrue = policyEvaluationDirect(mdp, pi, GAMMA)
  const td = tdPrediction(env, pi, { gamma: GAMMA, alpha: 0.1, episodes: 400, seed: 5 }, vTrue)
  const mc = mcPrediction(env, pi, { gamma: GAMMA, episodes: 400, seed: 5 }, vTrue)
  note(`400 回合后的最大误差： TD ${td.err.at(-1)!.toFixed(4)}   MC ${mc.err.at(-1)!.toFixed(4)}`)
  check('TD 预测在收敛', td.err.at(-1)! < td.err[0], `${td.err[0].toFixed(3)} → ${td.err.at(-1)!.toFixed(3)}`)
  check('MC 预测在收敛', mc.err.at(-1)! < mc.err[0], `${mc.err[0].toFixed(3)} → ${mc.err.at(-1)!.toFixed(3)}`)

  const n1 = nStepSarsa(env, 1, { gamma: GAMMA, alpha: 0.1, eps: 0.2, episodes: 400, seed: 11, decay: true })
  const n8 = nStepSarsa(env, 8, { gamma: GAMMA, alpha: 0.1, eps: 0.2, episodes: 400, seed: 11, decay: true })
  check('n 步 Sarsa 数值稳定（n=1 与 n=8）', n1.q.flat().every(Number.isFinite) && n8.q.flat().every(Number.isFinite), 'q 全部有限')

  /* 悬崖对决 —— 正文最强的一句断言 */
  const cEnv = cliffEnv()
  const cStar = valueIterationSolve(cliffMDP(), 1)
  check('悬崖最优回报 = −13', Math.abs(cStar.v[36] + 13) < 1e-6, `v*(起点) = ${cStar.v[36].toFixed(4)}`)

  const opts = { gamma: 1, alpha: 0.5, eps: 0.1, episodes: 600, seed: 2, probes: 30 }
  const sarsa = tdControl(cEnv, 'sarsa', opts)
  const ql = tdControl(cEnv, 'qlearning', opts)

  const rowOf = (s: number) => Math.floor(s / 12)
  const stat = (q: number[][]) => {
    const p = greedyPath(cEnv, q)
    return { len: p.length, minRow: Math.min(...p.map(rowOf)) }
  }
  const ps = stat(sarsa.q)
  const pq = stat(ql.q)
  note(`Sarsa 贪心路径 ${ps.len} 步，最远绕到第 ${ps.minRow} 行`)
  note(`Q-learning 贪心路径 ${pq.len} 步，最远绕到第 ${pq.minRow} 行（行号越大越贴悬崖）`)
  check('Q-learning 学到的路径更贴悬崖（正文断言）', pq.minRow >= ps.minRow, `${pq.minRow} ≥ ${ps.minRow}`)

  const t100 = (a: number[]) => a.slice(-100).reduce((x, y) => x + y, 0) / 100
  const rs = t100(sarsa.episodeReturn)
  const rq = t100(ql.episodeReturn)
  note(`末 100 回合的实际在线回报： Sarsa ${rs.toFixed(1)}   Q-learning ${rq.toFixed(1)}`)
  check('ε-贪心在线时 Sarsa 摔得更少（正文断言）', rs > rq, `${rs.toFixed(1)} > ${rq.toFixed(1)}`)
}

/* ─────────────────── 第 8 章 值函数近似 ─────────────────── */
head('第 8 章 · 值函数近似')
{
  const small = smallGridMDP()
  const pi = uniformPolicy(small.nS, small.nA)
  const vTrue = policyEvaluationDirect(small, pi, GAMMA)

  const fits = (['poly1', 'poly2', 'poly3', 'tabular'] as const).map((kind) => {
    const w = leastSquaresFit(small, kind, vTrue)
    const phi = featureMap(small, kind)
    const vHat = Array.from({ length: small.nS }, (_, s) => phi(s).reduce((a, x, i) => a + x * w[i], 0))
    return { kind, dim: phi(0).length, err: normInf(vHat, vTrue) }
  })
  for (const f of fits) note(`${f.kind.padEnd(8)} 参数 ${String(f.dim).padStart(2)} 个，最小二乘残差 ${f.err.toFixed(5)}`)
  check(
    '参数越多，拟合上限越好（正文断言）',
    fits.every((f, i) => i === 0 || f.err <= fits[i - 1].err + 1e-9),
    fits.map((f) => f.err.toFixed(4)).join(' ≥ '),
  )
  check('表格法是函数近似的特例（残差为 0）', fits.at(-1)!.err < 1e-6, `残差 = ${fits.at(-1)!.err.toExponential(2)}`)

  const env = mdpToEnv(small, { start: 0, horizon: 40 })
  const sg = semiGradientTD(env, small, pi, 'poly2', { gamma: GAMMA, alpha: 0.02, episodes: 1500, seed: 6 }, vTrue)
  check('半梯度 TD 的误差在下降', sg.err.at(-1)! < sg.err[0], `${sg.err[0].toFixed(3)} → ${sg.err.at(-1)!.toFixed(3)}`)

  /* 致命三位一体：三条腿齐全才发散 */
  const base = { alpha: 0.01, gamma: 0.99, steps: 1000, seed: 4 }
  const full = bairdCounterexample({ ...base, offPolicy: true, bootstrap: true, approx: true })
  note(`三条腿齐全   ‖w‖ ${full.norm[0].toFixed(2)} → ${full.norm.at(-1)!.toExponential(2)}`)
  check('三条腿齐全时参数发散（正文断言）', full.norm.at(-1)! > full.norm[0] * 10, `涨了 ${(full.norm.at(-1)! / full.norm[0]).toExponential(1)} 倍`)

  const legs = [
    ['去掉离策略  ', { offPolicy: false, bootstrap: true, approx: true }],
    ['去掉自举    ', { offPolicy: true, bootstrap: false, approx: true }],
    ['去掉函数近似', { offPolicy: true, bootstrap: true, approx: false }],
  ] as const
  for (const [label, cfg] of legs) {
    const r = bairdCounterexample({ ...base, ...cfg })
    note(`${label} ‖w‖ ${r.norm[0].toFixed(2)} → ${r.norm.at(-1)!.toExponential(2)}`)
    check(`${label.trim()}后不再发散`, r.norm.at(-1)! < full.norm.at(-1)! / 100, `${r.norm.at(-1)!.toExponential(2)} ≪ ${full.norm.at(-1)!.toExponential(2)}`)
  }
}

/* ─────────────────── 第 9~10 章 策略梯度与 Actor-Critic ─────────────────── */
head('第 9~10 章 · 策略梯度与 Actor-Critic')
{
  const env = mdpToEnv(mdp, { start: 0, horizon: 30 })
  const variants = ['reinforce', 'reinforce-baseline', 'qac', 'a2c'] as const
  const runs = variants.map((v) => ({
    v,
    r: policyGradient(env, v, {
      gamma: GAMMA,
      alphaTheta: 0.05,
      alphaW: 0.1,
      episodes: 1500,
      seed: 8,
      probes: 40,
    }),
  }))
  const avg = (a: number[]) => a.reduce((x, y) => x + y, 0) / a.length

  for (const { v, r } of runs) {
    const h = avg(r.episodeReturn.slice(0, 100))
    const t = avg(r.episodeReturn.slice(-100))
    note(`${v.padEnd(19)} 回报 ${h.toFixed(2)} → ${t.toFixed(2)}`)
    check(`${v} 的回报确实在上升`, t > h, `${h.toFixed(2)} → ${t.toFixed(2)}`)
  }

  const varOf = (a: number[]) => {
    const s = a.slice(-400)
    const m = avg(s)
    return avg(s.map((x) => (x - m) ** 2))
  }
  const vPlain = varOf(runs[0].r.episodeReturn)
  const vBase = varOf(runs[1].r.episodeReturn)
  note(`末段回报方差： REINFORCE ${vPlain.toFixed(4)}   加基线后 ${vBase.toFixed(4)}`)
  check('加基线不会把回报方差做大（正文卖点）', vBase <= vPlain * 1.05, `${vBase.toFixed(4)} vs ${vPlain.toFixed(4)}`)

  const gPlain = avg(runs[0].r.gradNorm.slice(-400))
  const gBase = avg(runs[1].r.gradNorm.slice(-400))
  note(`末段更新量模长： REINFORCE ${gPlain.toFixed(3)}   加基线后 ${gBase.toFixed(3)}`)
  check('基线让更新量更小更稳（方差缩减的直接证据）', gBase < gPlain, `${gBase.toFixed(3)} < ${gPlain.toFixed(3)}`)

  const probs = runs[3].r.policy
  check('softmax 策略每个动作概率恒正（天然探索，无需 ε）', Math.min(...probs.flat()) > 0, `min π = ${Math.min(...probs.flat()).toExponential(2)}`)
  const sums = probs.map((row) => row.reduce((a, b) => a + b, 0))
  check(
    '每个状态的动作概率和为 1',
    sums.every((x) => Math.abs(x - 1) < 1e-9),
    `最大偏差 ${Math.max(...sums.map((x) => Math.abs(x - 1))).toExponential(2)}`,
  )
}

console.log(
  `\n${failed === 0 ? '\x1b[1;32m全部通过\x1b[0m' : '\x1b[1;31m有断言失败\x1b[0m'}  ${passed} passed, ${failed} failed\n`,
)
if (failed > 0) process.exit(1)
