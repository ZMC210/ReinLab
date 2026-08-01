import { useMemo, useState } from 'react'
import { affineRewards, buildGridMDP, classicGrid, symmetricGrid } from '../core/mdp'
import { deterministicPolicy, qFromV } from '../core/policy'
import {
  normInf,
  policyEvaluationDirect,
  valueIterationSolve,
  valueIterationTrace,
} from '../core/solvers'
import { useBus } from '../highlight/bus'
import { Act, Beat } from '../narrative/Act'
import { ChapterGlance, ChapterHero, type Glance } from '../narrative/ChapterShell'
import { PredictChoice, PredictNumber } from '../narrative/Predict'
import { LiveFormula } from '../formula/LiveFormula'
import { useFormulaCtx } from '../formula/core'
import { bellmanOptimality, contractionStatement } from '../formula/bellman'
import { GridWorld } from '../viz/GridWorld'
import { LineChart } from '../viz/LineChart'
import { Scrubber } from '../ui/Scrubber'
import { Callout, Code, Details, M, MB, Panel, Seg, Slider, Toggle } from '../ui/prims'
import { fmt, useColors } from '../theme'

const GLANCE: Glance = {
  formula: String.raw`v_*(s) \;=\; \max_{a}\left[
  \sum_{r} p(r\mid s,a)\,r \; + \;\gamma\sum_{s'} p(s'\mid s,a)\,v_*(s')\right]`,
  formulaNote:
    '贝尔曼最优公式（BOE）。与上一章唯一的差别是把「按 π 加权平均」换成了「取最大」—— 一个 max，就把「评估一个策略」变成了「找出最好的策略」。',
  takeaways: [
    <>
      策略之间的比较是<strong>逐状态</strong>的：<M>{'\\pi_1 \\ge \\pi_2'}</M> 意味着
      在<em>每一个</em>状态上都不差。最优策略的存在性因此并不显然，需要定理保证。
    </>,
    <>
      BOE 的右端是一个压缩映射，所以 <M>{'v_*'}</M> <strong>存在且唯一</strong>，
      从任意初值反复迭代都会收敛到它。这个迭代就是<strong>值迭代算法</strong>。
    </>,
    <>
      <M>{'v_*'}</M> 唯一，但 <M>{'\\pi_*'}</M> <strong>可以有多个</strong>：
      两个动作打平时选谁都行。
    </>,
    <>
      奖励做仿射变换 <M>{'r \\to ar+b\\ (a>0)'}</M> 不改变最优策略；
      而改变 <M>{'\\gamma'}</M> 会实实在在地翻转箭头。
    </>,
  ],
  traps: [
    <>
      以为 BOE 是「另一个贝尔曼公式」。它<strong>不对应任何给定策略</strong>，
      是一个关于 <M>{'v_*'}</M> 自身的非线性方程（max 不是线性算子）。
    </>,
    <>
      以为「最优」= 奖励最大。最优是<strong>折扣回报的期望</strong>最大，
      短期吃亏换长期收益完全可能是最优的。
    </>,
    <>
      想靠「给禁区加大惩罚」来禁止智能体进禁区。加常数没用（仿射不变性），
      要改变行为得改<strong>相对</strong>大小或改 <M>{'\\gamma'}</M>。
    </>,
  ],
}

const F_BOE_MAX = bellmanOptimality('max')
const F_BOE_PI = bellmanOptimality('pi')
const F_CONTRACT = contractionStatement()

/** 有多少个格子的最优动作会「主动踏进禁区」——抄近道的直接证据 */
function countForbiddenEntries(mdp: ReturnType<typeof buildGridMDP>, policy: number[][]): number {
  const { grid } = mdp
  let n = 0
  for (let s = 0; s < mdp.nS; s++) {
    const a = policy[s].indexOf(Math.max(...policy[s]))
    const nr = Math.floor(s / grid.cols) + ACTION_DELTA[a][0]
    const nc = (s % grid.cols) + ACTION_DELTA[a][1]
    if (nr < 0 || nr >= grid.rows || nc < 0 || nc >= grid.cols) continue
    if (grid.cells[nr * grid.cols + nc] === 'forbidden') n++
  }
  return n
}

const ACTION_DELTA: [number, number][] = [
  [-1, 0],
  [0, 1],
  [1, 0],
  [0, -1],
  [0, 0],
]

/** 两个各有胜负、无法比较的策略：一个向右优先，一个向下优先 */
function rivalPolicies(nS: number, cols: number) {
  const right = deterministicPolicy(
    Array.from({ length: nS }, (_, s) => (s % cols === cols - 1 ? 2 : 1)),
  )
  const down = deterministicPolicy(
    Array.from({ length: nS }, (_, s) => (s >= nS - cols ? 1 : 2)),
  )
  return { right, down }
}

export function Chapter3() {
  const C = useColors()
  const [gamma, setGamma] = useState(0.9)
  const [k, setK] = useState(0)
  const [alpha, setAlpha] = useState(1)
  const [beta, setBeta] = useState(0)
  const [showTies, setShowTies] = useState(true)
  const [boeForm, setBoeForm] = useState<'pi' | 'max'>('pi')
  const [logAxis, setLogAxis] = useState(true)

  const focus = useBus((s) => s.focus)
  const setFocus = useBus((s) => s.setFocus)

  const mdp = useMemo(() => buildGridMDP(classicGrid()), [])
  const s = focus ?? 6

  /* 值迭代轨迹：既是本章的求解算法，也是第 4 章的预告 */
  const trace = useMemo(() => valueIterationTrace(mdp, gamma, 90), [mdp, gamma])
  const kk = Math.min(k, trace.length - 1)
  const step = trace[kk]
  const final = trace[trace.length - 1]

  const errors = useMemo(
    () => trace.map((t) => normInf(t.v, final.v)),
    [trace, final],
  )
  const boundCurve = useMemo(
    () => errors.map((_, i) => (errors[0] || 1) * Math.pow(gamma, i)),
    [errors, gamma],
  )

  /* 对称世界：并列最优的现场 */
  const symMdp = useMemo(() => buildGridMDP(symmetricGrid()), [])
  const sym = useMemo(() => valueIterationSolve(symMdp, 0.9), [symMdp])
  const tieCount = sym.ties.filter((t) => t.length > 1).length

  /* 仿射变换：换了奖励，最优策略动不动 */
  const affMdp = useMemo(
    () => buildGridMDP(affineRewards(classicGrid(), alpha, beta)),
    [alpha, beta],
  )
  const aff = useMemo(() => valueIterationSolve(affMdp, gamma), [affMdp, gamma])
  const base = useMemo(() => valueIterationSolve(mdp, gamma), [mdp, gamma])

  const policyDiff = useMemo(
    () =>
      base.policy.reduce(
        (acc, row, i) => acc + (row.indexOf(Math.max(...row)) === aff.policy[i].indexOf(Math.max(...aff.policy[i])) ? 0 : 1),
        0,
      ),
    [base, aff],
  )

  /* γ 实验：和 γ = 0.9 相比，有几个格子的箭头翻了 */
  const ref09 = useMemo(() => valueIterationSolve(mdp, 0.9).policy, [mdp])
  const countFlips = (p: number[][]) =>
    p.reduce(
      (acc, row, i) =>
        acc + (row.indexOf(Math.max(...row)) === ref09[i].indexOf(Math.max(...ref09[i])) ? 0 : 1),
      0,
    )
  const flips = useMemo(() => countFlips(base.policy), [base, ref09])
  const flipsAt01 = useMemo(() => countFlips(valueIterationSolve(mdp, 0.1).policy), [mdp, ref09])

  /**
   * 「敢不敢穿禁区」随 γ 变化的相变曲线。
   * 这是 γ 实验里最有说服力的一张图：它把「性格」量化成了一个可以看见的数。
   */
  const GAMMAS = useMemo(
    () => Array.from({ length: 39 }, (_, i) => 0.05 + i * 0.024),
    [],
  )
  const riskCurve = useMemo(
    () => GAMMAS.map((g) => countForbiddenEntries(mdp, valueIterationSolve(mdp, g).policy)),
    [GAMMAS, mdp],
  )
  const gammaIdx = Math.round((gamma - 0.05) / 0.024)
  const risk = countForbiddenEntries(mdp, base.policy)

  const rivals = useMemo(() => rivalPolicies(mdp.nS, mdp.grid.cols), [mdp])
  const vRight = useMemo(() => policyEvaluationDirect(mdp, rivals.right, 0.9), [mdp, rivals])
  const vDown = useMemo(() => policyEvaluationDirect(mdp, rivals.down, 0.9), [mdp, rivals])
  const rightWins = vRight.filter((x, i) => x > vDown[i] + 1e-9).length
  const downWins = vDown.filter((x, i) => x > vRight[i] + 1e-9).length

  const ctx = useFormulaCtx(mdp, base.policy, gamma, base.v, s)

  return (
    <div>
      <ChapterHero
        n={3}
        hook="能给策略打分了。可全世界这么多策略，怎么找到最好的那一个？"
        lead={
          <>
            <p>
              这一章和上一章的全部差别，就是一个 <M>{'\\max'}</M>。
              但就是这一笔，把一个规规矩矩的线性方程组，变成了一个非线性方程 ——
              代价是没法一步解出来，回报是它刻画了「最优」。
            </p>
            <p>
              更要紧的是，这一章要先解决一个容易被跳过的问题：
              <em>「最好」这两个字，在有 25 个状态的世界里到底是什么意思？</em>
              如果策略 A 在左半边更好、策略 B 在右半边更好，谁更好？
            </p>
          </>
        }
        gains={[
          '知道「策略之间的比较」是一个偏序，以及为什么最优策略仍然存在',
          '能从 max 的形式推出 max_π Σπq = max_a q',
          '理解贝尔曼最优公式为什么有且只有一个解',
          '亲手验证：最优策略可以不唯一',
          '亲手验证：奖励做仿射变换，最优策略纹丝不动',
          '看见 γ 如何在「抄近道」和「绕远路」之间翻转箭头',
        ]}
      />

      <ChapterGlance g={GLANCE} />

      {/* ───────────────────────── 第 1 幕 ───────────────────────── */}
      <Act
        id="a1"
        no="第 1 幕"
        title="「更好」是什么意思"
        goal="策略之间的比较是逐状态的，这让「最好」的存在性成了一件需要证明的事。"
        minutes={9}
        points={[
          <>
            <M>{'\\pi_1 \\ge \\pi_2'}</M> 的定义是
            <M>{'v_{\\pi_1}(s)\\ge v_{\\pi_2}(s)\\ \\forall s'}</M> ——
            <strong>每一个状态</strong>都不能差。
          </>,
          <>
            这是<strong>偏序</strong>不是全序：两个策略完全可能各有胜场、互不可比。
          </>,
          <>
            所以「存在一个在所有状态上都最好的策略」是一个
            <strong>需要证明的结论</strong>，不是定义。
          </>,
        ]}
        stage={() => (
          <div className="space-y-5">
            <Panel title="策略甲：一路向右">
              <GridWorld mdp={mdp} v={vRight} policy={rivals.right} cell={48} quiet />
            </Panel>
            <Panel title="策略乙：一路向下">
              <GridWorld mdp={mdp} v={vDown} policy={rivals.down} cell={48} quiet />
            </Panel>
            <Panel title="逐格比较">
              <div className="grid grid-cols-2 gap-3 text-center">
                <div className="rounded-xl border border-line bg-surface2 p-3">
                  <div className="text-[11px] text-faint">甲更好的格子</div>
                  <div className="mt-1 font-mono text-[22px]" style={{ color: C.accent }}>
                    {rightWins}
                  </div>
                </div>
                <div className="rounded-xl border border-line bg-surface2 p-3">
                  <div className="text-[11px] text-faint">乙更好的格子</div>
                  <div className="mt-1 font-mono text-[22px]" style={{ color: C.qvalue }}>
                    {downWins}
                  </div>
                </div>
              </div>
              <p className="mt-3 text-[11.5px] leading-relaxed text-faint">
                两边都非零 —— 这两个策略<strong>无法比较</strong>。
              </p>
            </Panel>
          </div>
        )}
      >
        <Beat id="b1">
          <p>
            上一章给了我们一把尺子 <M>{'v_\\pi'}</M>。但这把尺子量出来的不是一个数，
            而是<strong>一个 25 维的向量</strong>。两个向量怎么比大小？
          </p>
          <p>强化学习采用最严格的定义：</p>
          <MB>{'\\pi_1 \\ge \\pi_2 \\quad \\Longleftrightarrow \\quad v_{\\pi_1}(s) \\ge v_{\\pi_2}(s) \\;\\; \\text{对每一个 } s \\text{ 都成立}'}</MB>
          <p>
            注意「对<em>每一个</em>」。哪怕只有一个状态上 <M>{'\\pi_1'}</M> 输了，
            这个不等号就不成立。这是一个<strong>偏序</strong>而不是全序 ——
            允许存在两个谁也压不倒谁的策略。
          </p>
        </Beat>

        <Beat id="b2">
          <p>
            右边就是一对这样的策略：甲一路向右、乙一路向下。
            甲在 <strong>{rightWins}</strong> 个格子上更好，乙在 <strong>{downWins}</strong> 个格子上更好。
            <em>它们无法比较。</em>
          </p>

          <PredictChoice
            id="ch3-partial-order"
            question="既然存在互不相让的策略，那么「最优策略」还一定存在吗？"
            options={[
              { id: 'a', label: '不一定。可能所有策略都互相压不倒，根本没有最好的。' },
              {
                id: 'b',
                label:
                  '一定存在。有限 MDP 里必定存在一个策略，它在每一个状态上都不比任何其它策略差。',
              },
              { id: 'c', label: '要看 γ。γ 小的时候存在，γ 接近 1 时可能不存在。' },
            ]}
            answer="b"
            explain={
              <>
                <p>
                  这是强化学习里最令人安心的一个结论：
                  <strong>对有限 MDP，总存在一个策略 <M>{'\\pi^*'}</M>，
                  使得对所有 <M>{'\\pi'}</M> 和所有 <M>{'s'}</M> 都有{' '}
                  <M>{'v_{\\pi^*}(s) \\ge v_\\pi(s)'}</M>。</strong>
                </p>
                <p>
                  它一点也不显然 —— 偏序集里通常没有最大元。
                  之所以这里有，是因为可以「逐状态取最好」而彼此不冲突：
                  在每个状态各自选最优动作拼出来的那个策略，恰好整体也是最优的。
                  <em>而这件事的证明，正是贝尔曼最优公式 + 压缩映射定理。</em>
                </p>
                <p>
                  所以这一章的顺序是：先写出方程 → 证明它有唯一解 →
                  证明那个解就是 <M>{'v^*'}</M> → 从解里读出 <M>{'\\pi^*'}</M>。
                </p>
              </>
            }
          />
        </Beat>
      </Act>

      {/* ───────────────────────── 第 2 幕 ───────────────────────── */}
      <Act
        id="a2"
        no="第 2 幕"
        title="加一个 max"
        goal="贝尔曼最优公式的两种等价写法，以及为什么最优策略一定可以是确定性的。"
        minutes={11}
        points={[
          <>
            两种写法等价：<M>{'v(s)=\\max_a q(s,a)'}</M> 与{' '}
            <M>{'v(s)=\\max_\\pi \\sum_a \\pi(a|s)q(s,a)'}</M>。
          </>,
          <>
            关键一步：<strong>对概率分布加权平均，永远不会超过最大值</strong>。
            所以最优权重就是「把 1 全押在 argmax 上」。
          </>,
          <>
            推论：<strong>一定存在确定性的最优策略</strong>。
            后面所有算法都能安心地只找确定性策略。
          </>,
        ]}
        stage={() => (
          <div className="space-y-5">
            <Panel
              title="贝尔曼最优公式"
              right={
                <Seg
                  size="sm"
                  value={boeForm}
                  onChange={setBoeForm}
                  options={[
                    { value: 'pi', label: '对策略取 max' },
                    { value: 'max', label: '对动作取 max' },
                  ]}
                />
              }
            >
              <LiveFormula node={boeForm === 'pi' ? F_BOE_PI : F_BOE_MAX} ctx={ctx} />
              <p className="mt-3 border-t border-line pt-3 text-[11.5px] leading-relaxed text-faint">
                两种写法完全等价。左边那种更贴近「在所有策略里挑最好的」这句话，
                右边那种才是能算的形式。
              </p>
            </Panel>
            <Panel title="最优策略与最优价值">
              <GridWorld
                mdp={mdp}
                v={base.v}
                q={qFromV(mdp, gamma, base.v)}
                policy={base.policy}
                ties={showTies ? base.ties : undefined}
                showQ
                cell={54}
                onCellClick={(x) => setFocus(x === focus ? null : x)}
              />
              <div className="mt-3">
                <Slider
                  label={<M>{'\\gamma'}</M>}
                  value={gamma}
                  min={0.1}
                  max={0.99}
                  step={0.01}
                  onChange={setGamma}
                  accent={C.gamma}
                />
              </div>
            </Panel>
          </div>
        )}
      >
        <Beat id="c1">
          <p>
            贝尔曼公式描述的是「某一个给定策略」的价值。
            现在我们不给定策略了，而是在每一步都挑最好的：
          </p>
          <MB>{'v(s) = \\max_{\\pi(s) \\in \\Pi} \\sum_{a} \\pi(a \\mid s) \\left[ r(s,a) + \\gamma \\sum_{s\'} p(s\'\\mid s,a) v(s\') \\right]'}</MB>
          <p>
            这就是<strong>贝尔曼最优公式</strong>（BOE）。
            和上一章那个式子逐字对比，差别只有最前面多了一个 <M>{'\\max'}</M>。
          </p>
          <Callout tone="trap" title="注意这里的微妙之处">
            这个方程里，<M>{'v'}</M> 出现在左右两边，而 <M>{'\\pi'}</M> 也是未知的。
            <strong>两个未知量、一个方程</strong>。这看上去像是无解的，
            但接下来会看到：<M>{'\\max'}</M> 这个操作会把 <M>{'\\pi'}</M> 自动消掉。
          </Callout>
        </Beat>

        <Beat id="c2">
          <h3>max 会把 π 吃掉</h3>
          <p>
            把方括号里的东西记作 <M>{'q(s,a)'}</M>，问题就变成了：
            给定一组数 <M>{'q(s,a_1), \\dots, q(s,a_5)'}</M>，
            如何选一个概率分布 <M>{'\\pi(\\cdot\\mid s)'}</M> 使加权和最大？
          </p>
          <MB>{'\\max_{\\pi(s)} \\sum_{a} \\pi(a\\mid s)\\, q(s,a) = \\max_{a} q(s,a)'}</MB>

          <Details summary="为什么加权平均的最大值就是最大项（三行证明）" defaultOpen>
            <p>
              设 <M>{'a^* = \\arg\\max_a q(s,a)'}</M>。对任意满足{' '}
              <M>{'\\sum_a \\pi(a|s) = 1,\\ \\pi \\ge 0'}</M> 的分布：
            </p>
            <MB>{'\\sum_a \\pi(a\\mid s) q(s,a) \\;\\le\\; \\sum_a \\pi(a\\mid s) q(s,a^*) \\;=\\; q(s,a^*)'}</MB>
            <p>
              而取 <M>{'\\pi(a^*\\mid s) = 1'}</M> 时等号成立。所以上确界可达，且
            </p>
            <MB>{'\\pi^*(a \\mid s) = \\begin{cases} 1, & a = a^* \\\\ 0, & \\text{否则} \\end{cases}'}</MB>
            <Callout tone="insight" title="一个重要推论">
              <strong>最优策略总可以取成确定性的。</strong>
              随机性在这里没有任何好处 —— 把概率分给次优动作只会拉低加权平均。
              <em>（注意：这个结论依赖「环境模型已知、可以放心贪心」。
              第 5 章之后需要探索时，我们会主动地、故意地把随机性加回来。）</em>
            </Callout>
          </Details>

          <p>于是 BOE 化简成一个只含 <M>{'v'}</M> 的方程：</p>
          <MB>{'v(s) = \\max_{a \\in \\mathcal{A}} \\left[ r(s,a) + \\gamma \\sum_{s\'} p(s\'\\mid s,a)\\, v(s\') \\right]'}</MB>
          <p>
            <M>{'\\pi'}</M> 消失了。但代价是：<strong><M>{'\\max'}</M> 让这个方程变成非线性的</strong>，
            上一章那招「移项、求逆」彻底失效。
          </p>
        </Beat>

        <Beat id="c3">
          <PredictChoice
            id="ch3-nonlinear"
            question={
              <>
                既然 <M>{'\\max'}</M> 让方程非线性，那我们怎么知道它<strong>有解</strong>，
                甚至<strong>只有一个解</strong>？
              </>
            }
            options={[
              { id: 'a', label: '非线性方程通常有多个解，所以只能碰运气找一个。' },
              {
                id: 'b',
                label:
                  '把右边看成一个映射 f(v)，证明它是压缩映射，再用 Banach 不动点定理。',
              },
              { id: 'c', label: '对所有 5^25 个确定性策略逐一验证，取最好的。' },
            ]}
            answer="b"
            explain={
              <>
                <p>
                  正是上一章那把锤子，原封不动地再用一次。
                </p>
                <p>
                  记 <M>{'f(v)'}</M> 为 BOE 的右端。关键的一步是证明：
                  尽管里面有 <M>{'\\max'}</M>，<M>{'f'}</M> 仍然满足{' '}
                  <M>{'\\|f(v_1) - f(v_2)\\|_\\infty \\le \\gamma \\|v_1 - v_2\\|_\\infty'}</M>。
                </p>
                <p>
                  用到的是一个小引理：
                  <strong><M>{'|\\max_a x_a - \\max_a y_a| \\le \\max_a |x_a - y_a|'}</M></strong>。
                  也就是说 <M>{'\\max'}</M> 不会把差距放大。下一幕我们把它跑出来给你看。
                </p>
              </>
            }
          />
        </Beat>
      </Act>

      {/* ───────────────────────── 第 3 幕 ───────────────────────── */}
      <Act
        id="a3"
        no="第 3 幕"
        title="压缩映射，第二次登场"
        goal="BOE 有唯一解，而且反复迭代就能收敛到它 —— 这已经是值迭代算法了。"
        minutes={12}
        points={[
          <>
            <M>{'f(v)=\\max_\\pi(r_\\pi+\\gamma P_\\pi v)'}</M> 是 <M>{'\\gamma'}</M>-压缩：
            <M>{'\\|f(v_1)-f(v_2)\\|_\\infty\\le\\gamma\\|v_1-v_2\\|_\\infty'}</M>。
          </>,
          <>
            压缩映射定理直接送给我们三件事：<strong>解存在、解唯一、迭代必收敛</strong>，
            且速率是 <M>{'\\gamma^k'}</M>。
          </>,
          <>
            <M>{'v_{k+1}=f(v_k)'}</M> 这行迭代<strong>就是值迭代算法本身</strong>。
            第 4 章只是给它换了个工程外壳。
          </>,
        ]}
        stage={() => (
          <div className="space-y-5">
            <Panel title={`第 ${kk} 次迭代`}>
              <GridWorld
                mdp={mdp}
                v={step.v}
                policy={step.policy}
                ties={showTies ? step.ties : undefined}
                cell={54}
                onCellClick={(x) => setFocus(x === focus ? null : x)}
              />
              <div className="mt-4">
                <Scrubber k={kk} setK={setK} max={trace.length - 1} />
              </div>
              <div className="mt-3 flex items-center justify-between gap-3">
                <span className="font-mono text-[12px] text-dim">
                  ‖v<sub>k</sub> − v*‖<sub>∞</sub> ={' '}
                  <span style={{ color: C.accent }}>{fmt(errors[kk], 4)}</span>
                </span>
                <Toggle label="标出并列最优" checked={showTies} onChange={setShowTies} />
              </div>
              <p className="mt-2 text-[11.5px] leading-relaxed text-faint">
                注意策略箭头：在价值还远没收敛的时候，箭头往往<strong>已经稳定了</strong>。
                这个现象是第 4 章「截断策略迭代」的种子。
              </p>
            </Panel>

            <Panel
              title="误差衰减"
              right={<Toggle label="对数纵轴" checked={logAxis} onChange={setLogAxis} />}
            >
              <LineChart
                series={[
                  { name: '‖v_k − v*‖∞', color: C.accent, data: errors },
                  { name: 'γᵏ 上界', color: C.gamma, data: boundCurve, dashed: true },
                ]}
                xLabel="迭代次数 k"
                logY={logAxis}
                marker={kk}
                height={200}
              />
            </Panel>

            <Panel title="压缩性">
              <LiveFormula node={F_CONTRACT} ctx={ctx} inert />
              <p className="text-[11.5px] leading-relaxed text-faint">
                和上一章一字不差，只是这里的 f 里面多了一个 max。
              </p>
            </Panel>
          </div>
        )}
      >
        <Beat id="d1">
          <p>
            把 BOE 的右端记作映射 <M>{'f'}</M>：
          </p>
          <MB>{'f(v)(s) := \\max_{a} \\left[ r(s,a) + \\gamma \\sum_{s\'} p(s\'\\mid s,a) v(s\') \\right]'}</MB>
          <p>
            那么 BOE 就是在找 <M>{'f'}</M> 的不动点 <M>{'v = f(v)'}</M>。
          </p>

          <Details summary="证明 f 是压缩映射（关键在 max 不放大差距）">
            <p>
              先看一个小引理。对任意两组实数 <M>{'\\{x_a\\}'}</M>、<M>{'\\{y_a\\}'}</M>：
            </p>
            <MB>{'\\left| \\max_a x_a - \\max_a y_a \\right| \\le \\max_a \\left| x_a - y_a \\right|'}</MB>
            <p>
              直觉：设 <M>{'\\max_a x_a = x_{a_1}'}</M>，则{' '}
              <M>{'\\max_a x_a - \\max_a y_a \\le x_{a_1} - y_{a_1} \\le \\max_a|x_a-y_a|'}</M>；
              交换 <M>{'x, y'}</M> 得到另一侧。
            </p>
            <p>现在对任意状态 <M>{'s'}</M>：</p>
            <MB>{'|f(v_1)(s) - f(v_2)(s)| \\le \\max_a \\left| \\gamma \\sum_{s\'} p(s\'\\mid s,a)\\left(v_1(s\') - v_2(s\')\\right) \\right| \\le \\gamma \\|v_1 - v_2\\|_\\infty'}</MB>
            <p>
              最后一步用了 <M>{'\\sum_{s\'} p(s\'|s,a) = 1'}</M>。对 <M>{'s'}</M> 取最大即得
              <M>{'\\|f(v_1)-f(v_2)\\|_\\infty \\le \\gamma\\|v_1-v_2\\|_\\infty'}</M>。
            </p>
            <Callout tone="rigor">
              于是 Banach 不动点定理给出三件事：<strong>不动点存在、唯一、
              且从任意初值迭代都以 <M>{'\\gamma^k'}</M> 的速度收敛到它</strong>。
              可以进一步证明这个唯一不动点就是最优价值 <M>{'v^*'}</M>，
              对它贪心得到的确定性策略就是最优策略 <M>{'\\pi^*'}</M>。
            </Callout>
          </Details>
        </Beat>

        <Beat id="d2">
          <h3>不动点定理直接给了我们算法</h3>
          <p>
            「从任意初值迭代都会收敛」这句话，翻译成代码就是四行。
            <strong>它有个名字，叫值迭代 —— 也就是下一章的主角。</strong>
          </p>
          <Code
            lang="python"
            code={`def value_iteration(P, R, gamma, tol=1e-9):
    v = np.zeros(P.shape[0])              # 初值任取
    while True:
        q = R + gamma * P @ v             # q[s,a]
        v_new = q.max(axis=1)             # ← 全部差别就是这个 max
        if np.max(np.abs(v_new - v)) < tol:
            return v_new, q.argmax(axis=1)   # 返回 v* 和贪心策略 π*
        v = v_new`}
          />
          <p>
            把它和第 2 章的策略评估放在一起对比：那里是{' '}
            <code>(pi * q).sum(axis=1)</code>，这里是 <code>q.max(axis=1)</code>。
            <em>「按策略加权平均」变成了「取最大」，一个函数换成另一个函数，仅此而已。</em>
          </p>
        </Beat>

        <Beat id="d3">
          <PredictChoice
            id="ch3-policy-stable"
            question={
              <>
                拖动右边的时间轴，观察箭头。你会发现价值还在明显变化的时候，
                <strong>大部分箭头已经不动了</strong>。这说明什么？
              </>
            }
            options={[
              { id: 'a', label: '算法有 bug，箭头应该和价值同步变化。' },
              {
                id: 'b',
                label:
                  '贪心策略只关心 q 值的相对大小。价值整体还在往上抬，但排序早就定了。',
              },
              { id: 'c', label: '巧合，换个 γ 就不会这样了。' },
            ]}
            answer="b"
            explain={
              <>
                <p>
                  <strong>策略只依赖 q 的排序，不依赖 q 的绝对值。</strong>
                  价值函数还需要几十次迭代才能精确收敛，但动作之间的高下往往几次迭代就分明了。
                </p>
                <p>
                  这个观察价值连城：如果我们的最终目的是拿到<em>策略</em>而不是<em>精确的价值</em>，
                  那么把策略评估算到底就是浪费。
                  第 4 章的「截断策略迭代」正是基于这一点 ——
                  它会告诉你，值迭代和策略迭代其实是同一个算法谱系上的两个端点。
                </p>
              </>
            }
          />
        </Beat>
      </Act>

      {/* ───────────────────────── 第 4 幕 ───────────────────────── */}
      <Act
        id="a4"
        no="第 4 幕"
        title="实验一：最优策略不唯一"
        goal="v* 唯一，但 π* 可以有很多个 —— 分清这两件事。"
        minutes={7}
        points={[
          <>
            <M>{'v_*'}</M> 是方程的解，<strong>唯一</strong>；
            <M>{'\\pi_*'}</M> 是从 <M>{'v_*'}</M> 读出的 argmax，<strong>可以不唯一</strong>。
          </>,
          <>
            对称世界里，两条等长路径的 q 值完全打平 —— 选谁都是最优。
          </>,
        ]}
        stage={() => (
          <Panel title="完全对称的世界">
            <GridWorld
              mdp={symMdp}
              v={sym.v}
              policy={sym.policy}
              ties={showTies ? sym.ties : undefined}
              cell={62}
              showLabels
              quiet
            />
            <div className="mt-3 flex items-center justify-between">
              <span className="text-[12px] text-dim">
                出现并列最优动作的格子：
                <span className="ml-1 font-mono text-[15px]" style={{ color: C.qvalue }}>
                  {tieCount}
                </span>
              </span>
              <Toggle label="标出并列（空心箭头）" checked={showTies} onChange={setShowTies} />
            </div>
            <p className="mt-3 text-[11.5px] leading-relaxed text-faint">
              目标在正中央，四个角到目标都有两条等长的路。
              这些格子上「先右后下」和「先下后右」的 q 值<strong>严格相等</strong>。
            </p>
          </Panel>
        )}
      >
        <Beat id="e1">
          <PredictChoice
            id="ch3-unique"
            question={
              <>
                贝尔曼最优公式的解 <M>{'v^*'}</M> 是唯一的。
                那么最优策略 <M>{'\\pi^*'}</M> 也是唯一的吗？
              </>
            }
            options={[
              { id: 'a', label: '是。v* 唯一，对它贪心的结果当然也唯一。' },
              { id: 'b', label: '不一定。当某个状态上有多个动作的 q 值相等时，最优策略就不唯一。' },
              { id: 'c', label: '不是。因为 v* 其实也可能不唯一。' },
            ]}
            answer="b"
            explain={
              <>
                <p>
                  <strong><M>{'v^*'}</M> 唯一，<M>{'\\pi^*'}</M> 不一定唯一。</strong>
                  这是初学时最常见的混淆之一。
                </p>
                <p>
                  原因很简单：贪心操作 <M>{'\\arg\\max_a q^*(s,a)'}</M> 在出现并列时，
                  返回的是一个<em>集合</em>而不是一个元素。
                  从集合里任意挑一个（甚至在并列动作之间任意分配概率），
                  得到的都是最优策略。
                </p>
                <p>
                  右边这个对称世界里有 <strong>{tieCount}</strong> 个格子出现了并列。
                  打开「标出并列」，空心箭头就是那些同样最优的选择。
                  <em>光是从这几个格子的组合，就能拼出好几个不同的最优策略，
                  而它们的 v 完全一样。</em>
                </p>
              </>
            }
          />
          <Callout tone="intuition" title="换个说法">
            <M>{'v^*'}</M> 是「这个世界最好能有多好」，这是世界的属性，当然唯一。
            <M>{'\\pi^*'}</M> 是「怎么做才能那么好」，条条大路通罗马时，答案自然不止一个。
          </Callout>
        </Beat>
      </Act>

      {/* ───────────────────────── 第 5 幕 ───────────────────────── */}
      <Act
        id="a5"
        no="第 5 幕"
        title="实验二：把所有奖励加 100，会怎样"
        goal="奖励的仿射变换不改变最优策略 —— 一个能救你很多次的定理。"
        minutes={10}
        points={[
          <>
            <M>{'r \\to ar + b'}</M>（<M>{'a>0'}</M>）之后，
            <M>{"v_*' = a\\,v_* + \\tfrac{b}{1-\\gamma}"}</M>，
            <strong>最优策略完全不变</strong>。
          </>,
          <>
            直觉：所有动作都被同等地抬高了，<strong>相对排序</strong>没动，
            而 argmax 只看相对排序。
          </>,
          <>
            实践含义：调奖励时，加减常数是白费力气。
            要改行为，得改<strong>不同动作之间的差值</strong>。
          </>,
        ]}
        stage={() => (
          <div className="space-y-5">
            <Panel title="原始奖励">
              <GridWorld mdp={mdp} v={base.v} policy={base.policy} cell={48} quiet />
            </Panel>
            <Panel title={`变换后：r → ${fmt(alpha)}·r ${beta >= 0 ? '+' : '−'} ${fmt(Math.abs(beta))}`}>
              <GridWorld mdp={affMdp} v={aff.v} policy={aff.policy} cell={48} quiet />
              <div className="mt-4 space-y-3">
                <Slider
                  label={
                    <>
                      缩放 <M>{'\\alpha'}</M>（必须为正）
                    </>
                  }
                  value={alpha}
                  min={0.1}
                  max={5}
                  step={0.1}
                  onChange={setAlpha}
                  accent={C.reward}
                />
                <Slider
                  label={
                    <>
                      平移 <M>{'\\beta'}</M>
                    </>
                  }
                  value={beta}
                  min={-100}
                  max={100}
                  step={1}
                  onChange={setBeta}
                  accent={C.reward}
                />
              </div>
              <div
                className="mt-4 rounded-xl border p-3 text-center"
                style={{
                  borderColor: `color-mix(in srgb, ${
                    policyDiff === 0 ? 'var(--value)' : 'var(--danger)'
                  } 40%, transparent)`,
                  background: `color-mix(in srgb, ${
                    policyDiff === 0 ? 'var(--value)' : 'var(--danger)'
                  } 8%, transparent)`,
                }}
              >
                <div className="text-[11.5px] text-dim">与原始最优策略不同的格子数</div>
                <div
                  className="mt-1 font-mono text-[24px]"
                  style={{ color: policyDiff === 0 ? C.value : C.danger }}
                >
                  {policyDiff}
                </div>
              </div>
            </Panel>
          </div>
        )}
      >
        <Beat id="f1">
          <PredictChoice
            id="ch3-affine"
            question={
              <>
                把这个世界里<strong>所有</strong>奖励都加上 100
                （撞墙从 <M>{'-1'}</M> 变 <M>{'99'}</M>，到达目标从 <M>{'1'}</M> 变 <M>{'101'}</M>，
                普通格子从 <M>{'0'}</M> 变 <M>{'100'}</M>）。最优策略会变吗？
              </>
            }
            options={[
              { id: 'a', label: '会大变。禁区不再是惩罚了，机器人会开始乱走。' },
              { id: 'b', label: '完全不变。一个格子的箭头都不会动。' },
              { id: 'c', label: '会小变。大部分不变，少数临界格子会翻转。' },
            ]}
            answer="b"
            explain={
              <>
                <p>
                  <strong>一个格子都不会动。</strong>右边把 <M>{'\\beta'}</M> 拖到 100 亲自看看 ——
                  价值全变了（每个都涨了一大截），但箭头纹丝不动，差异计数始终是 0。
                </p>
                <p>
                  一般结论是：对任意 <M>{'\\alpha > 0'}</M> 和任意 <M>{'\\beta'}</M>，
                  把所有 <M>{'r \\to \\alpha r + \\beta'}</M> 之后，最优策略不变，且
                </p>
                <MB>{'v^*_{\\text{new}} = \\alpha\\, v^*_{\\text{old}} + \\frac{\\beta}{1-\\gamma}'}</MB>
                <p>
                  把 <M>{'\\alpha'}</M> 拖到 0.1 或者 5，你会看到数值等比例缩放，箭头依旧不动。
                </p>
              </>
            }
          />
        </Beat>

        <Beat id="f2">
          <Details summary="为什么（把新的 v 代回 BOE 验证一下即可）">
            <p>
              猜一个解：<M>{'v\' = \\alpha v^* + c\\mathbf{1}'}</M>，其中{' '}
              <M>{'c = \\beta/(1-\\gamma)'}</M>。代入新的 BOE 右端：
            </p>
            <MB>{'\\max_a \\left[ (\\alpha r + \\beta) + \\gamma \\sum_{s\'} p\\,(\\alpha v^* + c) \\right] = \\max_a \\left[ \\alpha r + \\gamma\\alpha \\sum_{s\'} p\\, v^* \\right] + \\beta + \\gamma c'}</MB>
            <p>
              因为 <M>{'\\sum_{s\'} p(s\'|s,a) = 1'}</M>，常数 <M>{'c'}</M> 直接穿了出来。
              提出 <M>{'\\alpha > 0'}</M>（<strong>正数才能穿过 max</strong>）：
            </p>
            <MB>{'= \\alpha \\max_a \\left[ r + \\gamma \\sum_{s\'} p\\, v^* \\right] + \\beta + \\gamma c = \\alpha v^* + \\beta + \\gamma c = \\alpha v^* + c = v\''}</MB>
            <p>
              最后一步用了 <M>{'\\beta + \\gamma c = \\beta + \\gamma\\beta/(1-\\gamma) = \\beta/(1-\\gamma) = c'}</M>。
              所以 <M>{'v\''}</M> 确实是新 BOE 的解；由唯一性，它就是新的 <M>{'v^*'}</M>。
            </p>
            <p>
              而 <M>{'\\arg\\max'}</M> 在整体缩放正数、平移常数之后不变，所以{' '}
              <M>{'\\pi^*'}</M> 不变。
            </p>
            <Callout tone="trap" title="α 必须为正">
              把 <M>{'\\alpha'}</M> 取成负数，<M>{'\\max'}</M> 会翻成 <M>{'\\min'}</M>，
              整个论证崩塌 —— 那相当于把奖励变成惩罚，最优策略当然会天翻地覆。
            </Callout>
          </Details>

          <Callout tone="insight" title="这个定理什么时候能救你">
            <ul>
              <li>
                <strong>数值稳定性</strong>：奖励尺度过大导致训练发散时，
                可以放心地整体缩放，不必担心改变了要解的问题。
              </li>
              <li>
                <strong>调试直觉</strong>：如果你只是给所有奖励加了个常数，
                智能体的行为却大变 —— 那一定是别的地方有 bug。
              </li>
              <li>
                <strong>提防误用</strong>：第 1 章那道题里「只给普通格子加 <M>{'-0.1'}</M>」
                不是仿射变换，它确实改变了问题。区别就在<strong>「所有」</strong>两个字。
              </li>
            </ul>
          </Callout>
        </Beat>
      </Act>

      {/* ───────────────────────── 第 6 幕 ───────────────────────── */}
      <Act
        id="a6"
        no="第 6 幕"
        title="实验三：γ 在「抄近道」和「绕远路」之间翻转箭头"
        goal="折扣因子不只是收敛的技术条件，它实实在在地改变智能体的行为。"
        minutes={9}
        points={[
          <>
            γ 小 = 目光短浅：宁可挨一下罚也要抄近道，因为远处的奖励被折没了。
          </>,
          <>
            γ 大 = 有耐心：愿意多绕几步避开禁区，因为未来的收益还值钱。
          </>,
          <>
            所以 <strong>γ 是一个行为超参数</strong>，不只是保证收敛的技术条件。
          </>,
        ]}
        stage={() => (
          <Panel title="最优策略如何随 γ 变化">
            <GridWorld
              mdp={mdp}
              v={base.v}
              policy={base.policy}
              ties={base.ties}
              cell={54}
              onCellClick={(x) => setFocus(x === focus ? null : x)}
            />
            <div className="mt-4">
              <Slider
                label={
                  <>
                    折扣因子 <M>{'\\gamma'}</M>
                  </>
                }
                value={gamma}
                min={0.05}
                max={0.99}
                step={0.01}
                onChange={setGamma}
                accent={C.gamma}
                hint={`有效视野约 ${fmt(1 / (1 - gamma), 0)} 步`}
              />
            </div>
            <div className="mt-4 grid grid-cols-2 gap-3">
              <div
                className="rounded-xl border p-3 text-center transition-colors"
                style={{
                  borderColor:
                    flips > 0 ? 'color-mix(in srgb, var(--reward) 45%, transparent)' : 'var(--line)',
                  background:
                    flips > 0 ? 'color-mix(in srgb, var(--reward) 8%, transparent)' : 'transparent',
                }}
              >
                <div className="text-[11px] leading-snug text-dim">
                  与 <span className="font-mono">γ=0.9</span> 相比
                  <br />
                  箭头翻转的格子
                </div>
                <div
                  className="mt-1 font-mono text-[24px]"
                  style={{ color: flips > 0 ? C.reward : 'var(--ink-faint)' }}
                >
                  {flips}
                </div>
              </div>
              <div
                className="rounded-xl border p-3 text-center transition-colors"
                style={{
                  borderColor: `color-mix(in srgb, ${
                    risk > 0 ? 'var(--danger)' : 'var(--value)'
                  } 40%, transparent)`,
                  background: `color-mix(in srgb, ${
                    risk > 0 ? 'var(--danger)' : 'var(--value)'
                  } 7%, transparent)`,
                }}
              >
                <div className="text-[11px] leading-snug text-dim">
                  主动踏进禁区
                  <br />
                  抄近道的格子
                </div>
                <div
                  className="mt-1 font-mono text-[24px]"
                  style={{ color: risk > 0 ? C.danger : C.value }}
                >
                  {risk}
                </div>
              </div>
            </div>

            <div className="mt-5 border-t border-line pt-4">
              <div className="mb-1 text-[11.5px] text-faint">
                「敢穿禁区的格子数」随 γ 变化 —— 一条相变曲线
              </div>
              <LineChart
                series={[{ name: '主动踏进禁区的格子数', color: C.danger, data: riskCurve }]}
                xLabel="γ 从 0.05 到 0.98"
                marker={Math.max(0, Math.min(riskCurve.length - 1, gammaIdx))}
                height={170}
                yMin={0}
              />
              <p className="mt-2 text-[11.5px] leading-relaxed text-faint">
                左半边是一条贴着 0 的平线：<strong>短视的智能体一步也不肯踏进禁区。</strong>
                越过某个临界点之后突然跳起来 —— 它开始愿意为了早两步到手的奖励挨一下打。
              </p>
            </div>
          </Panel>
        )}
      >
        <Beat id="g1">
          <p>
            前面两个实验说的都是「什么不会变」。这一个反过来：
            <strong>有一个参数，它变一点点，整个行为就会重写。</strong>
          </p>
          <p>
            把右边的 <M>{'\\gamma'}</M> 从 0.9 慢慢往下拖。
            一开始什么都不动，然后在 <M>{'\\gamma \\approx 0.6'}</M> 附近，
            某几个格子的箭头会<em>啪地翻向另一边</em>，两个计数器同时跳动。
          </p>
          <p>
            右下角那条曲线把这件事量化了：<strong>
            <M>{'\\gamma \\ge 0.7'}</M> 时有 5~6 个格子的最优动作是「主动踏进禁区」，
            而 <M>{'\\gamma \\le 0.5'}</M> 时这个数字是 <M>{'0'}</M> —— 一个都不肯踏。
            </strong>
            同一个世界、同一套奖励，只因为一个数变了，智能体从「不怕挨打的抄近道派」
            变成了「宁可绕远的谨慎派」。
          </p>
        </Beat>

        <Beat id="g2">
          <h3>翻转的机理</h3>
          <p>
            考虑一个格子，它面前有两条路通往目标：
          </p>
          <ul>
            <li>
              <strong>近道</strong>：穿过禁区，立刻吃一个 <M>{'-1'}</M>，但少走两步；
            </li>
            <li>
              <strong>远路</strong>：绕开禁区，不吃罚，但目标晚两步到手。
            </li>
          </ul>
          <p>两者的价值差大致是</p>
          <MB>{'\\underbrace{-1}_{\\text{近道的罚，就在眼前}} \\quad \\text{vs} \\quad \\underbrace{(\\gamma^{k} - \\gamma^{k+2})\\cdot v_{\\text{target}}}_{\\text{远路的代价：奖励晚到两步}}'}</MB>
          <p>
            当 <M>{'\\gamma'}</M> 大时，远处的奖励几乎不打折，「晚两步」损失惨重，
            于是<strong>宁可吃一个 <M>{'-1'}</M> 也要抄近道</strong>。
            当 <M>{'\\gamma'}</M> 小时，目标本来就看不太清了，
            眼前这个实打实的 <M>{'-1'}</M> 反而成了主要矛盾，于是<strong>绕路</strong>。
          </p>
          <Callout tone="insight" title="γ 是性格参数">
            <strong>大 <M>{'\\gamma'}</M> = 有耐心、有远见、敢为长远利益吃眼前亏。</strong>
            <br />
            <strong>小 <M>{'\\gamma'}</M> = 短视、保守、只想避开眼前的疼。</strong>
            <br />
            所以调 <M>{'\\gamma'}</M> 从来不只是「调收敛速度」——
            <em>你是在调这个智能体的性格</em>。
          </Callout>

          <PredictNumber
            id="ch3-gamma-flip"
            question={
              <>
                把 <M>{'\\gamma'}</M> 一路调到 <M>{'0.1'}</M>（极度短视）。
                与 <M>{'\\gamma = 0.9'}</M> 相比，25 个格子里会有多少个箭头翻转？
                先猜，再拖动滑块验证。
              </>
            }
            min={0}
            max={25}
            step={1}
            truth={flipsAt01}
            tolerance={3}
            unit=" 个"
            explain={
              <>
                <p>
                  多数人会低估这个数字。<strong><M>{'\\gamma'}</M> 越小，
                  离目标越远的格子越「看不见」目标</strong>，
                  它们的最优动作退化成「就近避开惩罚」，与远视时的路线大相径庭。
                </p>
                <p>
                  这也解释了一个实践中常见的现象：
                  <em>把 <M>{'\\gamma'}</M> 调小往往能让训练更快更稳，但学出来的策略是「近视」的</em>，
                  在需要长程规划的任务上会明显变笨。这是一个货真价实的权衡，
                  不是可以随手拍脑袋定的数。
                </p>
              </>
            }
          />
        </Beat>

        <Beat id="g3">
          <Callout tone="intuition" title="第 3 章的一句话">
            贝尔曼最优公式 = 贝尔曼公式 + 一个 <M>{'\\max'}</M>。
            这个 max 消掉了策略、带来了非线性、也带来了「最优策略可以是确定性的」这个礼物。
            压缩映射保证它有唯一解，而<strong>「反复迭代直到不动」这句话本身，
            就已经是一个能跑的算法了</strong>。
          </Callout>
          <p>
            但你可能已经注意到了：第 3 幕那段代码里，
            <code>P</code> 和 <code>R</code> 是<strong>直接给定的</strong>。
            也就是说，我们从头到尾都假设自己拿到了这个世界的完整说明书。
          </p>
          <p>
            下一章会把这个算法讲透，并且揭示值迭代和策略迭代其实是同一件事。
            <em>而再往后，我们必须面对那个真正的问题：如果没有说明书呢？</em>
          </p>
        </Beat>
      </Act>
    </div>
  )
}
