import { useMemo, useState } from 'react'
import { buildGridMDP, classicGrid, corridorGrid } from '../core/mdp'
import { argmaxActions, greedyFromQ, qFromV, uniformPolicy, type Policy } from '../core/policy'
import {
  normInf,
  policyEvaluationDirect,
  truncatedPolicyIterationTrace,
  valueIterationSolve,
  valueIterationTrace,
} from '../core/solvers'
import { useBus } from '../highlight/bus'
import { Act, Beat } from '../narrative/Act'
import { ChapterGlance, ChapterHero, type Glance } from '../narrative/ChapterShell'
import { PredictChoice, PredictNumber } from '../narrative/Predict'
import { GridWorld } from '../viz/GridWorld'
import { LineChart } from '../viz/LineChart'
import { Scrubber } from '../ui/Scrubber'
import { Callout, Code, Details, M, MB, Panel, Seg, Slider, Stat } from '../ui/prims'
import { fmt, useColors } from '../theme'

const GLANCE: Glance = {
  formula: String.raw`\underbrace{v_{k+1} = \max_\pi \left( r_\pi + \gamma P_\pi v_k \right)}_{\text{值迭代}}
  \qquad
  \underbrace{\begin{cases} v_{\pi_k} = r_{\pi_k} + \gamma P_{\pi_k} v_{\pi_k} \\[2pt] \pi_{k+1} = \arg\max_\pi (r_\pi + \gamma P_\pi v_{\pi_k}) \end{cases}}_{\text{策略迭代}}`,
  formulaNote:
    '两个算法看起来完全不同，其实是同一条谱系的两端：中间那一步「策略评估要迭代几次」，取 1 就是值迭代，取 ∞ 就是策略迭代。',
  takeaways: [
    <>
      值迭代就是把贝尔曼最优公式<strong>当成赋值语句反复执行</strong>。压缩映射保证它一定收敛，
      而且误差按 <M>{'\\gamma^k'}</M> 衰减。
    </>,
    <>
      值迭代里的 <M>{'v_k'}</M> <strong>不是任何策略的状态价值</strong>，它只是一个中间量。
      只有收敛后的 <M>{'v^*'}</M> 才有价值的含义。
    </>,
    <>
      策略迭代每轮都要把 <M>{'v_{\\pi_k}'}</M> 算准，靠的是<strong>策略改进定理</strong>：
      贪心一下，新策略在每个状态都不会更差。
    </>,
    <>
      截断策略迭代把两者连成一条线。实践中取 3~10 次评估往往最划算 ——
      <strong>总轮数明显少于值迭代，每轮又比精确求解便宜得多</strong>。
    </>,
  ],
  traps: [
    <>
      把值迭代的 <M>{'v_k'}</M> 读成「第 k 个策略的价值」。它不是。中途的 <M>{'v_k'}</M>
      可能不对应任何策略。
    </>,
    <>
      以为策略迭代的内层要迭代到「完全收敛」才合法。其实截断在任何有限步都仍然收敛，
      只是路径不同。
    </>,
    <>
      这一章的两个算法都要用 <M>{'p(s\'|s,a)'}</M> 和 <M>{'r(s,a)'}</M>。
      它们是 <strong>model-based</strong> 的 —— 这正是第 5 章要造反的地方。
    </>,
  ],
}

/** 截断到 n 步评估时，要多少轮才第一次达到最优策略 */
function roundsToOptimal(
  trace: ReturnType<typeof truncatedPolicyIterationTrace>,
  target: number[],
): number {
  for (let i = 0; i < trace.length; i++) {
    const a = argmaxActions(trace[i].policy)
    if (a.every((x, j) => x === target[j])) return i + 1
  }
  return trace.length
}

export function Chapter4() {
  const C = useColors()
  const [gamma, setGamma] = useState(0.9)
  const [k, setK] = useState(0)
  const [pk, setPk] = useState(0)
  const [evalSteps, setEvalSteps] = useState(3)
  const [view, setView] = useState<'vi' | 'pi'>('vi')

  const focus = useBus((s) => s.focus)
  const setFocus = useBus((s) => s.setFocus)

  const mdp = useMemo(() => buildGridMDP(classicGrid()), [])

  const star = useMemo(() => valueIterationSolve(mdp, gamma), [mdp, gamma])
  const targetActions = useMemo(() => argmaxActions(star.policy), [star])

  /* 值迭代的完整轨迹 */
  const vi = useMemo(() => valueIterationTrace(mdp, gamma, 60), [mdp, gamma])
  const kk = Math.min(k, vi.length - 1)
  const viErr = useMemo(() => vi.map((t) => normInf(t.v, star.v)), [vi, star])

  /** v_k 到底是不是某个策略的价值：拿 v_k 导出的贪心策略去精确评估，看差多少 */
  const notAPolicyValue = useMemo(() => {
    const pol = greedyFromQ(qFromV(mdp, gamma, vi[kk].v)).policy
    const vReal = policyEvaluationDirect(mdp, pol, gamma)
    return normInf(vi[kk].v, vReal)
  }, [mdp, gamma, vi, kk])

  /* 策略迭代（内层精确求解） */
  const piTrace = useMemo(
    () => truncatedPolicyIterationTrace(mdp, gamma, Infinity, 12, uniformPolicy(mdp.nS)),
    [mdp, gamma],
  )
  const pkk = Math.min(pk, piTrace.length - 1)
  const piErr = useMemo(() => piTrace.map((t) => normInf(t.vEval, star.v)), [piTrace, star])

  /*
   * 截断谱系用的是蛇形走廊，不是 5×5 经典网格。
   * 经典网格太通透，价值信息几步就传遍全场，内层评估准不准几乎不影响外层轮数，
   * 谱系曲线会平成一条直线，看不出截断的意义。走廊把唯一通路拉长，差别才显形。
   */
  const cor = useMemo(() => buildGridMDP(corridorGrid(9, 9)), [])
  const corStar = useMemo(() => valueIterationSolve(cor, gamma), [cor, gamma])
  const corTarget = useMemo(() => argmaxActions(corStar.policy), [corStar])

  const truncated = useMemo(
    () => truncatedPolicyIterationTrace(cor, gamma, evalSteps, 60, uniformPolicy(cor.nS)),
    [cor, gamma, evalSteps],
  )
  const truncErr = useMemo(
    () => truncated.map((t) => normInf(t.vEval, corStar.v)),
    [truncated, corStar],
  )
  const corViErr = useMemo(
    () => valueIterationTrace(cor, gamma, 60).map((t) => normInf(t.v, corStar.v)),
    [cor, gamma, corStar],
  )
  const corPiTrace = useMemo(
    () => truncatedPolicyIterationTrace(cor, gamma, Infinity, 60, uniformPolicy(cor.nS)),
    [cor, gamma],
  )
  const corPiErr = useMemo(
    () => corPiTrace.map((t) => normInf(t.vEval, corStar.v)),
    [corPiTrace, corStar],
  )

  const SPECTRUM = useMemo(() => [1, 2, 3, 5, 8, 13, 21, 34], [])
  const spectrumRounds = useMemo(
    () =>
      SPECTRUM.map((n) =>
        roundsToOptimal(
          truncatedPolicyIterationTrace(cor, gamma, n, 120, uniformPolicy(cor.nS)),
          corTarget,
        ),
      ),
    [SPECTRUM, cor, gamma, corTarget],
  )
  const corPiRounds = roundsToOptimal(corPiTrace, corTarget)
  const corViRounds = useMemo(
    () => roundsToOptimal(truncatedPolicyIterationTrace(cor, gamma, 1, 120, uniformPolicy(cor.nS)), corTarget),
    [cor, gamma, corTarget],
  )
  const piRounds = roundsToOptimal(piTrace, targetActions)
  const viRounds = useMemo(() => {
    for (let i = 0; i < vi.length; i++) {
      const a = argmaxActions(vi[i].policy)
      if (a.every((x, j) => x === targetActions[j])) return i + 1
    }
    return vi.length
  }, [vi, targetActions])

  /* 策略改进定理的现场证据：改进一次，有几个格子的价值真的涨了 */
  const improveDemo = useMemo(() => {
    const pi0: Policy = uniformPolicy(mdp.nS)
    const v0 = policyEvaluationDirect(mdp, pi0, gamma)
    const pi1 = greedyFromQ(qFromV(mdp, gamma, v0)).policy
    const v1 = policyEvaluationDirect(mdp, pi1, gamma)
    const up = v1.filter((x, i) => x > v0[i] + 1e-9).length
    const down = v1.filter((x, i) => x < v0[i] - 1e-9).length
    return { pi0, v0, pi1, v1, up, down }
  }, [mdp, gamma])

  return (
    <div>
      <ChapterHero
        n={4}
        hook="方程有了，但多了 max 之后它是非线性的，没法一步解出来。怎么办？"
        lead={
          <>
            <p>
              上一章证明了贝尔曼最优公式有且只有一个解，但<strong>没有告诉你怎么找到它</strong>。
              这一章补上这一刀：既然它是压缩映射的不动点，那就<em>反复把它当成赋值语句执行</em>，
              让迭代自己走到不动点上去。
            </p>
            <p>
              然后你会遇到第二个算法 —— 策略迭代。它看起来完全是另一套：先评估，再改进，两层循环。
              这一章最有价值的收获，是发现<strong>这两个算法其实是同一个算法</strong>，
              中间隔着一个可以连续调节的旋钮。
            </p>
          </>
        }
        gains={[
          '把贝尔曼最优公式变成一个真的能跑的算法',
          '看清值迭代中间的 v_k 为什么不是任何策略的价值',
          '理解策略改进定理：贪心一下，为什么每个状态都不会变差',
          '亲手把「值迭代」和「策略迭代」用一个滑块连起来',
          '知道在什么情况下该截断到几步',
        ]}
      />

      <ChapterGlance g={GLANCE} />

      {/* ───────────────────── 第 1 幕：值迭代 ───────────────────── */}
      <Act
        id="a1"
        no="第 1 幕"
        title="把方程当成赋值语句"
        goal="贝尔曼最优公式是一个不动点方程；反复代入自己，就是值迭代。"
        minutes={9}
        points={[
          <>
            值迭代：<M>{'v_{k+1} = \\max_a\\, q_k(s,a)'}</M>，其中{' '}
            <M>{'q_k(s,a) = r(s,a) + \\gamma\\sum_{s\'}p(s\'|s,a)v_k(s\')'}</M>。
          </>,
          <>
            它由两步交替组成：<strong>策略更新</strong>（对 q 取贪心）和
            <strong>价值更新</strong>（把最大的 q 抄进 v）。
          </>,
          <>
            收敛速度由 <M>{'\\gamma'}</M> 决定：误差不超过{' '}
            <M>{'\\gamma^k\\|v_0 - v^*\\|_\\infty'}</M>。γ 越大收敛越慢。
          </>,
        ]}
        stage={() => (
          <div className="space-y-4">
            <Panel
              title={`第 ${kk} 次迭代`}
              right={
                <span className="font-mono text-[11.5px]" style={{ color: C.accent }}>
                  ‖v_k − v*‖∞ = {fmt(viErr[kk], 4)}
                </span>
              }
            >
              <GridWorld
                mdp={mdp}
                v={vi[kk].v}
                policy={vi[kk].policy}
                ties={vi[kk].ties}
                cell={56}
                onCellClick={(x) => setFocus(x === focus ? null : x)}
              />
              <div className="mt-3">
                <Scrubber k={kk} setK={setK} max={vi.length - 1} />
              </div>
            </Panel>

            <Panel title="v_k 是某个策略的价值吗">
              <div className="grid grid-cols-2 gap-3">
                <Stat
                  label={<>v_k 与「对 v_k 贪心得到的策略」的真实价值之差</>}
                  value={fmt(notAPolicyValue, 3)}
                  color={notAPolicyValue > 1e-6 ? C.danger : C.value}
                />
                <Stat label="离最优还差" value={fmt(viErr[kk], 3)} color={C.accent} />
              </div>
              <p className="mt-3 text-[11.5px] leading-relaxed text-faint">
                只要这个数不是 0，就说明此刻的 <M>{'v_k'}</M>
                「不属于任何人」—— 它既不是初始策略的价值，也不是当前贪心策略的价值。
                拖到最后，它才变成 0。
              </p>
            </Panel>

            <Panel title="误差衰减">
              <LineChart
                height={168}
                logY
                marker={kk}
                xLabel="迭代次数 k"
                series={[
                  { name: '‖v_k − v*‖∞', color: C.accent, data: viErr },
                  {
                    name: 'γᵏ 上界',
                    color: C.gamma,
                    data: viErr.map((_, i) => (viErr[0] || 1) * Math.pow(gamma, i)),
                    dashed: true,
                  },
                ]}
              />
            </Panel>
          </div>
        )}
      >
        <Beat id="b1" keep>
          <p>
            上一章的结论是：<M>{'v^* = f(v^*)'}</M>，而 <M>{'f'}</M> 是一个 <M>{'\\gamma'}</M>
            -压缩映射。压缩映射定理不只保证解存在唯一，它还<strong>附赠了一个算法</strong>：
            从任意 <M>{'v_0'}</M> 出发，反复作用 <M>{'f'}</M>，一定收敛到 <M>{'v^*'}</M>。
          </p>
          <MB>{'v_{k+1} = f(v_k) = \\max_{\\pi}\\left( r_\\pi + \\gamma P_\\pi v_k \\right)'}</MB>
          <p>就这么简单。这个算法叫值迭代。</p>
        </Beat>

        <Beat id="b2" keep>
          <p>展开成逐元素的形式，每一轮其实做了两件事：</p>
          <MB>
            {String.raw`\begin{aligned}
            \text{① 策略更新：}\quad & \pi_{k+1}(s) = \arg\max_a\, q_k(s,a) \\[4pt]
            \text{② 价值更新：}\quad & v_{k+1}(s) = \max_a\, q_k(s,a)
            \end{aligned}`}
          </MB>
          <p>
            其中 <M>{"q_k(s,a) = r(s,a) + \\gamma\\sum_{s'}p(s'|s,a)\\,v_k(s')"}</M>。
            拖右边的时间轴，你会看到箭头（策略）和底色（价值）在同步变化 ——
            它们本来就是同一次计算的两个产物。
          </p>
        </Beat>

        <Beat id="b3" keep>
          <PredictChoice
            id="ch4-vk"
            question={
              <>
                值迭代跑到第 5 步时，<M>{'v_5'}</M> 是什么？
              </>
            }
            options={[
              { id: 'a', label: '第 5 个策略 π₅ 的状态价值' },
              { id: 'b', label: '初始策略在 5 步之内的期望回报' },
              { id: 'c', label: '什么都不是，只是一个中间数值' },
            ]}
            answer="c"
            explain={
              <>
                <p>
                  这是值迭代最容易被误解的地方。<M>{'v_k'}</M> 只是迭代过程中的一个向量，
                  它<strong>不满足任何策略的贝尔曼公式</strong>。
                </p>
                <p>
                  右边那个数字就是证据：把 <M>{'v_k'}</M> 贪心出来的策略拿去精确评估，
                  得到的价值和 <M>{'v_k'}</M> 本身相差 <strong>{fmt(notAPolicyValue, 3)}</strong>。
                  只有在收敛之后，这个差才会归零。
                </p>
              </>
            }
          />
        </Beat>

        <Beat id="b4">
          <p>
            这个区分不是咬文嚼字。它直接决定了你能不能提前停：因为 <M>{'v_k'}</M>
            没有语义，所以你不能在中途说「我现在的策略有 <M>{'v_k'}</M> 这么好」。
            但你可以说「我离最优不超过 <M>{'\\gamma^k \\|v_0 - v^*\\|_\\infty'}</M>」 ——
            右下角那张对数坐标图画的就是这条上界。
          </p>
        </Beat>

        <Beat id="b5">
          <Details summary="展开：为什么误差按 γᵏ 衰减">
            <p>
              由压缩性，<M>{'\\|v_{k+1} - v^*\\| = \\|f(v_k) - f(v^*)\\| \\le \\gamma\\|v_k - v^*\\|'}</M>。
              递推 <M>{'k'}</M> 次即得
            </p>
            <MB>{'\\|v_k - v^*\\|_\\infty \\le \\gamma^k \\|v_0 - v^*\\|_\\infty'}</MB>
            <p>
              取对数后是一条直线，斜率是 <M>{'\\ln\\gamma'}</M>。所以在对数坐标下，
              实际误差曲线应该<strong>贴着或压在</strong>那条虚线之下 —— 右边的图正是如此。
              γ 越接近 1，这条线越平，需要的迭代次数越多。
            </p>
          </Details>
        </Beat>
      </Act>

      {/* ───────────────────── 第 2 幕：策略迭代 ───────────────────── */}
      <Act
        id="a2"
        no="第 2 幕"
        title="另一条路：评估，然后贪心"
        goal="策略迭代的两层循环，以及支撑它的策略改进定理。"
        minutes={11}
        points={[
          <>
            策略迭代 = <strong>策略评估</strong>（解出 <M>{'v_{\\pi_k}'}</M>）+{' '}
            <strong>策略改进</strong>（对 <M>{'q_{\\pi_k}'}</M> 贪心）。
          </>,
          <>
            策略改进定理：<M>{'\\pi_{k+1} \\ge \\pi_k'}</M> 在<strong>每一个状态</strong>上成立，
            不只是平均意义上更好。
          </>,
          <>
            有限 MDP 里确定性策略只有有限个，而策略序列单调不减，所以
            <strong>有限步内必然停在最优策略上</strong>。
          </>,
        ]}
        stage={() => (
          <div className="space-y-4">
            <Panel
              title={`策略迭代 · 第 ${pkk + 1} 轮`}
              right={
                <span className="font-mono text-[11.5px]" style={{ color: C.accent }}>
                  ‖v − v*‖∞ = {fmt(piErr[pkk], 4)}
                </span>
              }
            >
              <GridWorld
                mdp={mdp}
                v={piTrace[pkk].vEval}
                policy={piTrace[pkk].policy}
                ties={piTrace[pkk].ties}
                cell={56}
              />
              <div className="mt-3">
                <Scrubber k={pkk} setK={setPk} max={piTrace.length - 1} label="轮次" fps={2} />
              </div>
            </Panel>

            <Panel title="改进一次的效果">
              <div className="grid grid-cols-3 gap-2.5">
                <Stat label="价值上升的格子" value={improveDemo.up} color={C.value} />
                <Stat label="价值下降的格子" value={improveDemo.down} color={C.danger} />
                <Stat
                  label="策略迭代需要的轮数"
                  value={piRounds}
                  color={C.accent}
                  hint="第一次达到最优策略"
                />
              </div>
              <p className="mt-3 text-[11.5px] leading-relaxed text-faint">
                「下降的格子 = 0」不是运气好，是定理保证的。哪怕只有一个格子下降，
                策略改进定理就被推翻了。
              </p>
            </Panel>
          </div>
        )}
      >
        <Beat id="c1" keep>
          <p>
            换一个思路。既然第 2 章已经会「给定策略求价值」了，第 3 章又知道「对 q 贪心能变好」，
            那就把这两件事交替做：
          </p>
          <MB>
            {String.raw`\pi_0 \xrightarrow{\text{评估}} v_{\pi_0} \xrightarrow{\text{改进}} \pi_1
            \xrightarrow{\text{评估}} v_{\pi_1} \xrightarrow{\text{改进}} \pi_2 \to \cdots`}
          </MB>
          <p>这就是策略迭代。它的每一轮都要把内层的贝尔曼公式解到底。</p>
        </Beat>

        <Beat id="c2" keep>
          <PredictNumber
            id="ch4-improve"
            question={
              <>
                从均匀随机策略出发，做<strong>一次</strong>「评估 + 贪心」。
                在 25 个格子里，价值<strong>下降</strong>的会有几个？
              </>
            }
            min={0}
            max={25}
            step={1}
            truth={improveDemo.down}
            tolerance={0}
            explain={
              <>
                <p>
                  答案是 <strong>0</strong>，而且不是碰巧 —— 这就是<strong>策略改进定理</strong>：
                </p>
                <MB>{'\\pi_{k+1} = \\arg\\max_\\pi (r_\\pi + \\gamma P_\\pi v_{\\pi_k}) \\;\\Longrightarrow\\; v_{\\pi_{k+1}} \\ge v_{\\pi_k}'}</MB>
                <p>
                  注意这个 <M>{'\\ge'}</M> 是<strong>逐状态</strong>的。同时有{' '}
                  <strong>{improveDemo.up}</strong> 个格子的价值严格上升了。
                </p>
              </>
            }
          />
        </Beat>

        <Beat id="c3">
          <Details summary="展开：策略改进定理的证明（三行）">
            <p>
              记 <M>{'\\pi\' = \\arg\\max_\\pi (r_\\pi + \\gamma P_\\pi v_\\pi)'}</M>。由 argmax 的定义，
            </p>
            <MB>{'r_{\\pi\'} + \\gamma P_{\\pi\'} v_\\pi \\;\\ge\\; r_\\pi + \\gamma P_\\pi v_\\pi = v_\\pi'}</MB>
            <p>
              把这个不等式反复代入自己（注意 <M>{'P_{\\pi\'}'}</M> 每个元素非负，
              所以左乘它保持不等号方向）：
            </p>
            <MB>
              {String.raw`v_\pi \le r_{\pi'} + \gamma P_{\pi'} v_\pi
              \le r_{\pi'} + \gamma P_{\pi'}\left(r_{\pi'} + \gamma P_{\pi'} v_\pi\right) \le \cdots`}
            </MB>
            <p>
              取极限，右端收敛到 <M>{"(I-\\gamma P_{\\pi'})^{-1} r_{\\pi'} = v_{\\pi'}"}</M>，于是
              <M>{"v_\\pi \\le v_{\\pi'}"}</M>。证毕。
            </p>
            <p>
              这里唯一用到的性质是 <strong>P 非负</strong>和 <M>{'\\gamma < 1'}</M>。
              前者是概率矩阵的必然，后者是折扣的功劳 —— 又一次。
            </p>
          </Details>
        </Beat>

        <Beat id="c4">
          <Callout tone="insight" title="为什么它一定会停">
            <p>
              有限 MDP 里确定性策略只有 <M>{'|\\mathcal{A}|^{|\\mathcal{S}|}'}</M> 个 —— 有限多。
              而策略序列单调不减，一旦某轮策略不再变化，就说明它满足了贝尔曼最优公式。
              所以策略迭代<strong>在有限步内精确终止</strong>，
              这一点比值迭代（只在极限意义下收敛）更强。
            </p>
          </Callout>
        </Beat>
      </Act>

      {/* ───────────────────── 第 3 幕：谱系 ───────────────────── */}
      <Act
        id="a3"
        no="第 3 幕"
        title="一个滑块，两个算法"
        goal="截断策略迭代把值迭代和策略迭代连成一条连续的谱系。"
        minutes={10}
        points={[
          <>
            内层评估迭代 <M>{'j'}</M> 次：<M>{'j=1'}</M> 就是值迭代，
            <M>{'j=\\infty'}</M> 就是策略迭代。中间全是合法算法。
          </>,
          <>
            <M>{'j'}</M> 越大，<strong>外层轮数越少</strong>，但<strong>每轮越贵</strong>。
            最划算的点通常在 3~10 之间。
          </>,
          <>
            这解释了为什么教科书上的两个算法长得完全不一样却收敛到同一个答案 ——
            它们本来就是同一件事。
          </>,
        ]}
        stage={() => (
          <div className="space-y-4">
            <Panel title={`蛇形走廊 · 第 8 轮时，内层评估 ${evalSteps} 次`}>
              <GridWorld
                mdp={cor}
                v={truncated[Math.min(7, truncated.length - 1)].vEval}
                policy={truncated[Math.min(7, truncated.length - 1)].policy}
                cell={32}
                quiet
              />
              <div className="mt-3">
                <Slider
                  label={
                    <>
                      内层评估步数 <M>{'j'}</M>
                    </>
                  }
                  value={evalSteps}
                  min={1}
                  max={30}
                  step={1}
                  onChange={setEvalSteps}
                  format={(v) => String(v)}
                  hint={
                    evalSteps === 1
                      ? '此刻它就是值迭代'
                      : evalSteps >= 25
                        ? '此刻它几乎就是策略迭代'
                        : '中间地带：截断策略迭代'
                  }
                />
              </div>
            </Panel>

            <Panel title="外层轮数 vs 内层步数">
              <LineChart
                height={180}
                xLabel="内层评估步数 j（1, 2, 3, 5, 8, 13, 21, 34）"
                series={[
                  { name: '达到最优策略所需外层轮数', color: C.accent, data: spectrumRounds },
                ]}
              />
              <div className="mt-3 grid grid-cols-2 gap-2.5">
                <Stat label="值迭代（j=1）需要" value={`${corViRounds} 轮`} color={C.gamma} />
                <Stat label="策略迭代（j=∞）需要" value={`${corPiRounds} 轮`} color={C.policy} />
              </div>
            </Panel>

            <Panel title="收敛曲线对照">
              <LineChart
                height={168}
                logY
                xLabel="外层轮数"
                series={[
                  { name: `截断 j=${evalSteps}`, color: C.accent, data: truncErr.slice(0, 40) },
                  { name: '值迭代', color: C.gamma, data: corViErr.slice(0, 40), dashed: true },
                  { name: '策略迭代', color: C.policy, data: corPiErr.slice(0, 40) },
                ]}
              />
            </Panel>
          </div>
        )}
      >
        <Beat id="d1" keep>
          <p>
            现在把两个算法并排放在一起看。策略迭代的内层是在解{' '}
            <M>{'v_{\\pi_k} = r_{\\pi_k} + \\gamma P_{\\pi_k} v_{\\pi_k}'}</M>，
            而解它的办法通常也是迭代：
          </p>
          <MB>{'v^{(j+1)} = r_{\\pi_k} + \\gamma P_{\\pi_k} v^{(j)}, \\qquad j = 0,1,2,\\dots'}</MB>
          <p>
            问题来了：<strong>这个内层非要迭代到收敛不可吗？</strong>
          </p>
        </Beat>

        <Beat id="d2" keep>
          <p>
            不必。把它截断在第 <M>{'j'}</M> 步就停，得到的算法叫<strong>截断策略迭代</strong>。
            而当你把 <M>{'j'}</M> 一路调下去：
          </p>
          <MB>
            {String.raw`\underbrace{j = 1}_{\text{值迭代}} \;\longleftrightarrow\;
            \underbrace{j = 2,3,\dots}_{\text{截断策略迭代}} \;\longleftrightarrow\;
            \underbrace{j = \infty}_{\text{策略迭代}}`}
          </MB>
          <p>
            两个「不同的算法」，其实是同一条谱系的两个端点。拖右边那个滑块，
            亲眼看着一个算法变成另一个。
          </p>
        </Beat>

        <Beat id="d2b">
          <Callout tone="insight" title="这一幕换了个世界：蛇形走廊">
            <p>
              右边的舞台不再是那个 5×5 网格，而是一条 9×9 的蛇形走廊。
              换世界不是为了好看 —— 是因为<strong>经典网格上这个实验做不出效果</strong>。
            </p>
            <p>
              经典网格太通透：目标就在正中央，任何一格离它都只有几步，
              价值信息两三轮就传遍全场。这时候内层评估准不准根本不重要，
              曲线会平成一条直线。
            </p>
            <p>
              走廊把唯一通路折成几十格长，价值信息必须<strong>一跳一跳地爬回起点</strong>。
              内层每多迭代一次，信息就多爬一段 —— 截断的代价这才显形。
            </p>
          </Callout>
        </Beat>

        <Beat id="d3" keep>
          <PredictChoice
            id="ch4-trunc"
            question="把内层评估步数从 1 加到 30，外层需要的轮数会怎么变？"
            options={[
              { id: 'a', label: '单调下降，一直降到 1 轮' },
              { id: 'b', label: '先快速下降，然后基本不动' },
              { id: 'c', label: '先降后升，中间有个最优点' },
            ]}
            answer="b"
            explain={
              <>
                <p>
                  外层轮数从 <strong>{spectrumRounds[0]}</strong> 掉到{' '}
                  <strong>{spectrumRounds[2]}</strong>，之后就几乎躺平在{' '}
                  <strong>{spectrumRounds[spectrumRounds.length - 1]}</strong> 附近 ——
                  因为内层再精确，也不可能让外层少于策略迭代的 {corPiRounds} 轮。
                  那是这条谱系的地板。
                </p>
                <p>
                  这条「先陡后平」的曲线正是截断的价值所在：
                  <strong>前几步评估贡献了绝大部分收益，后面全是边际递减。</strong>
                  实际工程里取 <M>{'j \\in [3, 10]'}</M> 通常最划算。
                </p>
              </>
            }
          />
        </Beat>

        <Beat id="d4">
          <Callout tone="trap" title="别只看轮数">
            <p>
              「策略迭代轮数少」不等于「策略迭代更快」。它每一轮要解一个{' '}
              <M>{'|\\mathcal{S}| \\times |\\mathcal{S}|'}</M> 的线性方程组，
              代价是 <M>{'O(|\\mathcal{S}|^3)'}</M>；值迭代每轮只要 <M>{'O(|\\mathcal{S}|^2|\\mathcal{A}|)'}</M>。
              该比的是<strong>轮数 × 每轮代价</strong>。
            </p>
          </Callout>
        </Beat>
      </Act>

      {/* ───────────────────── 第 4 幕：代码 ───────────────────── */}
      <Act
        id="a4"
        no="第 4 幕"
        title="写成代码，然后看清它的死穴"
        goal="三十行实现两个算法，同时暴露它们共同的致命前提。"
        minutes={7}
        points={[
          <>
            两个算法的代码只差一个 <code>for</code> 循环的次数。
          </>,
          <>
            它们都在<strong>读 P 和 R</strong> —— 这就是 model-based 的定义，
            也是下一章要拆掉的东西。
          </>,
        ]}
        stage={() => (
          <div className="space-y-4">
            <Panel title="最终结果" right={<Seg size="sm" value={view} onChange={setView} options={[{ value: 'vi', label: '值迭代' }, { value: 'pi', label: '策略迭代' }]} />}>
              <GridWorld
                mdp={mdp}
                v={view === 'vi' ? star.v : piTrace[piTrace.length - 1].vEval}
                policy={view === 'vi' ? star.policy : piTrace[piTrace.length - 1].policy}
                ties={star.ties}
                cell={56}
              />
              <p className="mt-3 text-center text-[11.5px] text-faint">
                两条完全不同的路径，同一个终点。
              </p>
            </Panel>
            <Panel title="折扣因子">
              <Slider
                label={<M>{'\\gamma'}</M>}
                value={gamma}
                min={0.1}
                max={0.99}
                step={0.01}
                onChange={setGamma}
                accent={C.gamma}
                hint={`值迭代 ${viRounds} 轮 · 策略迭代 ${piRounds} 轮`}
              />
            </Panel>
          </div>
        )}
      >
        <Beat id="e1" keep>
          <Code
            code={`def value_iteration(P, R, gamma, tol=1e-10):
    """P[s][a][s'], R[s][a] —— 注意：这两样都必须事先知道"""
    nS, nA = len(R), len(R[0])
    v = [0.0] * nS
    while True:
        q = [[R[s][a] + gamma * sum(P[s][a][sp] * v[sp] for sp in range(nS))
              for a in range(nA)] for s in range(nS)]
        v_new = [max(q[s]) for s in range(nS)]          # 价值更新
        if max(abs(a - b) for a, b in zip(v, v_new)) < tol:
            break
        v = v_new
    pi = [max(range(nA), key=lambda a: q[s][a]) for s in range(nS)]  # 策略更新
    return v, pi`}
          />
        </Beat>

        <Beat id="e2" keep>
          <Code
            code={`def policy_iteration(P, R, gamma, j=None):
    """j = 1   -> 值迭代
       j = None -> 策略迭代（内层解到收敛）
       其它     -> 截断策略迭代"""
    nS, nA = len(R), len(R[0])
    pi = [0] * nS
    v = [0.0] * nS
    while True:
        # ① 策略评估：迭代 j 次（或直到收敛）
        for _ in range(j or 10_000):
            v = [R[s][pi[s]] + gamma * sum(P[s][pi[s]][sp] * v[sp] for sp in range(nS))
                 for s in range(nS)]
        # ② 策略改进：对 q 贪心
        q = [[R[s][a] + gamma * sum(P[s][a][sp] * v[sp] for sp in range(nS))
              for a in range(nA)] for s in range(nS)]
        pi_new = [max(range(nA), key=lambda a: q[s][a]) for s in range(nS)]
        if pi_new == pi:
            return v, pi
        pi = pi_new`}
          />
        </Beat>

        <Beat id="e3" keep>
          <Callout tone="trap" title="两段代码里最刺眼的东西">
            <p>
              把上面两个函数的签名放在一起看：<code>P</code> 和 <code>R</code>。
              整章的算法都建立在「我知道这个世界的转移概率和奖励函数」之上。
            </p>
            <p>
              可现实里 —— 机器人不知道自己电机的精确动力学，
              下棋程序不知道对手的策略分布，推荐系统不知道用户点击的真实概率。
              <strong>模型几乎从来不给你。</strong>
            </p>
            <p>
              这一章解决了「有模型怎么办」。下一章要面对的问题是：
              <em>如果 p 和 r 这两个字母根本写不出来呢？</em>
            </p>
          </Callout>
        </Beat>
      </Act>
    </div>
  )
}
