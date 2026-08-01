import { useMemo, useState } from 'react'
import { buildGridMDP, classicGrid } from '../core/mdp'
import { argmaxActions, uniformPolicy } from '../core/policy'
import { policyEvaluationDirect, valueIterationSolve } from '../core/solvers'
import { cliffEnv, cliffMDP, mdpToEnv } from '../core/env'
import { greedyPath, mcPrediction, nStepSarsa, smooth, tdControl, tdPrediction } from '../core/learn'
import { Act, Beat } from '../narrative/Act'
import { ChapterGlance, ChapterHero, type Glance } from '../narrative/ChapterShell'
import { PredictChoice } from '../narrative/Predict'
import { LiveFormula } from '../formula/LiveFormula'
import { useFormulaCtx } from '../formula/core'
import { tdUpdate } from '../formula/bellman'
import { GridWorld } from '../viz/GridWorld'
import { LineChart } from '../viz/LineChart'
import { Scrubber } from '../ui/Scrubber'
import { Callout, Code, Details, M, MB, Panel, Seg, Slider, Stat } from '../ui/prims'
import { fmt, useColors } from '../theme'

const F_TD = tdUpdate()

const GLANCE: Glance = {
  formula: String.raw`v(s_t) \leftarrow v(s_t) + \alpha\Big[\underbrace{r_{t+1} + \gamma v(s_{t+1})}_{\text{TD 目标}} - v(s_t)\Big]`,
  formulaNote:
    '把第 6 章的 RM 算法套在贝尔曼公式上，一行就得到 TD。它不等回合结束、只用一次转移，代价是用估计去更新估计（自举）。',
  takeaways: [
    <>
      TD 与 MC 的分工：<strong>MC 无偏但方差大</strong>（用真实回报），
      <strong>TD 有偏但方差小</strong>（用一步真实 + 估计）。n 步 TD 是它们之间的连续谱。
    </>,
    <>
      Sarsa 与 Q-learning 只差一个符号：目标里用 <M>{'q(s\',a\')'}</M> 还是{' '}
      <M>{'\\max_a q(s\',a)'}</M>。前者 on-policy，后者 off-policy。
    </>,
    <>
      off-policy 的意义：<strong>行为策略负责探索，目标策略负责最优</strong>。
      Q-learning 因此能一边乱走一边学到最优策略。
    </>,
    <>
      悬崖世界里两者<strong>学到的答案都对</strong>，但在线表现相反 ——
      Q-learning 学最优路径却常掉崖，Sarsa 学到一条更保守但实际得分更高的路。
    </>,
  ],
  traps: [
    <>
      把 TD 误差 <M>{'\\delta_t'}</M> 说成「预测误差」就完了。它更准确的名字是
      <strong>贝尔曼误差的采样版本</strong> —— 它衡量的是当前估计违反贝尔曼公式的程度。
    </>,
    <>
      以为 Q-learning「更好」。它学到的是最优策略，但如果你关心
      <strong>学习过程中的实际收益</strong>，Sarsa 常常赢。
    </>,
    <>
      忘了 TD 的收敛也要满足第 6 章的步长条件。常数 <M>{'\\alpha'}</M> 只能进入邻域。
    </>,
  ],
}

export function Chapter7() {
  const C = useColors()
  const [gamma] = useState(0.9)
  const [alpha, setAlpha] = useState(0.1)
  const [eps, setEps] = useState(0.1)
  const [nStep, setNStep] = useState(1)
  const [episodes, setEpisodes] = useState(500)
  const [k, setK] = useState(0)
  const [duel, setDuel] = useState<'sarsa' | 'qlearning'>('sarsa')

  /* ── 预测：TD vs MC ── */
  const mdp = useMemo(() => buildGridMDP(classicGrid()), [])
  const pi = useMemo(() => uniformPolicy(mdp.nS), [mdp.nS])
  const vTrue = useMemo(() => policyEvaluationDirect(mdp, pi, gamma), [mdp, pi, gamma])
  const env = useMemo(() => mdpToEnv(mdp, { start: 0, horizon: 40 }), [mdp])

  const td = useMemo(
    () => tdPrediction(env, pi, { gamma, alpha, episodes: 400, seed: 5 }, vTrue),
    [env, pi, gamma, alpha, vTrue],
  )
  const mc = useMemo(
    () => mcPrediction(env, pi, { gamma, episodes: 400, seed: 5 }, vTrue),
    [env, pi, gamma, vTrue],
  )

  const zeros = useMemo(() => new Array<number>(mdp.nS).fill(0), [mdp.nS])
  const ctx = useFormulaCtx(mdp, pi, gamma, zeros, 6)

  /* ── 控制：网格上的 Sarsa / Q-learning ── */
  const star = useMemo(() => valueIterationSolve(mdp, gamma), [mdp, gamma])
  const optActions = useMemo(() => argmaxActions(star.policy), [star])

  const runs = useMemo(
    () =>
      (['sarsa', 'qlearning', 'expected-sarsa'] as const).map((variant) => ({
        variant,
        res: tdControl(env, variant, {
          gamma,
          alpha,
          eps,
          episodes,
          seed: 11,
          probes: 40,
          decay: true,
        }),
      })),
    [env, gamma, alpha, eps, episodes],
  )
  const cur = runs.find((r) => r.variant === duel)!.res
  const kk = Math.min(k, cur.snaps.length - 1)
  const snap = cur.snaps[kk]
  const wrong = useMemo(
    () => argmaxActions(snap.policy).filter((a, i) => a !== optActions[i]).length,
    [snap, optActions],
  )

  /* ── n 步 Sarsa 谱系 ── */
  const nRun = useMemo(
    () =>
      nStepSarsa(env, nStep, {
        gamma,
        alpha,
        eps,
        episodes: 400,
        seed: 11,
        probes: 30,
        decay: true,
      }),
    [env, nStep, gamma, alpha, eps],
  )

  /* ── 悬崖对决 ── */
  const cEnv = useMemo(() => cliffEnv(), [])
  const cMdp = useMemo(() => cliffMDP(), [])
  const cliffSarsa = useMemo(
    () => tdControl(cEnv, 'sarsa', { gamma: 1, alpha: 0.5, eps: 0.1, episodes: 600, seed: 2, probes: 30 }),
    [cEnv],
  )
  const cliffQ = useMemo(
    () => tdControl(cEnv, 'qlearning', { gamma: 1, alpha: 0.5, eps: 0.1, episodes: 600, seed: 2, probes: 30 }),
    [cEnv],
  )
  const pathSarsa = useMemo(() => greedyPath(cEnv, cliffSarsa.q), [cEnv, cliffSarsa])
  const pathQ = useMemo(() => greedyPath(cEnv, cliffQ.q), [cEnv, cliffQ])
  const avgLast = (xs: number[]) => xs.slice(-100).reduce((a, b) => a + b, 0) / 100

  return (
    <div>
      <ChapterHero
        n={7}
        hook="这张许可证如果用回贝尔曼公式，会长成什么算法？"
        lead={
          <>
            <p>
              这一章几乎不需要新想法。把第 6 章的 Robbins-Monro 算法，
              原封不动地套在第 2 章的贝尔曼公式上 —— TD 算法就掉出来了。
            </p>
            <p>
              真正值得琢磨的是它带来的一个哲学转向：
              <strong>用估计去更新估计</strong>。听起来像左脚踩右脚，
              但正是这一步让「不等回合结束、走一步就学一次」成为可能。
            </p>
            <p>
              然后这一章会以一场对决收尾。同样的悬崖，同样的起点，
              Sarsa 和 Q-learning 会走出两条完全不同的路 ——
              <em>而先掉下去的那个，恰恰是学得更"正确"的那个。</em>
            </p>
          </>
        }
        gains={[
          '亲手把 RM 算法套进贝尔曼公式，推出 TD',
          '看清 TD 的偏差-方差权衡与 MC 正好相反',
          '用一个滑块把 TD 和 MC 用 n 步 TD 连起来',
          '理解 on-policy 与 off-policy 的真正分界',
          '在悬崖世界里看到「学得对」和「跑得好」的分离',
        ]}
      />

      <ChapterGlance g={GLANCE} />

      {/* ───────────────────── 第 1 幕 ───────────────────── */}
      <Act
        id="a1"
        no="第 1 幕"
        title="一行推导，TD 就出来了"
        goal="把贝尔曼公式看成求根问题，RM 算法直接给出 TD。"
        minutes={9}
        points={[
          <>
            贝尔曼公式 <M>{"v(s)=\\mathbb{E}[R+\\gamma v(S')|s]"}</M> 等价于求{' '}
            <M>{"g(v)=v(s)-\\mathbb{E}[R+\\gamma v(S')]=0"}</M> 的根。
          </>,
          <>
            一次真实转移 <M>{"(s,r,s')"}</M> 就是 <M>{'g'}</M> 的一个带噪观测。
            套 RM，得到 TD(0)。
          </>,
          <>
            <M>{"\\delta_t = r_{t+1}+\\gamma v(s_{t+1})-v(s_t)"}</M> 叫 TD 误差，
            它是<strong>贝尔曼误差的采样版</strong>，为 0 意味着估计满足贝尔曼公式。
          </>,
        ]}
        stage={() => (
          <div className="space-y-4">
            <Panel title="活的公式：TD(0)">
              <LiveFormula node={F_TD} ctx={ctx} />
              <p className="mt-3 border-t border-line pt-3 text-[11.5px] leading-relaxed text-faint">
                和第 6 章那三块积木对照：新估计 = 旧估计 + 步长 × 误差。
                这里的误差换成了 TD 误差，仅此而已。
              </p>
            </Panel>

            <Panel title="TD vs MC：谁抖得厉害">
              <LineChart
                height={200}
                logY
                xLabel="回合数"
                series={[
                  { name: `TD(0)  α=${fmt(alpha, 2)}`, color: C.accent, data: smooth(td.err, 5) },
                  { name: '蒙特卡洛', color: C.reward, data: smooth(mc.err, 5) },
                ]}
              />
              <div className="mt-3">
                <Slider
                  label={<>步长 <M>{'\\alpha'}</M></>}
                  value={alpha}
                  min={0.01}
                  max={0.5}
                  step={0.01}
                  onChange={setAlpha}
                  accent={C.gamma}
                  hint="α 大则学得快但抖得厉害；α 小则平稳但慢"
                />
              </div>
            </Panel>

            <Panel title="两种估计的最终形态">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <div className="mb-1.5 text-center text-[11px] text-faint">TD 估计</div>
                  <GridWorld mdp={mdp} v={td.v} showPolicy={false} cell={44} quiet />
                </div>
                <div>
                  <div className="mb-1.5 text-center text-[11px] text-faint">真值 v_π</div>
                  <GridWorld mdp={mdp} v={vTrue} showPolicy={false} cell={44} quiet />
                </div>
              </div>
            </Panel>
          </div>
        )}
      >
        <Beat id="b1" keep>
          <p>
            第 2 章说：<M>{"v_\\pi(s) = \\mathbb{E}\\left[R_{t+1} + \\gamma v_\\pi(S_{t+1}) \\mid S_t = s\\right]"}</M>。
            把它挪项，写成求根的形式：
          </p>
          <MB>{"g(v) \\;\\doteq\\; v(s) - \\mathbb{E}\\left[R + \\gamma v(S') \\mid s\\right] \\;=\\; 0"}</MB>
          <p>
            我们不知道那个期望。但每走一步，环境就免费给我们一个样本 ——
            一次真实的转移 <M>{"(s, r, s')"}</M>，它给出
          </p>
          <MB>{"\\tilde g(v) = v(s) - \\left( r + \\gamma v(s') \\right) = g(v) + \\underbrace{\\eta}_{\\text{采样噪声}}"}</MB>
        </Beat>

        <Beat id="b2" keep>
          <p>剩下的就是照抄第 6 章的 RM 更新式 <M>{'w_{k+1}=w_k-\\alpha_k\\tilde g(w_k)'}</M>：</p>
          <MB>{"v(s_t) \\leftarrow v(s_t) - \\alpha\\left[ v(s_t) - \\left( r_{t+1} + \\gamma v(s_{t+1}) \\right) \\right]"}</MB>
          <p>整理一下符号，就是教科书上的 TD(0)：</p>
          <MB>{"v(s_t) \\leftarrow v(s_t) + \\alpha\\underbrace{\\left[ r_{t+1} + \\gamma v(s_{t+1}) - v(s_t) \\right]}_{\\delta_t\\ \\text{（TD 误差）}}"}</MB>
          <p>
            <strong>没有新东西。</strong>这一整章的核心推导就是上面这三行。
          </p>
        </Beat>

        <Beat id="b3" keep>
          <PredictChoice
            id="ch7-tdmc"
            question={
              <>
                用 TD 和 MC 同时估计同一个 <M>{'v_\\pi'}</M>，跑同样多的回合。
                谁的误差曲线更平滑？
              </>
            }
            options={[
              { id: 'a', label: 'MC —— 它是无偏的，所以更准也更稳' },
              { id: 'b', label: 'TD —— 它每步只吸收一个随机奖励，方差小' },
              { id: 'c', label: '差不多，两者都是无偏估计' },
            ]}
            answer="b"
            explain={
              <>
                <p>
                  MC 的目标是 <M>{'G_t'}</M> —— 整条轨迹上几十个随机奖励的加权和，
                  方差自然大。TD 的目标是 <M>{"r_{t+1}+\\gamma v(s_{t+1})"}</M>，
                  只含<strong>一个</strong>随机奖励。
                </p>
                <p>
                  代价是 TD <strong>有偏</strong>：<M>{"v(s_{t+1})"}</M> 是当前的估计，不是真值。
                  所以这是一次典型的偏差-方差交换 —— MC 无偏高方差，TD 有偏低方差。
                </p>
                <p>
                  右边那张对数图里，MC 那条线在低误差区抖得明显更凶。
                </p>
              </>
            }
          />
        </Beat>

        <Beat id="b4">
          <Callout tone="insight" title="「自举」这个词">
            <p>
              TD 用 <M>{"v(s_{t+1})"}</M> 去更新 <M>{'v(s_t)'}</M> ——
              用一个还不准的估计，去改进另一个还不准的估计。这叫<strong>自举</strong>
              （bootstrapping）。
            </p>
            <p>
              它听起来不可靠，但其实非常合理：贝尔曼公式说的正是
              「相邻状态的价值之间必须满足某种关系」。TD 做的事就是
              <em>反复把这个关系强加到估计上</em>，直到它们彼此自洽。
            </p>
            <p>
              自举带来了效率，也带来了风险。第 8 章会看到，
              当自举遇上函数近似和离策略，三者合流会让算法直接发散。
            </p>
          </Callout>
        </Beat>
      </Act>

      {/* ───────────────────── 第 2 幕 ───────────────────── */}
      <Act
        id="a2"
        no="第 2 幕"
        title="n 步 TD：TD 与 MC 之间的连续谱"
        goal="n 从 1 到 ∞，算法从 TD 平滑地变成蒙特卡洛。"
        minutes={8}
        points={[
          <>
            <M>{'n'}</M> 步目标：<M>{"G_t^{(n)} = r_{t+1}+\\cdots+\\gamma^{n-1}r_{t+n}+\\gamma^n q(s_{t+n},a_{t+n})"}</M>。
          </>,
          <>
            <M>{'n=1'}</M> 是 Sarsa，<M>{'n=\\infty'}</M> 是蒙特卡洛。
            中间的 <M>{'n'}</M> 通常比两端都好。
          </>,
          <>
            <M>{'n'}</M> 越大：<strong>偏差越小、方差越大</strong>。
            它就是偏差-方差旋钮本身。
          </>,
        ]}
        stage={() => (
          <div className="space-y-4">
            <Panel title={`n 步 Sarsa（n = ${nStep}）`}>
              <GridWorld
                mdp={mdp}
                v={nRun.q.map((r) => Math.max(...r))}
                policy={nRun.policy}
                cell={52}
                quiet
              />
              <div className="mt-3">
                <Slider
                  label={<>步数 <M>{'n'}</M></>}
                  value={nStep}
                  min={1}
                  max={20}
                  step={1}
                  onChange={setNStep}
                  format={(v) => String(v)}
                  hint={
                    nStep === 1
                      ? '此刻它就是 Sarsa'
                      : nStep >= 18
                        ? '此刻它已经很接近蒙特卡洛了'
                        : '中间地带：偏差小一点，方差大一点'
                  }
                />
              </div>
            </Panel>
            <Panel title="学习曲线">
              <LineChart
                height={180}
                xLabel="回合"
                series={[{ name: `n=${nStep}`, color: C.accent, data: smooth(nRun.episodeReturn, 20) }]}
              />
            </Panel>
          </div>
        )}
      >
        <Beat id="c1" keep>
          <p>
            TD 只往前看一步，MC 看到底。既然是两个极端，中间当然可以取值：
            往前看 <M>{'n'}</M> 步，剩下的用估计顶上。
          </p>
          <MB>
            {String.raw`\begin{aligned}
            n=1:\quad & G_t^{(1)} = r_{t+1} + \gamma q(s_{t+1},a_{t+1}) && \text{Sarsa} \\
            n=2:\quad & G_t^{(2)} = r_{t+1} + \gamma r_{t+2} + \gamma^2 q(s_{t+2},a_{t+2}) \\
            & \;\;\vdots \\
            n=\infty:\quad & G_t^{(\infty)} = r_{t+1} + \gamma r_{t+2} + \cdots && \text{蒙特卡洛}
            \end{aligned}`}
          </MB>
        </Beat>

        <Beat id="c2" keep>
          <p>
            这个滑块的意义不只是「多一个超参数」。它把两个看起来不同的方法族
            <strong>放到了同一根轴上</strong>：偏差和方差之间，
            <M>{'n'}</M> 就是那个刻度。
          </p>
          <p>
            实践中最好的 <M>{'n'}</M> 常常既不是 1 也不是 ∞，而是 3 到 10 之间的某个数 ——
            和第 4 章截断策略迭代的最优区间惊人地相似。
          </p>
        </Beat>
      </Act>

      {/* ───────────────────── 第 3 幕 ───────────────────── */}
      <Act
        id="a3"
        no="第 3 幕"
        title="Sarsa 与 Q-learning：一个符号的分野"
        goal="on-policy 与 off-policy 的区别，全在 TD 目标里那一项。"
        minutes={10}
        points={[
          <>
            Sarsa：<M>{"r+\\gamma q(s',a')"}</M>，<M>{"a'"}</M> 是<strong>实际会走</strong>的动作 →
            on-policy。
          </>,
          <>
            Q-learning：<M>{"r+\\gamma\\max_a q(s',a)"}</M>，不管实际走哪个 → off-policy。
          </>,
          <>
            off-policy 让<strong>行为策略</strong>（负责探索）与<strong>目标策略</strong>
            （负责最优）分离，这是经验回放、DQN 得以成立的前提。
          </>,
          <>
            Expected Sarsa 是折中：用 <M>{"\\sum_a\\pi(a|s')q(s',a)"}</M>，
            去掉了 <M>{"a'"}</M> 的采样方差。
          </>,
        ]}
        stage={() => (
          <div className="space-y-4">
            <Panel
              title={`${duel === 'sarsa' ? 'Sarsa' : 'Q-learning'} · 第 ${snap.ep} 回合`}
              right={
                <span className="font-mono text-[11.5px]" style={{ color: wrong ? C.danger : C.value }}>
                  错 {wrong} / 25
                </span>
              }
            >
              <GridWorld
                mdp={mdp}
                v={snap.q.map((r) => Math.max(...r))}
                policy={snap.policy}
                cell={52}
                quiet
              />
              <div className="mt-3">
                <Scrubber k={kk} setK={setK} max={cur.snaps.length - 1} label="快照" fps={5} />
              </div>
              <div className="mt-3">
                <Seg
                  value={duel}
                  onChange={setDuel}
                  size="sm"
                  options={[
                    { value: 'sarsa', label: 'Sarsa' },
                    { value: 'qlearning', label: 'Q-learning' },
                  ]}
                />
              </div>
            </Panel>
            <Panel title="三个变体的学习曲线">
              <LineChart
                height={186}
                xLabel="回合"
                series={runs.map((r, i) => ({
                  name: { sarsa: 'Sarsa', qlearning: 'Q-learning', 'expected-sarsa': 'Expected Sarsa' }[
                    r.variant
                  ],
                  color: [C.accent, C.qvalue, C.value][i],
                  data: smooth(r.res.episodeReturn, 25),
                }))}
              />
              <div className="mt-3 space-y-3">
                <Slider
                  label={<M>{'\\varepsilon'}</M>}
                  value={eps}
                  min={0}
                  max={0.5}
                  step={0.01}
                  onChange={setEps}
                  accent={C.reward}
                />
                <Slider
                  label="回合数"
                  value={episodes}
                  min={100}
                  max={2000}
                  step={100}
                  onChange={setEpisodes}
                  format={(v) => String(v)}
                />
              </div>
            </Panel>
          </div>
        )}
      >
        <Beat id="d1" keep>
          <p>
            把 TD 从「估 v」改成「估 q」，就得到了能做控制的算法。
            此时 TD 目标里需要 <M>{"q(s', \\cdot)"}</M>，
            而<strong>用哪个动作的 q</strong>，就成了一个真正的选择。
          </p>
          <MB>
            {String.raw`\begin{aligned}
            \text{Sarsa:}\quad & q(s,a) \leftarrow q(s,a) + \alpha\left[r + \gamma\, q(s',a') - q(s,a)\right] \\[6pt]
            \text{Q-learning:}\quad & q(s,a) \leftarrow q(s,a) + \alpha\left[r + \gamma\, \max_{a''} q(s',a'') - q(s,a)\right]
            \end{aligned}`}
          </MB>
          <p>
            差别只有一处：<M>{"q(s',a')"}</M> 还是 <M>{"\\max_{a''} q(s',a'')"}</M>。
            但这一处改动，改变了这个算法在学<strong>谁</strong>。
          </p>
        </Beat>

        <Beat id="d2" keep>
          <p>
            Sarsa 里的 <M>{"a'"}</M> 是<em>你接下来真的要执行的那个动作</em> ——
            包括 ε-贪心里那些随机乱走的动作。所以它评估的是
            <strong>「我这个带探索的策略」本身有多好</strong>。这叫 on-policy。
          </p>
          <p>
            Q-learning 里的 <M>{'\\max'}</M> 不管你实际走哪一步，
            它永远假设「下一步会走最好的」。所以它评估的是
            <strong>最优策略</strong>，尽管数据是用另一个策略采的。这叫 off-policy。
          </p>
          <MB>
            {String.raw`\underbrace{\pi_b}_{\text{行为策略：负责探索}} \quad\neq\quad \underbrace{\pi_t}_{\text{目标策略：负责最优}}`}
          </MB>
        </Beat>

        <Beat id="d3">
          <Details summary="展开：为什么 off-policy 这么重要">
            <p>三件事因它而可能：</p>
            <ul>
              <li>
                <strong>经验回放。</strong>旧策略采的数据仍然能用来改进当前策略 ——
                DQN 的样本效率全靠这个。
              </li>
              <li>
                <strong>从他人的经验中学。</strong>人类演示、其它智能体的日志，
                都能直接拿来训练。
              </li>
              <li>
                <strong>探索与利用彻底解耦。</strong>行为策略可以尽情乱走，
                不必担心污染目标策略的最优性。
              </li>
            </ul>
            <p>
              代价是：off-policy 严格来说需要重要性采样来纠正分布差异。
              Q-learning 之所以不需要显式的重要性权重，
              是因为它的目标里用的是 <M>{'\\max'}</M> 而不是对 <M>{'\\pi_t'}</M> 求期望 ——
              这是一个巧妙的特例，不是普遍规律（第 10 章的离策略 Actor-Critic 就躲不掉）。
            </p>
          </Details>
        </Beat>
      </Act>

      {/* ───────────────────── 第 4 幕：悬崖 ───────────────────── */}
      <Act
        id="a4"
        no="第 4 幕"
        title="悬崖对决"
        goal="同一个世界，两个算法给出两条不同的路 —— 而学得更「对」的那个跑得更差。"
        minutes={10}
        points={[
          <>
            Q-learning 学到<strong>贴着悬崖的最短路</strong>，因为它假设下一步永远最优。
          </>,
          <>
            Sarsa 学到<strong>绕远一行的安全路</strong>，因为它知道自己会以 ε 的概率乱走。
          </>,
          <>
            在线平均回报：<strong>Sarsa 更高</strong>。「学到最优策略」和
            「学习期间表现最好」是两个不同的目标。
          </>,
          <>
            把 ε 衰减到 0，两者最终会一致 —— 分歧来自探索，不来自算法本身的对错。
          </>,
        ]}
        stage={() => (
          <div className="space-y-4">
            <Panel title="Sarsa 学到的路（贪心执行）">
              <GridWorld
                mdp={cMdp}
                policy={cliffSarsa.policy}
                trail={pathSarsa}
                showValues={false}
                showHeatmap={false}
                cell={34}
                maxH="24vh"
                quiet
              />
            </Panel>
            <Panel title="Q-learning 学到的路（贪心执行）">
              <GridWorld
                mdp={cMdp}
                policy={cliffQ.policy}
                trail={pathQ}
                showValues={false}
                showHeatmap={false}
                cell={34}
                maxH="24vh"
                quiet
              />
            </Panel>
            <Panel title="在线表现：每回合的累计奖励">
              <LineChart
                height={186}
                xLabel="回合"
                yMin={-160}
                yMax={0}
                series={[
                  { name: 'Sarsa', color: C.accent, data: smooth(cliffSarsa.episodeReturn, 25) },
                  { name: 'Q-learning', color: C.qvalue, data: smooth(cliffQ.episodeReturn, 25) },
                ]}
              />
              <div className="mt-3 grid grid-cols-2 gap-2.5">
                <Stat
                  label="Sarsa 后 100 回合均值"
                  value={fmt(avgLast(cliffSarsa.episodeReturn), 1)}
                  color={C.accent}
                />
                <Stat
                  label="Q-learning 后 100 回合均值"
                  value={fmt(avgLast(cliffQ.episodeReturn), 1)}
                  color={C.qvalue}
                />
              </div>
            </Panel>
          </div>
        )}
      >
        <Beat id="e1" keep>
          <p>
            舞台是一个 4×12 的世界。左下角出发，右下角是终点，
            中间整排是悬崖：踩上去扣 100 分并被扔回起点。其余每走一步扣 1 分。
          </p>
          <p>
            最短路显然是<strong>贴着悬崖直着走</strong>，代价 −13。
            绕上一行再走，代价 −15。理论最优毫无悬念。
          </p>
        </Beat>

        <Beat id="e2" keep>
          <PredictChoice
            id="ch7-cliff"
            question={
              <>
                两个算法都用 <M>{'\\varepsilon = 0.1'}</M> 训练 600 回合。
                <strong>训练期间的平均得分</strong>谁更高？
              </>
            }
            options={[
              { id: 'a', label: 'Q-learning —— 它学的是最优策略，当然更高' },
              { id: 'b', label: 'Sarsa —— 它学到的路更保守，反而得分更高' },
              { id: 'c', label: '一样高，两者最终都会收敛到最优' },
            ]}
            answer="b"
            explain={
              <>
                <p>
                  Q-learning 学到的确实是<strong>那条最优路</strong>：贴着悬崖走。
                  可它在训练时用的是 ε-贪心 —— 每一步有 10% 概率乱走。
                  紧贴悬崖时，一次乱走就是 −100。
                </p>
                <p>
                  Sarsa 的 TD 目标里用的是<em>实际会执行的动作</em>，
                  所以它「知道」自己有 10% 概率手滑。于是它学出来的价值
                  自动把这份风险计入，最终选择了绕远一行的安全路。
                </p>
                <p>
                  右边两条曲线的差距（{fmt(avgLast(cliffSarsa.episodeReturn), 1)} vs{' '}
                  {fmt(avgLast(cliffQ.episodeReturn), 1)}）就是这份「谨慎」的价值。
                </p>
                <p>
                  <strong>这不是说 Q-learning 错了。</strong>它回答的是
                  「最优策略是什么」；Sarsa 回答的是「我这个会手滑的家伙该怎么走」。
                  你关心哪一个，取决于训练过程中的损失要不要算钱。
                </p>
              </>
            }
          />
        </Beat>

        <Beat id="e3" keep>
          <Code
            code={`# 两个算法只差第 8 行
def td_control(env, variant, gamma=1.0, alpha=0.5, eps=0.1, episodes=600):
    q = [[0.0] * env.nA for _ in range(env.nS)]
    for _ in range(episodes):
        s = env.reset()
        a = eps_greedy(q[s], eps)
        while True:
            sp, r, done = env.step(s, a)
            ap = eps_greedy(q[sp], eps)

            if variant == "sarsa":          target = r + gamma * q[sp][ap]
            elif variant == "qlearning":    target = r + gamma * max(q[sp])
            else:  # expected sarsa
                target = r + gamma * expected(q[sp], eps)

            q[s][a] += alpha * (target - q[s][a])
            if done: break
            s, a = sp, ap
    return q`}
          />
        </Beat>

        <Beat id="e4" keep>
          <Callout tone="trap" title="这一章留下的问题">
            <p>
              到这里，无模型的表格法已经完备了：能预测、能控制、能在线、能离策略。
              但请数一下这些算法要存多少东西：一张 <M>{'|\\mathcal S|\\times|\\mathcal A|'}</M> 的表。
            </p>
            <p>
              围棋有 <M>{'10^{170}'}</M> 个状态。自动驾驶的状态是连续的，
              <strong>压根不可数</strong>。表格这条路，在这里走到头了。
            </p>
            <p>
              下一章要做的事听起来很自然：<em>用一个带参数的函数去代替那张表。</em>
              但这一步远比想象中危险 —— 它会让前面所有的收敛保证一起失效。
            </p>
          </Callout>
        </Beat>
      </Act>
    </div>
  )
}
