import { useMemo, useState } from 'react'
import { ACTIONS, buildGridMDP, classicGrid } from '../core/mdp'
import { argmaxActions, qFromV, uniformPolicy } from '../core/policy'
import { epsilonGreedyOptimal, policyEvaluationDirect, valueIterationSolve } from '../core/solvers'
import { runningMean, sampleActionReturns } from '../core/sample'
import { mdpToEnv, withExploringStarts } from '../core/env'
import { mcControl, smooth } from '../core/learn'
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
  formula: String.raw`q_\pi(s,a) = \mathbb{E}\!\left[G_t \mid S_t=s, A_t=a\right]
  \;\approx\; \frac{1}{N}\sum_{i=1}^{N} G^{(i)}`,
  formulaNote:
    '整章只有这一个想法：期望算不出来，就用样本平均去顶。大数定律保证 N 足够大时它一定靠得住 —— 代价是必须真的把回合跑完。',
  takeaways: [
    <>
      没有模型时，<strong>直接估 q 而不是 v</strong>。因为从 v 导出策略需要 p 和 r，
      而从 q 取 argmax 什么都不需要。
    </>,
    <>
      MC Basic = 策略迭代把「用模型算 q」换成「用采样估 q」。
      算法骨架一个字没改，只换了取数的方式。
    </>,
    <>
      一条轨迹的<strong>每一个后缀</strong>都是一份合法样本。用上这一点，
      数据效率能提高一两个数量级 —— 这就是 MC Exploring Starts。
    </>,
    <>
      探索必须显式保证。<M>{'\\varepsilon'}</M>-贪心用「留一点概率乱走」换来了访问所有
      <M>{'(s,a)'}</M> 的可能性，代价是最优性会打折。
    </>,
  ],
  traps: [
    <>
      以为「样本多了就准了」意味着可以少跑几条。方差随 <M>{'1/\\sqrt{N}'}</M> 下降，
      想把误差减半要<strong>四倍</strong>的样本。
    </>,
    <>
      混淆 first-visit 与 every-visit。二者都收敛到 <M>{'q_\\pi'}</M>，
      但 every-visit 的样本不独立，有限样本下略有偏。
    </>,
    <>
      忘了 <M>{'\\varepsilon'}</M>-贪心学到的是「最优的 ε-贪心策略」，
      <strong>不是最优策略</strong>。ε 越大，两者差得越远。
    </>,
  ],
}

export function Chapter5() {
  const C = useColors()
  const [gamma] = useState(0.9)
  const [eps, setEps] = useState(0.2)
  const [episodes, setEpisodes] = useState(600)
  const [nSamples, setNSamples] = useState(40)
  const [k, setK] = useState(0)
  const [starts, setStarts] = useState<'fixed' | 'exploring'>('exploring')

  const focus = useBus((s) => s.focus)
  const setFocus = useBus((s) => s.setFocus)

  const mdp = useMemo(() => buildGridMDP(classicGrid()), [])
  const star = useMemo(() => valueIterationSolve(mdp, gamma), [mdp, gamma])
  const optActions = useMemo(() => argmaxActions(star.policy), [star])

  /* ── 单个 (s,a) 的蒙特卡洛估计 ── */
  const pi0 = useMemo(() => uniformPolicy(mdp.nS), [mdp.nS])
  const v0 = useMemo(() => policyEvaluationDirect(mdp, pi0, gamma), [mdp, pi0, gamma])
  const q0 = useMemo(() => qFromV(mdp, gamma, v0), [mdp, gamma, v0])
  const demoS = focus ?? 6
  const demoA = 1
  const truthQ = q0[demoS][demoA]

  const draws = useMemo(
    () => sampleActionReturns(mdp, pi0, gamma, demoS, demoA, 300, 100, 2024),
    [mdp, pi0, gamma, demoS],
  )
  const means = useMemo(() => runningMean(draws), [draws])
  const estimate = means[Math.min(nSamples, means.length) - 1]

  /* ── MC 控制 ── */
  const env = useMemo(() => mdpToEnv(mdp, { start: 0, horizon: 30 }), [mdp])
  const envES = useMemo(() => withExploringStarts(env), [env])

  const mc = useMemo(
    () =>
      mcControl(starts === 'exploring' ? envES : env, {
        gamma,
        alpha: 0,
        eps,
        episodes,
        seed: 3,
        probes: 40,
        decay: false,
      }),
    [env, envES, starts, gamma, eps, episodes],
  )
  const kk = Math.min(k, mc.snaps.length - 1)
  const snap = mc.snaps[kk]
  const wrong = useMemo(
    () => argmaxActions(snap.policy).filter((a, i) => a !== optActions[i]).length,
    [snap, optActions],
  )
  const finalWrong = useMemo(
    () => argmaxActions(mc.policy).filter((a, i) => a !== optActions[i]).length,
    [mc, optActions],
  )
  const snapV = useMemo(() => snap.q.map((row) => Math.max(...row)), [snap])

  /* ── ε 的代价 ── */
  const epsOpt = useMemo(() => epsilonGreedyOptimal(mdp, gamma, eps), [mdp, gamma, eps])
  const epsMismatch = useMemo(
    () => epsOpt.greedyActions.filter((a, i) => a !== optActions[i]).length,
    [epsOpt, optActions],
  )
  const epsCurve = useMemo(() => {
    const xs = Array.from({ length: 21 }, (_, i) => i * 0.05)
    return xs.map(
      (e) => epsilonGreedyOptimal(mdp, gamma, e).greedyActions.filter((a, i) => a !== optActions[i]).length,
    )
  }, [mdp, gamma, optActions])

  return (
    <div>
      <ChapterHero
        n={5}
        hook="算法要用到 p(s′|s,a) 和 r(s,a) —— 可现实里根本没人给我模型。"
        lead={
          <>
            <p>
              前四章的每一个公式里都藏着一个 <M>{'\\mathbb{E}'}</M>，
              而算这个期望需要知道概率分布。现在把这个特权收回：
              <strong>你只能与世界交互，观察发生了什么。</strong>
            </p>
            <p>
              办法其实小学生都知道 —— <em>不知道平均值，就多试几次取平均</em>。
              这一章真正值得学的不是这个想法本身，而是把它塞进策略迭代之后，
              一连串被逼出来的工程决策：样本从哪来、怎么才够用、没走过的路怎么办。
            </p>
          </>
        }
        gains={[
          '理解为什么无模型时必须估 q 而不是 v',
          'MC Basic：把策略迭代里唯一的模型依赖换成采样',
          '看见蒙特卡洛估计的抖动，以及它随 1/√N 收敛的速度',
          '用一条轨迹的所有后缀，把数据效率提上去',
          '亲手量出 ε-贪心为了探索付出的最优性代价',
        ]}
      />

      <ChapterGlance g={GLANCE} />

      {/* ───────────────────── 第 1 幕 ───────────────────── */}
      <Act
        id="a1"
        no="第 1 幕"
        title="把期望换成平均"
        goal="没有模型时，唯一还能做的事情是采样；大数定律负责兜底。"
        minutes={9}
        points={[
          <>
            无模型时估 <strong>q 而非 v</strong>：从 q 取 argmax 不需要模型，
            从 v 取 argmax 需要 <M>{'p'}</M> 和 <M>{'r'}</M>。
          </>,
          <>
            <M>{'q_\\pi(s,a) \\approx \\frac1N\\sum_i G^{(i)}'}</M>，
            其中每个 <M>{'G^{(i)}'}</M> 是从 <M>{'(s,a)'}</M> 出发跑完一整条轨迹的折扣回报。
          </>,
          <>
            误差按 <M>{'1/\\sqrt N'}</M> 收缩：要精度翻倍，样本得翻四倍。
            这是蒙特卡洛的原罪。
          </>,
        ]}
        stage={() => (
          <div className="space-y-4">
            <Panel title={`估计 q(s${demoS + 1}, ${ACTIONS[demoA].name})`}>
              <GridWorld
                mdp={mdp}
                v={v0}
                policy={pi0}
                cell={54}
                showValues={false}
                onCellClick={(x) => setFocus(x === focus ? null : x)}
              />
              <p className="mt-2 text-center text-[11.5px] text-faint">
                点一个格子，换一个要估计的 (s, a)
              </p>
            </Panel>

            <Panel title="样本平均在往真值爬">
              <LineChart
                height={186}
                marker={Math.min(nSamples, means.length) - 1}
                xLabel="用了多少条轨迹"
                series={[
                  { name: '样本平均', color: C.accent, data: means.slice(0, 120) },
                  {
                    name: '真值 q_π(s,a)',
                    color: C.value,
                    data: new Array(Math.min(120, means.length)).fill(truthQ),
                    dashed: true,
                  },
                ]}
              />
              <div className="mt-3">
                <Slider
                  label="轨迹条数 N"
                  value={nSamples}
                  min={1}
                  max={120}
                  step={1}
                  onChange={setNSamples}
                  format={(v) => String(v)}
                />
              </div>
              <div className="mt-3 grid grid-cols-3 gap-2.5">
                <Stat label="估计值" value={fmt(estimate, 3)} color={C.accent} />
                <Stat label="真值" value={fmt(truthQ, 3)} color={C.value} />
                <Stat
                  label="误差"
                  value={fmt(Math.abs(estimate - truthQ), 3)}
                  color={Math.abs(estimate - truthQ) > 0.1 ? C.danger : C.value}
                />
              </div>
            </Panel>
          </div>
        )}
      >
        <Beat id="b1" keep>
          <p>
            先看清楚模型到底卡在哪一步。策略改进要做的是
          </p>
          <MB>{"\\pi_{k+1}(s) = \\arg\\max_a\\underbrace{\\left[ r(s,a) + \\gamma\\sum_{s'}p(s'|s,a)v_{\\pi_k}(s') \\right]}_{q_{\\pi_k}(s,a)}"}</MB>
          <p>
            方括号里两个模型量：<M>{'r(s,a)'}</M> 和 <M>{"p(s'|s,a)"}</M>。
            但请注意 —— 它们只在<strong>把 v 换算成 q 的时候</strong>才被用到。
          </p>
          <p>
            于是有一个绕过去的办法：<em>别去估 v 了，直接估 q。</em>
          </p>
        </Beat>

        <Beat id="b2" keep>
          <p>
            <M>{'q_\\pi(s,a)'}</M> 的定义本身就是一个期望：
          </p>
          <MB>{'q_\\pi(s,a) = \\mathbb{E}\\left[ G_t \\mid S_t = s,\\, A_t = a \\right]'}</MB>
          <p>
            而期望有一个不需要知道分布的估计方法 —— 采样求平均。从 <M>{'(s,a)'}</M> 出发，
            按 <M>{'\\pi'}</M> 走完一整条轨迹，把折扣回报记下来；重复 <M>{'N'}</M> 次，取平均。
          </p>
          <MB>{'q_\\pi(s,a) \\;\\approx\\; \\frac{1}{N}\\sum_{i=1}^{N} G^{(i)}'}</MB>
          <p>就这样。这个方法叫蒙特卡洛。</p>
        </Beat>

        <Beat id="b3" keep>
          <PredictNumber
            id="ch5-samples"
            question={
              <>
                右边这条曲线，要用多少条轨迹，估计值才会稳定落在真值
                <M>{'\\pm 0.05'}</M> 以内？
              </>
            }
            min={5}
            max={120}
            step={5}
            truth={(() => {
              for (let i = 5; i < means.length; i++) {
                if (means.slice(i, i + 20).every((m) => Math.abs(m - truthQ) < 0.05)) return i
              }
              return 120
            })()}
            tolerance={25}
            unit=" 条"
            explain={
              <>
                <p>
                  多数人会低估。蒙特卡洛的标准误是{' '}
                  <M>{'\\sigma/\\sqrt{N}'}</M> —— 注意是<strong>根号</strong>。
                  想把误差从 0.2 压到 0.05，需要的样本不是 4 倍，而是 <strong>16 倍</strong>。
                </p>
                <p>
                  而且这还只是<strong>一个</strong> <M>{'(s,a)'}</M> 对。这个 5×5 的世界有 125 个
                  <M>{'(s,a)'}</M> 对，每个都要这么来一遍 —— MC Basic 的低效在这里就已经写在脸上了。
                </p>
              </>
            }
          />
        </Beat>

        <Beat id="b4">
          <Details summary="展开：为什么是 1/√N，以及大数定律在这里到底保证了什么">
            <p>
              设 <M>{'G^{(1)},\\dots,G^{(N)}'}</M> 独立同分布，均值 <M>{'q'}</M>，方差{' '}
              <M>{'\\sigma^2'}</M>。样本均值 <M>{'\\bar G_N'}</M> 满足
            </p>
            <MB>{'\\mathbb{E}[\\bar G_N] = q, \\qquad \\mathrm{Var}[\\bar G_N] = \\frac{\\sigma^2}{N}'}</MB>
            <p>
              所以标准差是 <M>{'\\sigma/\\sqrt N'}</M>。<strong>无偏</strong>意味着它没有系统性偏差
              —— 这是蒙特卡洛相对后面 TD 方法的最大优势；
              <strong>方差大</strong>则是它最大的劣势，因为 <M>{'G'}</M>
              是一整条轨迹上几十个随机奖励的和，方差会累积。
            </p>
            <p>
              顺带说一句：这里假设了折扣回报有界，这依赖 <M>{'\\gamma<1'}</M>。
              γ 又一次在背后当地基。
            </p>
          </Details>
        </Beat>
      </Act>

      {/* ───────────────────── 第 2 幕 ───────────────────── */}
      <Act
        id="a2"
        no="第 2 幕"
        title="别浪费轨迹"
        goal="一条轨迹里藏着很多份样本；用上它们，数据效率差一个数量级。"
        minutes={9}
        points={[
          <>
            MC Basic 每次只用一条轨迹更新一个 <M>{'(s,a)'}</M>，其余全扔掉 —— 极度浪费。
          </>,
          <>
            一条轨迹 <M>{'s_0a_0r_1s_1a_1r_2\\cdots'}</M> 的<strong>每个后缀</strong>
            都是对应 <M>{'(s_t,a_t)'}</M> 的合法样本。从后往前扫一遍就能全部收下。
          </>,
          <>
            first-visit 只用每个 <M>{'(s,a)'}</M> 第一次出现的那份；every-visit 全用。
            两者都收敛，后者样本相关但更省数据。
          </>,
        ]}
        stage={() => (
          <div className="space-y-4">
            <Panel
              title={`MC 控制 · 第 ${snap.ep} 回合`}
              right={
                <span className="font-mono text-[11.5px]" style={{ color: wrong ? C.danger : C.value }}>
                  错 {wrong} / 25 格
                </span>
              }
            >
              <GridWorld mdp={mdp} v={snapV} policy={snap.policy} cell={54} />
              <div className="mt-3">
                <Scrubber k={kk} setK={setK} max={mc.snaps.length - 1} label="快照" fps={5} />
              </div>
            </Panel>
            <Panel title="起点方式">
              <Seg
                value={starts}
                onChange={setStarts}
                options={[
                  { value: 'exploring', label: 'Exploring Starts', hint: '每回合从随机 (s,a) 出发' },
                  { value: 'fixed', label: '固定起点', hint: '永远从左上角出发' },
                ]}
              />
              <p className="mt-3 text-[11.5px] leading-relaxed text-faint">
                切成「固定起点」看看：很多格子根本没被访问过，它们的策略也就永远学不对。
              </p>
              <div className="mt-3 grid grid-cols-2 gap-2.5">
                <Stat label="最终错误格子数" value={finalWrong} color={finalWrong ? C.danger : C.value} />
                <Stat label="用了多少回合" value={episodes} color={C.accent} />
              </div>
            </Panel>
          </div>
        )}
      >
        <Beat id="c1" keep>
          <p>
            MC Basic 能跑，但笨得离谱：为了估一个 <M>{'q(s,a)'}</M>，
            它专门去跑一批从 <M>{'(s,a)'}</M> 出发的轨迹，跑完就扔。
            可那条轨迹后面经过的<strong>每一个</strong> <M>{'(s_t,a_t)'}</M>，
            它的后缀不也正好是一份样本吗？
          </p>
          <MB>
            {String.raw`\underbrace{s_0,a_0,r_1,}_{\text{给 }(s_0,a_0)}\underbrace{s_1,a_1,r_2,}_{\text{也给 }(s_1,a_1)}\underbrace{s_2,a_2,r_3,\dots}_{\text{还给 }(s_2,a_2)}`}
          </MB>
          <p>
            从后往前扫一遍，用 <M>{'G \\leftarrow \\gamma G + r_{t+1}'}</M> 递推，
            一条轨迹就同时更新了几十个 <M>{'(s,a)'}</M>。这就是 MC Exploring Starts。
          </p>
        </Beat>

        <Beat id="c2" keep>
          <Callout tone="insight" title="「探索性起点」这四个字是有代价的">
            <p>
              为了让每个 <M>{'(s,a)'}</M> 都有机会被访问，这个算法要求
              <strong>每一回合可以从任意 (s,a) 开始</strong>。
              棋盘游戏里这没问题（随便摆个局面），
              但真实机器人没法「传送到任意位置再随便做个动作」。
            </p>
            <p>
              右边把它切成「固定起点」你就会看到问题：
              没访问过的格子，它的箭头指哪儿完全是随机的。
            </p>
          </Callout>
        </Beat>

        <Beat id="c3">
          <Details summary="展开：first-visit 与 every-visit 的区别，以及为什么都对">
            <p>
              一条轨迹里同一个 <M>{'(s,a)'}</M> 可能出现好几次。
              <strong>first-visit</strong> 只取第一次出现之后的回报作为样本；
              <strong>every-visit</strong> 每次出现都取。
            </p>
            <p>
              first-visit 的样本之间独立同分布，所以是标准的无偏估计。
              every-visit 的样本互相有重叠（后一个后缀是前一个的一部分），
              有限样本下有偏，但 <M>{'N\\to\\infty'}</M> 时同样收敛到 <M>{'q_\\pi'}</M>，
              而且实践中往往更省数据。
            </p>
            <p>本页的实现用的是 every-visit —— 这也是绝大多数工程代码的选择。</p>
          </Details>
        </Beat>
      </Act>

      {/* ───────────────────── 第 3 幕 ───────────────────── */}
      <Act
        id="a3"
        no="第 3 幕"
        title="探索的价格"
        goal="ε-贪心解决了探索问题，但它学到的不是最优策略，而是最优的 ε-贪心策略。"
        minutes={10}
        points={[
          <>
            <M>{'\\varepsilon'}</M>-贪心：以 <M>{'1-\\varepsilon+\\varepsilon/|\\mathcal A|'}</M>
            选最优动作，其余动作各分 <M>{'\\varepsilon/|\\mathcal A|'}</M>。
          </>,
          <>
            它保证了<strong>所有动作概率恒为正</strong>，于是不需要 exploring starts
            也能访问到每个 <M>{'(s,a)'}</M>。
          </>,
          <>
            代价：最优的 ε-贪心策略 ≠ 最优策略。ε 越大偏得越多 ——
            右边那条曲线把这个代价量成了「有几个格子的箭头指错」。
          </>,
        ]}
        stage={() => (
          <div className="space-y-4">
            <Panel title={`最优的 ε-贪心策略（ε = ${fmt(eps, 2)}）`}>
              <GridWorld mdp={mdp} v={epsOpt.v} policy={epsOpt.policy} cell={54} quiet />
              <div className="mt-3">
                <Slider
                  label={<M>{'\\varepsilon'}</M>}
                  value={eps}
                  min={0}
                  max={1}
                  step={0.02}
                  onChange={setEps}
                  accent={C.reward}
                  hint={
                    epsMismatch === 0
                      ? '此刻它的贪心动作和真正的最优策略完全一致'
                      : `此刻有 ${epsMismatch} 个格子的贪心动作已经和最优策略不同`
                  }
                />
              </div>
            </Panel>

            <Panel title="ε 越大，偏离最优越远">
              <LineChart
                height={176}
                marker={Math.round(eps / 0.05)}
                xLabel="ε（0 → 1）"
                series={[
                  {
                    name: '与最优策略不同的格子数',
                    color: C.danger,
                    data: epsCurve,
                  },
                ]}
              />
              <p className="mt-2 text-[11.5px] leading-relaxed text-faint">
                注意曲线在小 ε 段是<strong>平的</strong>：一定范围内的探索是免费的。
                越过某个阈值之后才开始付费。
              </p>
            </Panel>
          </div>
        )}
      >
        <Beat id="d1" keep>
          <p>
            Exploring starts 不现实，但它想要的东西是合理的：
            <strong>每个 (s,a) 都得有机会被试到。</strong>
            与其在起点上做文章，不如让策略本身永远保留一点随机性。
          </p>
          <MB>
            {String.raw`\pi(a|s) = \begin{cases}
            1 - \dfrac{|\mathcal{A}|-1}{|\mathcal{A}|}\varepsilon, & a = \arg\max_{a'} q(s,a') \\[10pt]
            \dfrac{\varepsilon}{|\mathcal{A}|}, & \text{其它动作}
            \end{cases}`}
          </MB>
          <p>
            这就是 <M>{'\\varepsilon'}</M>-贪心。所有动作的概率都严格为正，
            所以只要跑得足够久，每个 <M>{'(s,a)'}</M> 都会被访问无穷多次。
          </p>
        </Beat>

        <Beat id="d2" keep>
          <PredictChoice
            id="ch5-eps"
            question={
              <>
                用 <M>{'\\varepsilon = 0.5'}</M> 的 ε-贪心跑到完全收敛，
                得到的策略和真正的最优策略相比会怎样？
              </>
            }
            options={[
              { id: 'a', label: '完全一样 —— ε 只影响探索，不影响最终结果' },
              { id: 'b', label: '差不多，只有个别格子不同' },
              { id: 'c', label: '明显不同：好几个格子的最优动作都变了' },
            ]}
            answer="c"
            explain={
              <>
                <p>
                  把右边的滑块拖到 0.5，看看那个数字。ε 一大，
                  <strong>「最优」的定义本身就变了</strong> ——
                  你在优化的不再是「最优策略」，而是「所有 ε-贪心策略里最优的那个」。
                </p>
                <p>
                  直觉上也讲得通：既然有一半概率会乱走，那么贴着禁区走就变得很危险，
                  最优的 ε-贪心策略会主动绕远路。ε 越大，它越怂。
                </p>
                <p>
                  工程上的标准做法因此是<strong>让 ε 随时间衰减</strong>：
                  早期大胆探索，后期收敛到贪心。
                </p>
              </>
            }
          />
        </Beat>

        <Beat id="d3">
          <Callout tone="rigor" title="一致性（consistency）">
            <p>
              严格地说，只有当 <M>{'\\varepsilon'}</M> 足够小时，
              最优 ε-贪心策略的贪心动作才和最优策略一致。
              曲线在小 ε 段那一段平台，就是这个「足够小」的可视化 ——
              它的宽度取决于最优动作和次优动作之间 q 值的差距。
            </p>
          </Callout>
        </Beat>
      </Act>

      {/* ───────────────────── 第 4 幕 ───────────────────── */}
      <Act
        id="a4"
        no="第 4 幕"
        title="跑一遍，然后看清它的软肋"
        goal="完整的 MC ε-贪心实现，以及它无法回避的两个结构性缺陷。"
        minutes={8}
        points={[
          <>
            完整算法只有二十行：采样一条轨迹 → 从后往前算回报 → 增量更新 q → 重新贪心。
          </>,
          <>
            软肋一：<strong>必须等回合结束</strong>。持续型任务里根本没有「结束」。
          </>,
          <>
            软肋二：<strong>方差大</strong>。回报是几十个随机量之和，抖得厉害。
          </>,
        ]}
        stage={() => (
          <div className="space-y-4">
            <Panel title="学习曲线（每回合累计奖励，滑动平均）">
              <LineChart
                height={186}
                xLabel="回合"
                series={[
                  { name: '原始', color: C.gamma, data: mc.episodeReturn, width: 0.8 },
                  { name: '滑动平均', color: C.accent, data: smooth(mc.episodeReturn, 40) },
                ]}
              />
              <div className="mt-3">
                <Slider
                  label="总回合数"
                  value={episodes}
                  min={100}
                  max={3000}
                  step={100}
                  onChange={setEpisodes}
                  format={(v) => String(v)}
                  hint="拖大它，看那条灰线抖得多厉害 —— 那就是蒙特卡洛的方差"
                />
              </div>
            </Panel>
            <Panel title="学到的 vs 真正的最优">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <div className="mb-1.5 text-center text-[11px] text-faint">MC 学到的</div>
                  <GridWorld mdp={mdp} policy={mc.policy} showValues={false} cell={46} quiet />
                </div>
                <div>
                  <div className="mb-1.5 text-center text-[11px] text-faint">真正的最优</div>
                  <GridWorld mdp={mdp} policy={star.policy} showValues={false} cell={46} quiet />
                </div>
              </div>
            </Panel>
          </div>
        )}
      >
        <Beat id="e1" keep>
          <Code
            code={`def mc_epsilon_greedy(env, gamma=0.9, eps=0.1, episodes=1000):
    q   = [[0.0] * env.nA for _ in range(env.nS)]
    cnt = [[0]   * env.nA for _ in range(env.nS)]

    for _ in range(episodes):
        # ① 用当前的 ε-贪心策略采一条轨迹
        traj, s = [], env.reset()
        for _ in range(env.horizon):
            a = eps_greedy(q[s], eps)
            sp, r, done = env.step(s, a)
            traj.append((s, a, r))
            s = sp
            if done: break

        # ② 从后往前算回报，顺手把每个 (s,a) 都更新掉
        G = 0.0
        for s, a, r in reversed(traj):
            G = gamma * G + r
            cnt[s][a] += 1
            q[s][a] += (G - q[s][a]) / cnt[s][a]   # 增量式均值

    return q`}
          />
        </Beat>

        <Beat id="e2" keep>
          <p>
            注意最后那一行 <code>q += (G - q) / cnt</code>。
            它把「求平均」写成了<strong>增量式</strong>的样子 —— 不用存所有历史样本，
            来一个更新一个。
          </p>
          <MB>{'q_{k+1} = q_k + \\frac{1}{k}\\left( G_k - q_k \\right)'}</MB>
          <p>
            这一行会在下一章被单独拎出来放大来看。
            因为它其实是一个更普遍的东西的特例，而那个东西，
            会在第 7 章直接长成 TD 算法。
          </p>
        </Beat>

        <Beat id="e3" keep>
          <Callout tone="trap" title="蒙特卡洛的两个结构性缺陷">
            <p>
              <strong>一、必须等回合结束。</strong>
              <M>{'G_t'}</M> 的定义要求你知道整条轨迹。持续型任务（比如一个永不停机的推荐系统）
              里没有「结束」这回事，蒙特卡洛无从下手。
            </p>
            <p>
              <strong>二、方差大。</strong>
              右上那条灰线就是证据。<M>{'G'}</M> 是几十个随机奖励的加权和，
              每一个都在贡献方差。想压下去只能靠更多样本，而样本是要花钱买的。
            </p>
            <p>
              这两个缺陷指向同一个改进方向：<em>能不能不等回合结束，走一步就更新一次？</em>
              走一步只有一个随机奖励，方差自然小。但一步之后的未来怎么算？
              —— 用<strong>估计值</strong>去顶。
            </p>
            <p>
              这个「用估计更新估计」的想法既大胆又可疑。在把它变成算法之前，
              下一章要先把它的数学许可证办下来。
            </p>
          </Callout>
        </Beat>
      </Act>
    </div>
  )
}
