import { useMemo, useState } from 'react'
import { ACTIONS, buildGridMDP, classicGrid, stepResult } from '../core/mdp'
import { uniformPolicy } from '../core/policy'
import { Act, Beat } from '../narrative/Act'
import { ChapterGlance, ChapterHero, type Glance } from '../narrative/ChapterShell'
import { PredictChoice, PredictNumber } from '../narrative/Predict'
import { LiveFormula } from '../formula/LiveFormula'
import { useFormulaCtx } from '../formula/core'
import { returnDefinition } from '../formula/bellman'
import { GridWorld } from '../viz/GridWorld'
import { LineChart } from '../viz/LineChart'
import { Callout, Code, Details, M, MB, Panel, Seg, Slider } from '../ui/prims'
import { fmt, useColors } from '../theme'

const RETURN_DEF = returnDefinition()

const GLANCE: Glance = {
  formula: String.raw`G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots
  = \sum_{k=0}^{\infty}\gamma^{k} R_{t+k+1}`,
  formulaNote:
    '这就是强化学习真正要最大化的东西。注意它是一整条未来的加权总账，不是眼前那一步的奖励 —— 这一字之差是本章最值钱的东西。',
  takeaways: [
    <>
      一个强化学习问题 = <strong>五元组</strong>{' '}
      <M>{'(\\mathcal S, \\mathcal A, p(s\'|s,a), p(r|s,a), \\gamma)'}</M>，
      也就是 MDP。
    </>,
    <>
      策略 <M>{'\\pi(a|s)'}</M> 是<strong>概率分布</strong>；
      确定性策略只是它的一个特例。
    </>,
    <>
      <strong>奖励 ≠ 目标。</strong>目标是回报 <M>{'G_t'}</M> 的期望。
      设计奖励是在说明「什么算好」，而不是在教「该怎么走」。
    </>,
    <>
      <M>{'\\gamma'}</M> 同时是<strong>收敛条件</strong>和<strong>耐心程度</strong>，
      有效视野约 <M>{'1/(1-\\gamma)'}</M> 步。
    </>,
  ],
  traps: [
    <>
      把「最大化奖励」挂在嘴上。正确的说法永远是<strong>最大化期望回报</strong>。
    </>,
    <>
      以为马尔可夫性是环境的客观性质。它是<strong>你对状态的定义</strong>是否够用的问题。
    </>,
    <>
      把 <M>{'\\gamma'}</M> 当成随便调调的超参数。它一改，
      <strong>最优策略本身就变了</strong>（第 3 章会亲眼看到箭头翻转）。
    </>,
  ],
}

interface Step {
  s: number
  a: number
  r: number
  next: number
}

const START = 0

function useDrive() {
  const grid = useMemo(() => classicGrid(), [])
  const mdp = useMemo(() => buildGridMDP(grid), [grid])
  const [steps, setSteps] = useState<Step[]>([])

  const at = steps.length ? steps[steps.length - 1].next : START

  const go = (a: number) => {
    const { next, reward } = stepResult(grid, at, a)
    setSteps((prev) => [...prev, { s: at, a, r: reward, next }].slice(-40))
  }
  const reset = () => setSteps([])

  return { grid, mdp, steps, at, go, reset }
}

export function Chapter1() {
  const C = useColors()
  const { mdp, steps, at, go, reset } = useDrive()
  const pi = useMemo(() => uniformPolicy(mdp.nS), [mdp.nS])
  const [gamma, setGamma] = useState(0.9)
  const [returnMode, setReturnMode] = useState<'plain' | 'discounted'>('plain')

  const rewards = steps.map((x) => x.r)
  const demoRewards = rewards.length ? rewards : [0, 0, -1, 0, 1, 1, 1, 1, 1, 1]

  const cum = useMemo(() => {
    const out: number[] = []
    let acc = 0
    demoRewards.forEach((r, k) => {
      acc += (returnMode === 'discounted' ? Math.pow(gamma, k) : 1) * r
      out.push(acc)
    })
    return out
  }, [demoRewards, gamma, returnMode])

  const trail = steps.map((x) => x.s)
  const zeros = useMemo(() => new Array<number>(mdp.nS).fill(0), [mdp.nS])
  const ctx = useFormulaCtx(mdp, pi, gamma, zeros, at)

  return (
    <div>
      <ChapterHero
        n={1}
        hook="我们到底在优化什么？"
        lead={
          <>
            <p>
              强化学习的所有数学，都是为了把下面这句大白话说清楚：
              <strong>一个东西在一个环境里反复试，越试越会做事。</strong>
              这一章就是做翻译工作 —— 把「东西」「环境」「会做事」逐个变成有定义的符号。
            </p>
            <p>
              翻译听起来枯燥，但这一章有一个真正反直觉的结论藏在里面：
              <em>奖励并不是目标</em>。很多人学到第五章还在这里栽跟头。
            </p>
          </>
        }
        gains={[
          '能用 MDP 的语言描述任何一个强化学习问题',
          '说清楚「奖励」和「回报」到底差在哪',
          '理解折扣因子 γ 不是超参数，而是智能体的耐心',
          '知道马尔可夫性为什么是一个「假设」而不是「事实」',
        ]}
      />

      <ChapterGlance g={GLANCE} />

      {/* ───────────────────────── 第 1 幕 ───────────────────────── */}
      <Act
        id="a1"
        no="第 1 幕"
        title="先别看定义，你自己走一遍"
        goal="状态、动作、策略这三个词，各自到底指的是画面上的什么东西。"
        minutes={9}
        points={[
          <>
            <strong>状态</strong>是「我在哪」，<strong>动作</strong>是「我能干什么」，
            <strong>策略</strong> <M>{'\\pi(a|s)'}</M> 是「在每个状态上各以多大概率做什么」。
          </>,
          <>
            策略是一个<strong>概率分布</strong>，不是一张「该走哪」的表。
            确定性策略只是概率全押在一个动作上的特例。
          </>,
          <>
            奖励由<strong>「进入了什么格子」</strong>决定，撞墙也算一次转移（停在原地并受罚）。
          </>,
        ]}
        stage={() => (
          <Panel
            title="你就是智能体"
            right={
              <button
                onClick={reset}
                className="rounded-lg border border-line px-2.5 py-1 text-[11.5px] text-dim hover:text-ink"
              >
                重来
              </button>
            }
          >
            <GridWorld
              mdp={mdp}
              policy={undefined}
              showValues={false}
              showHeatmap={false}
              showLabels
              agent={at}
              trail={trail}
              cell={54}
            />
            <div className="mt-4 flex flex-col items-center gap-2">
              <div className="flex gap-2">
                {[0, 1, 2, 3, 4].map((a) => (
                  <button
                    key={a}
                    onClick={() => go(a)}
                    className="flex h-10 w-10 items-center justify-center rounded-xl border border-line text-[16px] text-ink transition-all hover:border-brand/50 hover:bg-brand/10 hover:text-brand active:scale-95"
                    title={ACTIONS[a].name}
                  >
                    {ACTIONS[a].glyph}
                  </button>
                ))}
              </div>
              <div className="font-mono text-[12px] text-faint">
                已走 {steps.length} 步 · 累计奖励{' '}
                <span className="text-warn">{fmt(rewards.reduce((x, y) => x + y, 0))}</span>
              </div>
            </div>

            {steps.length > 0 && (
              <div className="mt-4 max-h-36 overflow-y-auto rounded-xl border border-line bg-surface2 p-3">
                <div className="font-mono text-[11.5px] leading-relaxed text-dim">
                  {steps.map((x, i) => (
                    <div key={i} className="flex gap-2">
                      <span className="text-faint">t={i}</span>
                      <span style={{ color: C.state }}>s{x.s + 1}</span>
                      <span style={{ color: C.policy }}>{ACTIONS[x.a].glyph}</span>
                      <span className="text-faint">→</span>
                      <span style={{ color: C.state }}>s{x.next + 1}</span>
                      <span style={{ color: C.reward }}>r={fmt(x.r)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </Panel>
        )}
      >
        <Beat id="b1">
          <p>
            右边是一个 5×5 的网格。青色的点是你，虚线框的金色格子是目标，斜纹格子是禁区
            —— 可以进，但会挨罚。<strong>先别往下读，用下面五个按钮走几步。</strong>
          </p>
          <p>
            你在走的时候，脑子里其实已经在用三个概念了，只是还没给它们起名字。
          </p>
        </Beat>

        <Beat id="b2">
          <h3>状态：你现在在哪</h3>
          <p>
            每一格就是一个<strong>状态</strong>，记作 <M>{'s'}</M>。这个世界一共 25 个状态，
            它们组成<strong>状态空间</strong> <M>{'\\mathcal{S} = \\{s_1, \\dots, s_{25}\\}'}</M>。
          </p>
          <p>
            注意「状态」是一个抽象概念，它不一定是位置。下棋时状态是棋盘布局，
            开车时状态可能是速度、方向、周围车辆的组合。
            状态的定义权在你手里 —— <em>状态该包含什么，是建模者的选择，而这个选择的好坏会决定整个问题的难度</em>。
          </p>
        </Beat>

        <Beat id="b3">
          <h3>动作：你能做什么</h3>
          <p>
            在每个状态下可以采取的操作是<strong>动作</strong> <M>{'a'}</M>。
            这里有五个：<M>{'a_1'}</M> 上、<M>{'a_2'}</M> 右、<M>{'a_3'}</M> 下、
            <M>{'a_4'}</M> 左、<M>{'a_5'}</M> 原地不动。
          </p>
          <p>
            做完动作，世界会给你两样东西：一个新状态，和一个数。
            新状态由<strong>状态转移</strong>决定，那个数叫<strong>奖励</strong> —— 下一幕的主角。
          </p>
          <Callout tone="trap" title="第一个容易被略过的细节">
            撞墙的时候，你并没有离开原地，但你确实<strong>采取了</strong>一个动作，
            也确实<strong>收到了</strong>一个奖励（这里是 <M>{'-1'}</M>）。
            「动作没起效」和「没有动作」是两回事。
          </Callout>
        </Beat>

        <Beat id="b4">
          <h3>策略：你按什么规矩走</h3>
          <p>
            刚才每一步都是你自己拍脑袋决定的。如果把「在什么状态下该怎么选」写成一条固定的规矩，
            这条规矩就是<strong>策略</strong> <M>{'\\pi'}</M>。
          </p>
          <p>
            数学上，策略是一个条件概率分布：<M>{'\\pi(a \\mid s)'}</M> 表示在状态 <M>{'s'}</M> 下
            选择动作 <M>{'a'}</M> 的概率，且对每个 <M>{'s'}</M> 都有{' '}
            <M>{'\\sum_a \\pi(a\\mid s) = 1'}</M>。
          </p>
          <ul>
            <li>
              <strong>确定性策略</strong>：某个动作概率为 1，其余为 0。画在网格上就是一个箭头。
            </li>
            <li>
              <strong>随机性策略</strong>：多个动作各有概率。画在网格上是几个粗细不同的箭头。
            </li>
          </ul>

          <PredictChoice
            id="ch1-policy-1"
            question={
              <>
                如果一个策略在每个状态都以 <M>{'1/5'}</M> 的概率随机选五个动作之一，
                它算不算一个「合法的策略」？
              </>
            }
            options={[
              { id: 'a', label: '不算。策略必须能把智能体带到目标才叫策略。' },
              { id: 'b', label: '算。策略只是「怎么选动作」的规定，好坏是另一回事。' },
              { id: 'c', label: '不算。随机的东西不能叫策略，策略必须是确定的。' },
            ]}
            answer="b"
            explain={
              <>
                策略这个词在强化学习里是<strong>中性的</strong>：任何一个从状态到动作分布的映射都是策略，
                哪怕它蠢到原地转圈。「好」与「坏」需要一个打分标准 ——
                而那个标准正是下一章要建立的东西。把「是不是策略」和「是不是好策略」分开，
                是后面所有推导的前提。
              </>
            }
          />
        </Beat>
      </Act>

      {/* ───────────────────────── 第 2 幕 ───────────────────────── */}
      <Act
        id="a2"
        no="第 2 幕"
        title="奖励不是目标"
        goal="为什么「最大化奖励」这句话是错的，正确的说法是「最大化回报」。"
        points={[
          <>
            奖励 <M>{'R_{t+1}'}</M> 是<strong>一步</strong>的反馈；
            回报 <M>{'G_t = R_{t+1}+\\gamma R_{t+2}+\\cdots'}</M> 是<strong>一辈子</strong>的总账。
          </>,
          <>
            强化学习最大化的是<strong>回报的期望</strong>，不是当下的奖励。
            贪图眼前的奖励往往是最差的策略。
          </>,
          <>
            奖励设计是在<strong>说明「什么算好」</strong>，不是在<strong>教「该怎么做」</strong>。
            后者是算法的活。
          </>,
        ]}
        minutes={11}
        stage={() => (
          <Panel
            title="奖励序列与回报"
            right={
              <Seg
                size="sm"
                value={returnMode}
                onChange={setReturnMode}
                options={[
                  { value: 'plain', label: '不折扣' },
                  { value: 'discounted', label: '折扣' },
                ]}
              />
            }
          >
            <div className="mb-3 flex flex-wrap gap-1.5">
              {demoRewards.map((r, i) => (
                <div
                  key={i}
                  className="flex h-9 w-9 items-center justify-center rounded-lg border font-mono text-[12px]"
                  style={{
                    borderColor:
                      r > 0
                        ? 'color-mix(in srgb, var(--target) 45%, transparent)'
                        : r < 0
                          ? 'color-mix(in srgb, var(--danger) 45%, transparent)'
                          : 'var(--line)',
                    color: r > 0 ? C.target : r < 0 ? C.danger : 'var(--ink-faint)',
                    background: r !== 0 ? 'var(--surface-2)' : 'transparent',
                  }}
                  title={`R_${i + 1} = ${r}`}
                >
                  {fmt(r)}
                </div>
              ))}
            </div>
            <LineChart
              series={[
                {
                  name: returnMode === 'plain' ? '累计奖励 ΣR' : '折扣回报 Σγᵏ R',
                  color: C.value,
                  data: cum,
                },
              ]}
              xLabel="时间步 k"
              height={190}
            />
            <div className="mt-3">
              <Slider
                label={
                  <>
                    折扣因子 <M>{'\\gamma'}</M>
                  </>
                }
                value={gamma}
                min={0}
                max={0.99}
                step={0.01}
                onChange={setGamma}
                accent={C.gamma}
                hint={
                  returnMode === 'plain'
                    ? '当前是「不折扣」模式，γ 还不起作用。切到右边看看。'
                    : `有效视野约 1/(1-γ) ≈ ${fmt(1 / (1 - Math.min(gamma, 0.98)), 1)} 步`
                }
              />
            </div>
            <p className="mt-3 text-[11.5px] leading-relaxed text-faint">
              上面这串奖励来自你在第 1 幕走出来的轨迹；还没走过的话，用的是一条示范轨迹。
            </p>
          </Panel>
        )}
      >
        <Beat id="c1">
          <p>
            你每走一步，世界都会回你一个数：进目标 <M>{'+1'}</M>，进禁区 <M>{'-1'}</M>，
            撞墙 <M>{'-1'}</M>，其余 <M>{'0'}</M>。这个数就是<strong>奖励</strong> <M>{'r'}</M>。
          </p>
          <p>
            奖励是人设计的，它是我们和智能体唯一的沟通渠道 ——
            <em>你不是在告诉它怎么做，你是在告诉它什么算好</em>。这也是为什么奖励设计经常比算法本身更难。
          </p>
        </Beat>

        <Beat id="c2">
          <h3>一次尝试的完整记录：轨迹</h3>
          <p>
            把「状态—动作—奖励」按时间串起来，就是一条<strong>轨迹</strong>：
          </p>
          <MB>{'s_1 \\xrightarrow{\\;a_2,\\, r=0\\;} s_2 \\xrightarrow{\\;a_3,\\, r=-1\\;} s_7 \\xrightarrow{\\;a_3,\\, r=1\\;} s_{12} \\cdots'}</MB>
          <p>
            这条轨迹上所有奖励的和，叫做<strong>回报</strong>（return）。这才是我们真正要最大化的东西。
          </p>
        </Beat>

        <Beat id="c3">
          <Callout tone="insight" title="奖励 ≠ 目标">
            <p>
              一步的奖励只描述「这一步好不好」。但一步走得好，不代表整条路走得好 ——
              为了绕开禁区多走两格，这两步的奖励是 <M>{'0'}</M>，可它避免了后面的 <M>{'-1'}</M>。
            </p>
            <p>
              所以强化学习最大化的是<strong>回报</strong>，不是奖励。
              一个只会最大化眼前奖励的智能体，叫贪心，不叫智能。
            </p>
          </Callout>

          <PredictChoice
            id="ch1-reward-1"
            question={
              <>
                我们希望机器人尽快到达目标。有人提议：把「每走一步」的奖励从{' '}
                <M>{'0'}</M> 改成 <M>{'-0.1'}</M>，这样它就会想快点结束。这个改法会怎样？
              </>
            }
            options={[
              { id: 'a', label: '没用。加一个常数不改变任何策略的相对好坏。' },
              { id: 'b', label: '有用。每走一步都亏一点，绕路就变得更不划算了。' },
              { id: 'c', label: '有害。负奖励会让机器人拒绝行动，直接原地不动。' },
            ]}
            answer="b"
            explain={
              <>
                <p>
                  有用。这是最常见的「时间惩罚」技巧：<strong>只给终点奖励时，早到和晚到的回报一样</strong>
                  （在不折扣的意义下），智能体没有理由着急。加上每步的小负值之后，
                  路径长度本身就进入了目标函数。
                </p>
                <p>
                  选 A 的直觉其实指向一个真实且重要的定理，只是用错了地方：
                  <strong>对奖励做仿射变换 <M>{'r \\to \\alpha r + \\beta\\ (\\alpha>0)'}</M> 确实不改变最优策略</strong>。
                  但这里只改了「普通格子」的奖励，没有同步改目标和禁区的，
                  所以它不是仿射变换，而是实实在在改变了问题。这个定理我们会在第 3 章亲手验证。
                </p>
              </>
            }
          />
        </Beat>

        <Beat id="c4">
          <h3>可是，无限走下去怎么办</h3>
          <p>
            机器人到达目标后不会消失，它可以停在那里，每一步都拿 <M>{'+1'}</M>。
            那么这条轨迹的回报是：
          </p>
          <MB>{'G = 1 + 1 + 1 + 1 + \\cdots = \\infty'}</MB>
          <p>
            无穷。而另一个策略先绕了一大圈再停在目标上，它的回报也是无穷。
            <strong>两个无穷没法比大小，我们的打分体系当场崩溃。</strong>
          </p>
          <Callout tone="question" title="停下来想一秒">
            数学上出问题的地方，往往正是概念上没想清楚的地方。
            这里真正的问题不是「和发散了」，而是我们还没回答：
            <strong>一百步之后的一块钱，和现在的一块钱，是一回事吗？</strong>
          </Callout>
        </Beat>
      </Act>

      {/* ───────────────────────── 第 3 幕 ───────────────────────── */}
      <Act
        id="a3"
        no="第 3 幕"
        title="γ 不是超参数，是耐心"
        goal="折扣因子同时做了两件事：让数学收敛，以及定义智能体的时间偏好。"
        points={[
          <>
            数学面：<M>{'\\gamma<1'}</M> 保证无穷和收敛（
            <M>{'\\sum\\gamma^k = 1/(1-\\gamma)'}</M>），否则回报可能是无穷大，没法比较。
          </>,
          <>
            行为面：<M>{'1/(1-\\gamma)'}</M> 大致是智能体的<strong>有效视野</strong>。
            γ=0.9 约看十步，γ=0.99 约看一百步。
          </>,
          <>
            所以调 γ 不是调精度，是<strong>换一个性格</strong>：
            短视的会绕开风险，有耐心的敢抄近道。
          </>,
        ]}
        minutes={9}
        stage={() => {
          const K = 40
          const decay = Array.from({ length: K }, (_, k) => Math.pow(gamma, k))
          const partial = Array.from({ length: K }, (_, k) =>
            gamma >= 1 ? k + 1 : (1 - Math.pow(gamma, k + 1)) / (1 - gamma),
          )
          return (
            <Panel title="γ 的两副面孔">
              <div className="mb-1 text-[11.5px] text-faint">
                第 k 步之后的奖励，在今天看来还剩多少权重
              </div>
              <LineChart
                series={[{ name: 'γᵏ', color: C.gamma, data: decay }]}
                xLabel="k 步之后"
                height={150}
                yMin={0}
                yMax={1}
              />
              <div className="mt-4 mb-1 text-[11.5px] text-faint">
                若每一步都拿 +1，前 k 步累积的折扣回报
              </div>
              <LineChart
                series={[{ name: 'Σ γᵏ', color: C.value, data: partial }]}
                xLabel="k"
                height={150}
              />
              <div className="mt-4 grid grid-cols-2 gap-3">
                <div className="rounded-xl border border-line bg-surface2 p-3">
                  <div className="text-[11px] text-faint">收敛上界 1/(1−γ)</div>
                  <div className="mt-1 font-mono text-[19px]" style={{ color: C.value }}>
                    {gamma >= 0.995 ? '∞' : fmt(1 / (1 - gamma), 2)}
                  </div>
                </div>
                <div className="rounded-xl border border-line bg-surface2 p-3">
                  <div className="text-[11px] text-faint">看得见的未来 ≈</div>
                  <div className="mt-1 font-mono text-[19px]" style={{ color: C.gamma }}>
                    {gamma >= 0.995 ? '∞' : fmt(1 / (1 - gamma), 0)} 步
                  </div>
                </div>
              </div>
              <div className="mt-4">
                <Slider
                  label={<M>{'\\gamma'}</M>}
                  value={gamma}
                  min={0}
                  max={0.99}
                  step={0.01}
                  onChange={setGamma}
                  accent={C.gamma}
                />
              </div>
            </Panel>
          )
        }}
      >
        <Beat id="d1">
          <p>
            解决办法出人意料地朴素：<strong>越远的奖励，打的折越狠。</strong>
            引入一个 <M>{'\\gamma \\in [0,1)'}</M>，把回报重新定义为
          </p>
          <div className="rounded-2xl border border-line bg-surface px-4 py-1">
            <LiveFormula node={RETURN_DEF} ctx={ctx} />
          </div>
          <p>
            <M>{'R_{t+1}'}</M> 原封不动，<M>{'R_{t+2}'}</M> 打 <M>{'\\gamma'}</M> 折，
            <M>{'R_{t+3}'}</M> 打 <M>{'\\gamma^2'}</M> 折 …… 于是即使奖励永远拿下去，
            总和也是有限的。
          </p>
        </Beat>

        <Beat id="d2">
          <h3>第一副面孔：让数学能算</h3>
          <p>
            如果每步奖励的绝对值不超过 <M>{'R_{\\max}'}</M>，那么
          </p>
          <MB>{'|G_t| \\le \\sum_{k=0}^{\\infty} \\gamma^k R_{\\max} = \\frac{R_{\\max}}{1-\\gamma} < \\infty'}</MB>
          <p>
            一个几何级数把无穷变成了有限数。这不只是「技术处理」——
            后面第 2、3 章能证明贝尔曼方程有唯一解，靠的正是这个 <M>{'\\gamma < 1'}</M>。
            <em>它是整座大厦的地基螺栓。</em>
          </p>
        </Beat>

        <Beat id="d3">
          <h3>第二副面孔：决定智能体有多远视</h3>
          <p>
            把 <M>{'\\gamma'}</M> 调到 <M>{'0'}</M>，回报就退化成 <M>{'R_{t+1}'}</M>：
            智能体只看下一步，彻底短视。把 <M>{'\\gamma'}</M> 推向 <M>{'1'}</M>，
            几十步以后的事情仍然清清楚楚地记在账上。
          </p>
          <p>
            一个好用的心算法则：<strong>智能体大致能「看见」<M>{'1/(1-\\gamma)'}</M> 步以内的未来。</strong>
            <M>{'\\gamma = 0.9'}</M> 约 10 步，<M>{'\\gamma = 0.99'}</M> 约 100 步。
          </p>

          <PredictNumber
            id="ch1-gamma-sum"
            question={
              <>
                智能体停在目标格上不动，每一步拿 <M>{'+1'}</M>，一直到永远。
                取 <M>{'\\gamma = 0.9'}</M>，它的折扣回报 <M>{'G'}</M> 是多少？
              </>
            }
            min={0}
            max={30}
            step={0.5}
            truth={10}
            tolerance={0.75}
            explain={
              <>
                <p>
                  <M>{'G = 1 + 0.9 + 0.9^2 + \\cdots = \\frac{1}{1-0.9} = 10'}</M>。
                </p>
                <p>
                  记住这个 10。等你在第 2、3 章看到目标格的价值稳定在 10 附近时，
                  你会立刻知道那不是巧合 —— <strong>它就是这个几何级数</strong>。
                  很多人算了一整章数值，都没意识到这两个 10 是同一个东西。
                </p>
              </>
            }
          />
        </Beat>

        <Beat id="d4">
          <Details summary="为什么不能直接取 γ = 1？（严格一点的说法）">
            <p>
              可以，但要付出代价。<M>{'\\gamma = 1'}</M> 的情形叫「无折扣」，
              此时只有在<strong>幕式任务</strong>（episodic task，轨迹必然在有限步内终止）下
              回报才保证有限。上面那个「永远停在目标上」的例子恰恰不是幕式任务，
              所以 <M>{'\\gamma = 1'}</M> 直接失效。
            </p>
            <p>
              对于持续型任务，除了折扣之外还有「平均奖励」等其它准则，
              但它们的理论会复杂得多。本书主线始终用 <M>{'0 \\le \\gamma < 1'}</M>，
              就是为了让压缩映射这把利器一直可用。
            </p>
          </Details>
        </Beat>
      </Act>

      {/* ───────────────────────── 第 4 幕 ───────────────────────── */}
      <Act
        id="a4"
        no="第 4 幕"
        title="把这一切装进一个盒子：MDP"
        goal="马尔可夫决策过程的五个部件，以及那个被叫做「假设」的关键性质。"
        points={[
          <>
            五个部件：状态集 <M>{'\\mathcal S'}</M>、动作集 <M>{'\\mathcal A'}</M>、
            转移 <M>{"p(s'|s,a)"}</M>、奖励 <M>{'p(r|s,a)'}</M>、折扣 <M>{'\\gamma'}</M>。
          </>,
          <>
            马尔可夫性：<M>{"p(s_{t+1}|s_t,a_t,\\text{历史}) = p(s_{t+1}|s_t,a_t)"}</M> ——
            <strong>未来只取决于现在</strong>。
          </>,
          <>
            它是<strong>假设</strong>不是事实。不满足时的通常做法是
            <em>把状态定义得更宽</em>（塞进更多历史），而不是换掉整套理论。
          </>,
        ]}
        minutes={8}
        stage={() => (
          <Panel title="这个世界的完整说明书">
            <GridWorld mdp={mdp} showValues={false} showHeatmap={false} showLabels cell={54} />
            <div className="mt-4 space-y-2 font-mono text-[12px]">
              {[
                ['状态空间 𝒮', `${mdp.nS} 个状态`],
                ['动作空间 𝒜(s)', `${mdp.nA} 个动作`],
                ['状态转移 p(s′|s,a)', '确定性：非 0 即 1'],
                ['奖励 p(r|s,a)', '确定性：由落点决定'],
                ['折扣因子 γ', fmt(gamma)],
              ].map(([k, v]) => (
                <div key={k} className="flex justify-between border-b border-line pb-1.5">
                  <span className="text-faint">{k}</span>
                  <span className="text-dim">{v}</span>
                </div>
              ))}
            </div>
            <p className="mt-4 text-[11.5px] leading-relaxed text-faint">
              这五样东西一旦定下来，环境就被完全确定了。剩下唯一的自由度，就是策略 π。
            </p>
          </Panel>
        )}
      >
        <Beat id="e1">
          <p>
            把前面所有零件装进一个盒子，这个盒子叫<strong>马尔可夫决策过程</strong>（MDP）：
          </p>
          <ul>
            <li>
              <strong>集合</strong>：状态空间 <M>{'\\mathcal{S}'}</M>、动作空间{' '}
              <M>{'\\mathcal{A}(s)'}</M>、奖励集合 <M>{'\\mathcal{R}(s,a)'}</M>
            </li>
            <li>
              <strong>概率分布</strong>：状态转移 <M>{'p(s\' \\mid s,a)'}</M>、
              奖励分布 <M>{'p(r \\mid s,a)'}</M>
            </li>
            <li>
              <strong>策略</strong>：<M>{'\\pi(a \\mid s)'}</M>
            </li>
            <li>
              <strong>折扣</strong>：<M>{'\\gamma'}</M>
            </li>
          </ul>
          <p>
            前三项里，<em>只有策略是我们能改的</em>。其余全是环境的属性 ——
            这个分界线，就是整个强化学习的战场边界。
          </p>
        </Beat>

        <Beat id="e2">
          <h3>「马尔可夫」这三个字在说什么</h3>
          <p>马尔可夫性质是一个关于「记忆」的断言：</p>
          <MB>{'p(s_{t+1} \\mid s_t, a_t, s_{t-1}, a_{t-1}, \\dots, s_0, a_0) = p(s_{t+1} \\mid s_t, a_t)'}</MB>
          <p>
            <strong>未来只取决于现在，与你是怎么走到现在的无关。</strong>
            在网格世界里这显然成立：你在 <M>{'s_7'}</M> 往右走会到哪，
            和你之前绕了多大一圈毫无关系。
          </p>

          <PredictChoice
            id="ch1-markov"
            question="下面哪一种说法最准确？"
            options={[
              { id: 'a', label: '现实问题基本都满足马尔可夫性，所以 MDP 是通用模型。' },
              {
                id: 'b',
                label:
                  '马尔可夫性通常不天然成立，但可以通过把「历史」塞进状态定义里来人为地让它成立。',
              },
              { id: 'c', label: '不满足马尔可夫性的问题就无法用强化学习处理。' },
            ]}
            answer="b"
            explain={
              <>
                <p>
                  <strong>马尔可夫性是状态定义的性质，不是世界的性质。</strong>
                  同一个问题，状态定义得不好就不马尔可夫，定义得好就马尔可夫。
                </p>
                <p>
                  经典例子：只用一帧游戏画面当状态，你看不出小球在往哪个方向飞 ——
                  不马尔可夫。把连续四帧叠起来当状态，速度信息就有了 —— 马尔可夫了。
                  DQN 玩 Atari 时用的正是这一手。
                </p>
                <p>
                  所以当你的算法怎么调都学不好时，值得回头问一句：
                  <em>我的状态里，是不是漏了某个决定未来的信息？</em>
                </p>
              </>
            }
          />
        </Beat>

        <Beat id="e3">
          <h3>用代码说一遍</h3>
          <p>
            整个环境说到底就是两张表：转移表 <code>P[s][a][s']</code> 和奖励表{' '}
            <code>R[s][a]</code>。这一章的所有概念，写出来就是下面这几行。
          </p>
          <Code
            lang="python"
            code={`import numpy as np

ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1), (0, 0)]   # 上 右 下 左 原地

def build_grid_mdp(rows, cols, forbidden, target,
                   r_step=0.0, r_bound=-1.0, r_forbid=-1.0, r_target=1.0):
    nS, nA = rows * cols, len(ACTIONS)
    P = np.zeros((nS, nA, nS))     # p(s' | s, a)
    R = np.zeros((nS, nA))         # r(s, a)

    for s in range(nS):
        r, c = divmod(s, cols)
        for a, (dr, dc) in enumerate(ACTIONS):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                P[s, a, s] = 1.0           # 撞墙：留在原地
                R[s, a] = r_bound
                continue
            sp = nr * cols + nc
            P[s, a, sp] = 1.0
            R[s, a] = (r_target if sp == target else
                       r_forbid if sp in forbidden else r_step)
    return P, R

# 策略就是一个 (nS, nA) 的矩阵，每行求和为 1
pi_uniform = np.full((25, 5), 1 / 5)`}
          />
          <Callout tone="intuition">
            这一章翻译完了：世界是 <code>P</code> 和 <code>R</code>，
            我们能改的只有 <code>pi</code>，要最大化的是折扣回报。
            <strong>可是给我一个 <code>pi</code>，我该怎么算出它值多少分？</strong>
          </Callout>
        </Beat>
      </Act>
    </div>
  )
}
