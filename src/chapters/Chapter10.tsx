import { useMemo, useState } from 'react'
import { buildGridMDP, classicGrid } from '../core/mdp'
import { argmaxActions } from '../core/policy'
import { valueIterationSolve } from '../core/solvers'
import { mdpToEnv } from '../core/env'
import { smooth } from '../core/learn'
import { PG_LABEL, policyGradient, type PGVariant } from '../core/pg'
import { Act, Beat } from '../narrative/Act'
import { ChapterGlance, ChapterHero, type Glance } from '../narrative/ChapterShell'
import { PredictChoice } from '../narrative/Predict'
import { FrameworkMap } from '../narrative/FrameworkMap'
import { GridWorld } from '../viz/GridWorld'
import { LineChart } from '../viz/LineChart'
import { Scrubber } from '../ui/Scrubber'
import { Callout, Code, Details, M, MB, Panel, Seg, Slider } from '../ui/prims'
import { fmt, useColors } from '../theme'

const GLANCE: Glance = {
  formula: String.raw`\underbrace{\theta \leftarrow \theta + \alpha_\theta\, \delta_t\, \nabla_\theta \ln\pi(a_t|s_t,\theta)}_{\text{演员：改策略}}
  \qquad
  \underbrace{w \leftarrow w + \alpha_w\, \delta_t\, \nabla_w \hat v(s_t,w)}_{\text{评论家：改估值}}`,
  formulaNote:
    '同一个 TD 误差 δ 同时驱动两条更新。评论家用它修正自己的预测，演员用它判断刚才那个动作值不值得多做 —— 前面九章在这两行里合流。',
  takeaways: [
    <>
      Actor-Critic = 策略梯度 + 用 TD 估 <M>{'q'}</M>。
      它不是新方法，是把第 7 章的解药喂给第 9 章的病人。
    </>,
    <>
      A2C 里的 TD 误差 <M>{"\\delta = r+\\gamma \\hat v(s') - \\hat v(s)"}</M>{' '}
      <strong>就是优势函数的无偏估计</strong>：
      <M>{'\\mathbb{E}[\\delta|s,a] = q_\\pi(s,a)-v_\\pi(s)'}</M>。
    </>,
    <>
      所以「带基线的策略梯度」和「用 TD 误差的 Actor-Critic」是同一个式子的两种读法 ——
      只是把 <M>{'v'}</M> 从<em>基线</em>重新解释成了<em>评论家</em>。
    </>,
    <>
      现代方法（TRPO / PPO / GRPO）都在这个骨架上加一件事：
      <strong>限制每次策略更新的步子</strong>，防止一步走坏。
    </>,
  ],
  traps: [
    <>
      以为评论家越准越好。评论家<strong>有偏</strong>会让演员的梯度有偏 ——
      Actor-Critic 用偏差换方差，和 TD 换 MC 是同一笔交易。
    </>,
    <>
      把 A2C 的「优势」理解成「这个状态好不好」。它衡量的是
      <strong>这个动作比该状态的平均水平好多少</strong>，是动作的相对功劳。
    </>,
    <>
      忘了两个学习率要配合。评论家学得太慢，演员就在用垃圾信号更新；
      演员学得太快，评论家永远追不上一个在变的目标。
    </>,
  ],
}

const VARIANTS: PGVariant[] = ['reinforce', 'reinforce-baseline', 'qac', 'a2c']

export function Chapter10() {
  const C = useColors()
  const [gamma] = useState(0.9)
  const [variant, setVariant] = useState<PGVariant>('a2c')
  const [aTheta, setATheta] = useState(0.4)
  const [aW, setAW] = useState(0.2)
  const [episodes, setEpisodes] = useState(1200)
  const [k, setK] = useState(0)

  const mdp = useMemo(() => buildGridMDP(classicGrid()), [])
  const env = useMemo(() => mdpToEnv(mdp, { start: 0, horizon: 30 }), [mdp])
  const star = useMemo(() => valueIterationSolve(mdp, gamma), [mdp, gamma])
  const optActions = useMemo(() => argmaxActions(star.policy), [star])

  const runs = useMemo(
    () =>
      VARIANTS.map((v) => ({
        v,
        res: policyGradient(env, v, {
          gamma,
          alphaTheta: aTheta,
          alphaW: aW,
          episodes,
          seed: 33,
          probes: 40,
        }),
      })),
    [env, gamma, aTheta, aW, episodes],
  )
  const cur = runs.find((r) => r.v === variant)!.res
  const kk = Math.min(k, cur.snaps.length - 1)
  const snap = cur.snaps[kk]
  const wrongOf = (p: number[][]) =>
    argmaxActions(p).filter((a, i) => a !== optActions[i]).length

  const varOf = (xs: number[]) => {
    const t = xs.slice(-300)
    const m = t.reduce((a, b) => a + b, 0) / t.length
    return t.reduce((a, b) => a + (b - m) ** 2, 0) / t.length
  }
  const finalOf = (xs: number[]) => xs.slice(-200).reduce((a, b) => a + b, 0) / 200

  return (
    <div>
      <ChapterHero
        n={10}
        hook="梯度公式里那个 q_π(s,a)，谁来估？"
        lead={
          <>
            <p>
              整本书的最后一章，做的事其实只有一行代码那么多：
              把 REINFORCE 里的真实回报 <M>{'G_t'}</M>，换成一个
              <strong>专门养出来的估值函数</strong>。
            </p>
            <p>
              但这一行代码是九章的汇合处。<em>演员</em>是第 9 章的参数化策略，
              <em>评论家</em>是第 7、8 章的价值近似，它们之间传递的信号
              是第 6 章的随机近似形式，而那个信号本身 ——
              TD 误差 —— 来自第 2 章的贝尔曼公式。
            </p>
            <p>整本书在这里收口。</p>
          </>
        }
        gains={[
          '把「用什么估 q」这一个选择，展开成四个算法',
          '证明 TD 误差就是优势函数的无偏估计',
          '看清「带基线的 REINFORCE」与 A2C 其实是同一个式子',
          '亲手比较四种变体的收敛速度与方差',
          '知道 TRPO / PPO 在这个骨架上加了什么',
        ]}
      />

      <ChapterGlance g={GLANCE} />

      {/* ───────────────────── 第 1 幕 ───────────────────── */}
      <Act
        id="a1"
        no="第 1 幕"
        title="换掉那个 q"
        goal="策略梯度的骨架不变，只把 q 的估计方式换掉，就得到一整族算法。"
        minutes={9}
        points={[
          <>
            骨架永远是 <M>{'\\theta \\leftarrow \\theta + \\alpha\\, \\text{[系数]}\\, \\nabla\\ln\\pi'}</M>。
            四个算法只是那个「系数」不同。
          </>,
          <>
            <M>{'G_t'}</M> → REINFORCE；<M>{'G_t-v(s)'}</M> → 带基线；
            <M>{'\\hat q(s,a)'}</M> → QAC；<M>{'\\delta_t'}</M> → A2C。
          </>,
          <>
            换成 <M>{'\\hat q'}</M> 或 <M>{'\\delta'}</M> 之后，
            <strong>不必等回合结束</strong>了 —— 这就是 Actor-Critic 相对 REINFORCE 的根本优势。
          </>,
        ]}
        stage={() => (
          <div className="space-y-4">
            <Panel title="四个算法，一个骨架">
              <div className="space-y-2">
                {[
                  ['REINFORCE', 'G_t', '整条轨迹的真实回报', '无偏 · 方差最大 · 要等回合结束', C.qvalue],
                  ['+ 基线', 'G_t − v(s)', '减去状态的平均水平', '无偏 · 方差小 · 仍要等回合结束', C.reward],
                  ['QAC', 'q̂(s,a,w)', '评论家直接估 q', '有偏 · 方差小 · 可在线', C.policy],
                  ['A2C', "δ = r + γv̂(s′) − v̂(s)", 'TD 误差 = 优势的估计', '有偏 · 方差最小 · 可在线', C.value],
                ].map(([name, coef, what, trait, col]) => (
                  <div
                    key={name as string}
                    className="rounded-xl border px-3 py-2.5"
                    style={{
                      borderColor: `color-mix(in srgb, ${col as string} 35%, transparent)`,
                      background: `color-mix(in srgb, ${col as string} 6%, transparent)`,
                    }}
                  >
                    <div className="flex items-baseline gap-2.5">
                      <span className="text-[12.5px] font-semibold text-ink">{name}</span>
                      <span className="font-mono text-[11.5px]" style={{ color: col as string }}>
                        系数 = {coef}
                      </span>
                    </div>
                    <div className="mt-1 text-[11.5px] leading-relaxed text-dim">{what}</div>
                    <div className="mt-0.5 text-[11px] text-faint">{trait}</div>
                  </div>
                ))}
              </div>
            </Panel>
          </div>
        )}
      >
        <Beat id="b1" keep>
          <p>
            第 9 章的结论是 <M>{'\\nabla J = \\mathbb{E}[\\nabla\\ln\\pi \\cdot q_\\pi]'}</M>，
            而 <M>{'q_\\pi'}</M> 未知。REINFORCE 的答案是「用蒙特卡洛估」。
            但第 5 章末尾我们已经知道蒙特卡洛的毛病，第 7 章也已经给过解药。
          </p>
          <p>把解药直接搬过来：<strong>养一个函数来估 q</strong>。</p>
          <MB>
            {String.raw`\underbrace{\theta \leftarrow \theta + \alpha_\theta\,\hat q(s,a,w)\,\nabla_\theta\ln\pi(a|s,\theta)}_{\text{演员 Actor}}
            \qquad
            \underbrace{w \leftarrow w + \alpha_w\,\delta_t\,\nabla_w \hat q(s,a,w)}_{\text{评论家 Critic}}`}
          </MB>
          <p>
            演员负责动作，评论家负责打分。这个分工就是 Actor-Critic 这个名字的来历。
          </p>
        </Beat>

        <Beat id="b2" keep>
          <Callout tone="insight" title="它到底新在哪">
            <p>
              严格地说，Actor-Critic <strong>没有引入任何新数学</strong>。
              它是策略梯度（第 9 章）+ 值函数近似（第 8 章）+ TD（第 7 章）的组合。
            </p>
            <p>
              但组合本身是有价值的：<em>REINFORCE 必须等回合结束，
              而 Actor-Critic 走一步就能更新一次。</em>
              对于长回合、甚至永不结束的任务，这是从「不能用」到「能用」的差别。
            </p>
          </Callout>
        </Beat>
      </Act>

      {/* ───────────────────── 第 2 幕 ───────────────────── */}
      <Act
        id="a2"
        no="第 2 幕"
        title="TD 误差就是优势"
        goal="一行期望计算，把第 9 章的基线和第 7 章的 TD 误差焊在一起。"
        minutes={9}
        points={[
          <>
            优势函数 <M>{'A_\\pi(s,a) = q_\\pi(s,a) - v_\\pi(s)'}</M>：
            这个动作比该状态的平均水平好多少。
          </>,
          <>
            <M>{"\\mathbb{E}\\left[r+\\gamma v_\\pi(S')\\mid s,a\\right] = q_\\pi(s,a)"}</M>，
            所以 <M>{'\\mathbb{E}[\\delta_t|s,a] = A_\\pi(s,a)'}</M> —— <strong>TD 误差是优势的无偏估计</strong>。
          </>,
          <>
            于是只需要学 <M>{'v'}</M>，不需要学 <M>{'q'}</M>：
            参数少一个动作维度，而且 <M>{'\\delta'}</M> 顺手就算出来了。
          </>,
        ]}
        stage={() => (
          <div className="space-y-4">
            <Panel title="评论家学到的 v̂">
              <GridWorld
                mdp={mdp}
                v={snap.v ?? new Array(mdp.nS).fill(0)}
                policy={snap.policy}
                cell={52}
                quiet
              />
              <div className="mt-3">
                <Scrubber k={kk} setK={setK} max={cur.snaps.length - 1} label="快照" fps={6} />
              </div>
            </Panel>
            <Panel title="真实的 v*（对照）">
              <GridWorld mdp={mdp} v={star.v} policy={star.policy} cell={52} quiet />
              <p className="mt-2 text-center text-[11.5px] text-faint">
                评论家学的是<strong>当前策略</strong>的 v，不是 v*。
                只有演员收敛到最优，两张图才会重合。
              </p>
            </Panel>
          </div>
        )}
      >
        <Beat id="c1" keep>
          <p>
            第 9 章说过：减一个只依赖状态的基线不改变梯度的期望，
            而最自然的基线是 <M>{'v_\\pi(s)'}</M>。于是系数变成
          </p>
          <MB>{'A_\\pi(s,a) \\;\\doteq\\; q_\\pi(s,a) - v_\\pi(s)'}</MB>
          <p>
            这叫<strong>优势函数</strong>。它的含义很朴素：
            在状态 <M>{'s'}</M> 上，动作 <M>{'a'}</M> 比「随便按 π 走」好多少。
            正数就该多做，负数就该少做。
          </p>
        </Beat>

        <Beat id="c2" keep>
          <p>
            现在做一个观察。TD 误差是
            <M>{"\\delta_t = r_{t+1} + \\gamma v_\\pi(s_{t+1}) - v_\\pi(s_t)"}</M>。
            对它取条件期望：
          </p>
          <MB>
            {String.raw`\mathbb{E}\left[\delta_t \mid S_t=s, A_t=a\right]
            = \underbrace{\mathbb{E}\left[r+\gamma v_\pi(S') \mid s,a\right]}_{=\;q_\pi(s,a)\ \text{（贝尔曼公式）}} - v_\pi(s)
            = A_\pi(s,a)`}
          </MB>
          <p>
            <strong>TD 误差就是优势函数的无偏估计。</strong>
            这意味着我们根本不用去学 <M>{'q'}</M> —— 学一个 <M>{'v'}</M>，
            顺手算出的 <M>{'\\delta'}</M> 就是我们要的系数。
          </p>
        </Beat>

        <Beat id="c3" keep>
          <PredictChoice
            id="ch10-adv"
            question={
              <>
                A2C 用 <M>{'\\delta_t'}</M> 当系数。这个 <M>{'\\delta_t'}</M>
                同时也被评论家用来更新自己。用同一个数干两件事，会不会有问题？
              </>
            }
            options={[
              { id: 'a', label: '会 —— 两个更新互相干扰，必须用两个独立的样本' },
              { id: 'b', label: '不会 —— 它们更新的是不同的参数，而 δ 对两者都是正确的信号' },
              { id: 'c', label: '会 —— 所以实践中要用两条独立的轨迹' },
            ]}
            answer="b"
            explain={
              <>
                <p>
                  <M>{'\\delta_t'}</M> 对评论家而言是「我的预测错了多少」，
                  对演员而言是「刚才那个动作比预期好多少」。
                  <strong>同一个数，两种正确的解读。</strong>
                </p>
                <p>
                  两者更新的参数不同（<M>{'w'}</M> 与 <M>{'\\theta'}</M>），
                  所以不存在「同一个参数被更新两次」的问题。
                </p>
                <p>
                  真正需要小心的是<strong>学习率的配比</strong>。
                  评论家太慢，演员就在用一个还没学会的信号更新；
                  演员太快，评论家追着一个不断变化的策略跑，永远估不准。
                  右边那两个滑块可以亲手体会 —— 把 <M>{'\\alpha_w'}</M> 调到很小试试。
                </p>
              </>
            }
          />
        </Beat>

        <Beat id="c4">
          <Details summary="展开：为什么这也解释了「基线」的另一个身份">
            <p>
              第 9 章里 <M>{'v(s)'}</M> 的身份是<em>方差缩减用的基线</em>，
              是一个纯技术性的东西。这一章它的身份变成了<em>评论家</em>，
              是算法的一个核心组件。
            </p>
            <p>
              但式子一模一样。这种「同一个东西在不同视角下有不同身份」的现象，
              在强化学习里反复出现 —— <M>{'\\gamma'}</M> 既是数学收敛条件也是性格参数，
              <M>{'\\delta'}</M> 既是预测误差也是动作评价，
              <M>{'v'}</M> 既是基线也是评论家。
            </p>
            <p>
              能在这些身份之间自由切换，基本就说明这本书学通了。
            </p>
          </Details>
        </Beat>
      </Act>

      {/* ───────────────────── 第 3 幕 ───────────────────── */}
      <Act
        id="a3"
        no="第 3 幕"
        title="四个算法同场竞技"
        goal="同一个世界、同样的回合数，看清偏差-方差交换的实际后果。"
        minutes={10}
        points={[
          <>
            方差从大到小：REINFORCE → 带基线 → QAC → A2C。
          </>,
          <>
            但方差小不等于最终更好：<strong>评论家的偏差会传递给演员</strong>。
            这是一笔交易，不是免费午餐。
          </>,
          <>
            两个学习率必须配合。<M>{'\\alpha_w'}</M> 太小，演员就在用垃圾信号更新。
          </>,
        ]}
        stage={() => (
          <div className="space-y-4">
            <Panel title="学习曲线（每回合折扣回报，滑动平均）">
              <LineChart
                height={200}
                xLabel="回合"
                series={runs.map((r, i) => ({
                  name: { reinforce: 'REINFORCE', 'reinforce-baseline': '+基线', qac: 'QAC', a2c: 'A2C' }[
                    r.v
                  ],
                  color: [C.qvalue, C.reward, C.policy, C.value][i],
                  data: smooth(r.res.episodeReturn, 40),
                  width: r.v === variant ? 2.8 : 1.3,
                  dashed: r.v !== variant,
                }))}
              />
            </Panel>

            <Panel title="四个算法的成绩单">
              <div className="space-y-1.5">
                {runs.map((r, i) => (
                  <div
                    key={r.v}
                    className="grid grid-cols-[1fr_auto_auto_auto] items-center gap-3 rounded-lg border border-line bg-surface2 px-3 py-2 text-[11.5px]"
                  >
                    <span style={{ color: [C.qvalue, C.reward, C.policy, C.value][i] }}>
                      {{ reinforce: 'REINFORCE', 'reinforce-baseline': '+基线', qac: 'QAC', a2c: 'A2C' }[r.v]}
                    </span>
                    <span className="font-mono text-faint">
                      终值 {fmt(finalOf(r.res.episodeReturn), 2)}
                    </span>
                    <span className="font-mono text-faint">方差 {fmt(varOf(r.res.episodeReturn), 2)}</span>
                    <span
                      className="font-mono"
                      style={{ color: wrongOf(r.res.policy) ? C.danger : C.value }}
                    >
                      错 {wrongOf(r.res.policy)}
                    </span>
                  </div>
                ))}
              </div>
            </Panel>

            <Panel title="参数">
              <div className="space-y-3">
                <Seg
                  value={variant}
                  onChange={setVariant}
                  size="sm"
                  options={VARIANTS.map((v) => ({
                    value: v,
                    label: { reinforce: 'REINFORCE', 'reinforce-baseline': '+基线', qac: 'QAC', a2c: 'A2C' }[v],
                    hint: PG_LABEL[v],
                  }))}
                />
                <Slider
                  label={<>演员学习率 <M>{'\\alpha_\\theta'}</M></>}
                  value={aTheta}
                  min={0.05}
                  max={1.5}
                  step={0.05}
                  onChange={setATheta}
                  accent={C.policy}
                />
                <Slider
                  label={<>评论家学习率 <M>{'\\alpha_w'}</M></>}
                  value={aW}
                  min={0.01}
                  max={0.8}
                  step={0.01}
                  onChange={setAW}
                  accent={C.value}
                  hint="调到 0.01 试试：评论家跟不上，演员就在用垃圾信号更新"
                />
                <Slider
                  label="回合数"
                  value={episodes}
                  min={200}
                  max={4000}
                  step={200}
                  onChange={setEpisodes}
                  format={(v) => String(v)}
                />
              </div>
            </Panel>
          </div>
        )}
      >
        <Beat id="d1" keep>
          <Code
            code={`def a2c(env, gamma=0.9, a_theta=0.4, a_w=0.2, episodes=2000):
    theta = zeros(env.nS, env.nA)   # 演员：策略参数
    v     = zeros(env.nS)           # 评论家：状态价值

    for _ in range(episodes):
        s, disc = env.reset(), 1.0
        while True:
            p = softmax(theta[s])
            a = sample(p)
            sp, r, done = env.step(s, a)

            # 一个 δ，两处使用
            delta = r + (0 if done else gamma * v[sp]) - v[s]

            v[s] += a_w * delta                                  # 评论家：修正预测
            for k in range(env.nA):                              # 演员：调整策略
                theta[s][k] += a_theta * disc * delta * ((k == a) - p[k])

            if done: break
            s, disc = sp, disc * gamma
    return theta, v`}
          />
        </Beat>

        <Beat id="d2" keep>
          <p>
            请注意这段代码里<strong>没有轨迹缓存</strong>，也没有反向遍历。
            每走一步，立刻更新演员和评论家。
            对比第 9 章 REINFORCE 那段必须先攒完一整条轨迹的代码 ——
            这就是「在线」两个字的分量。
          </p>
        </Beat>

        <Beat id="d3" keep>
          <Callout tone="rigor" title="偏差从哪来">
            <p>
              A2C 的 <M>{'\\delta'}</M> 里用的是 <M>{'\\hat v(s,w)'}</M>，不是真的{' '}
              <M>{'v_\\pi(s)'}</M>。所以「TD 误差是优势的无偏估计」这句话，
              严格来说只在评论家已经准确时成立。
            </p>
            <p>
              评论家不准 → <M>{'\\delta'}</M> 有偏 → 演员的梯度有偏。
              这条链和第 7 章「TD 有偏而 MC 无偏」是同一件事，
              只不过偏差这次传给了策略。
            </p>
            <p>
              换来的是方差的大幅下降。<strong>这笔交易在绝大多数实际问题里是划算的</strong>
              —— 现代深度强化学习几乎清一色是 Actor-Critic 架构。
            </p>
          </Callout>
        </Beat>
      </Act>

      {/* ───────────────────── 第 4 幕：收口 ───────────────────── */}
      <Act
        id="a4"
        no="第 4 幕"
        title="回到那张地图"
        goal="十章走完，把全景图重新看一遍 —— 现在每个节点你都该有画面了。"
        minutes={8}
        points={[
          <>
            四个分岔点：<strong>模型知不知道</strong>、<strong>学价值还是学策略</strong>、
            <strong>用整条轨迹还是只用一步</strong>、<strong>表格装不装得下</strong>。
          </>,
          <>
            现代方法（TRPO / PPO / GRPO）在 A2C 骨架上加的是
            <strong>信任域</strong>：限制每次策略更新的幅度。
          </>,
          <>
            RLHF 用的就是这一支：奖励模型提供 <M>{'r'}</M>，PPO 提供优化器。
          </>,
        ]}
        stage={() => (
          <div className="space-y-4">
            <Panel title="接下来往哪走">
              <div className="space-y-2.5 text-[12.5px] leading-relaxed">
                {[
                  ['TRPO', '给策略更新加一个 KL 散度的信任域约束，保证单调改进', C.policy],
                  ['PPO', '把 TRPO 的约束换成一个便宜的裁剪目标，工程上更好用', C.accent],
                  ['GRPO', '去掉评论家，用同一个提示的一组采样互相当基线', C.value],
                  ['DDPG / SAC', '连续动作空间；确定性策略梯度 / 最大熵框架', C.reward],
                  ['MuZero', '把模型也学出来 —— 绕回第 4 章，但模型是学的', C.qvalue],
                ].map(([a, b, col]) => (
                  <div
                    key={a as string}
                    className="flex gap-3 rounded-lg border border-line bg-surface2 px-3 py-2"
                  >
                    <span
                      className="w-[80px] shrink-0 font-mono text-[11.5px]"
                      style={{ color: col as string }}
                    >
                      {a}
                    </span>
                    <span className="text-dim">{b}</span>
                  </div>
                ))}
              </div>
            </Panel>
          </div>
        )}
      >
        <Beat id="e1" keep>
          <p>
            十章走完了。现在再看一遍开头那张全景图 ——
            每一个节点上，你都应该能立刻说出「它解决了什么、留下了什么」。
          </p>
        </Beat>

        <Beat id="e2" keep>
          <div className="my-6">
            <FrameworkMap current="ch10" onGo={(id) => (location.hash = `#/${id}`)} />
          </div>
        </Beat>

        <Beat id="e3" keep>
          <p>
            最后说一句关于 PPO 的话，因为它是今天最常被提起的名字。
            PPO 的骨架就是 A2C，一个字都没变。它加的东西只有一句：
          </p>
          <MB>
            {String.raw`\max_\theta\; \mathbb{E}\left[\min\left(
            \rho_t A_t,\; \mathrm{clip}(\rho_t,\, 1-\epsilon,\, 1+\epsilon)\, A_t
            \right)\right], \qquad \rho_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}`}
          </MB>
          <p>
            那个 <M>{'\\mathrm{clip}'}</M> 在说：<strong>如果这次更新想把某个动作的概率
            改动超过 ε 倍，就把收益截断掉，不给你这个激励。</strong>
          </p>
          <p>
            为什么需要它？因为策略梯度是一个<em>局部</em>的近似 ——
            它只在当前策略附近才准。走太远，采样分布就变了，梯度也就不再是梯度。
            这个担忧从第 9 章「<M>{'d_\\pi'}</M> 依赖 θ」那里就埋下了。
          </p>
        </Beat>

        <Beat id="e4" keep>
          <Callout tone="insight" title="这本书真正教了什么">
            <p>
              如果只能带走一句话，我希望是这句：
              <strong>强化学习的每一个算法，都是在某个「算不出来」的地方，
              换了一种估计方式。</strong>
            </p>
            <ul>
              <li>期望算不出来 → 用采样（第 5 章）</li>
              <li>整条轨迹等不起 → 用自举（第 7 章）</li>
              <li>表格装不下 → 用函数近似（第 8 章）</li>
              <li>argmax 做不到 → 用参数化策略（第 9 章）</li>
              <li>回报方差太大 → 用评论家（第 10 章）</li>
            </ul>
            <p>
              每一次替换都在<strong>用偏差换方差、用严谨换可行</strong>。
              看懂了这条线索，以后遇到任何新算法，
              你都可以先问一句：<em>它在哪里做了替换，付出了什么代价？</em>
            </p>
            <p>这个问题几乎总能问到点子上。</p>
          </Callout>
        </Beat>
      </Act>
    </div>
  )
}
