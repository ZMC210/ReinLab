import { useMemo, useState } from 'react'
import { ACTIONS, buildGridMDP, classicGrid } from '../core/mdp'
import { argmaxActions, uniformPolicy } from '../core/policy'
import { valueIterationSolve } from '../core/solvers'
import { mdpToEnv } from '../core/env'
import { smooth } from '../core/learn'
import { policyGradient, type PGVariant } from '../core/pg'
import { Act, Beat } from '../narrative/Act'
import { ChapterGlance, ChapterHero, type Glance } from '../narrative/ChapterShell'
import { PredictChoice } from '../narrative/Predict'
import { LiveFormula } from '../formula/LiveFormula'
import { useFormulaCtx } from '../formula/core'
import { policyGradient as pgFormula } from '../formula/bellman'
import { GridWorld } from '../viz/GridWorld'
import { LineChart } from '../viz/LineChart'
import { Scrubber } from '../ui/Scrubber'
import { Callout, Code, Details, M, MB, Panel, Seg, Slider, Stat } from '../ui/prims'
import { fmt, useColors } from '../theme'

const F_PG = pgFormula()

const GLANCE: Glance = {
  formula: String.raw`\nabla_\theta J(\theta) = \mathbb{E}_{S\sim\eta,\,A\sim\pi}
  \Big[ \nabla_\theta \ln \pi(A\mid S,\theta)\; q_\pi(S,A) \Big]`,
  formulaNote:
    '策略梯度定理。它最关键的性质是右边写成了期望 —— 只要能采样，就能估计梯度，完全不需要模型，也不需要对策略求导以外的任何东西。',
  takeaways: [
    <>
      策略直接参数化为 <M>{'\\pi(a|s,\\theta)'}</M>，用梯度<strong>上升</strong>
      最大化目标 <M>{'J(\\theta)'}</M>。价值不再是目的，只是梯度里的一个系数。
    </>,
    <>
      <M>{'\\nabla\\ln\\pi'}</M> 是「把这个动作的概率往上推」的方向，
      <M>{'q'}</M> 是「该推多用力、往哪边推」。好动作被抬高，坏动作被压低。
    </>,
    <>
      softmax 参数化让所有动作概率恒为正 —— <strong>探索是内建的</strong>，
      不需要外挂 ε-贪心。
    </>,
    <>
      减去一个只依赖状态的基线 <M>{'b(s)'}</M> <strong>不改变梯度的期望</strong>，
      却能大幅降低方差。最优基线接近 <M>{'v_\\pi(s)'}</M>。
    </>,
  ],
  traps: [
    <>
      把 <M>{'\\nabla\\ln\\pi'}</M> 当成某种技巧。它就是 <M>{'\\nabla\\pi/\\pi'}</M>，
      那个 <M>{'1/\\pi'}</M> 正是为了抵消采样时按 <M>{'\\pi'}</M> 抽样带来的偏置。
    </>,
    <>
      以为策略梯度是「另一套体系」。它和价值方法共用同一个 <M>{'q'}</M> ——
      区别只在于用 q 来<strong>选</strong>动作，还是用 q 来<strong>加权</strong>梯度。
    </>,
    <>
      忽略 REINFORCE 的高方差。它是无偏的，但一条轨迹的回报噪声极大，
      不加基线时常常要几千个回合才动得起来。
    </>,
  ],
}

export function Chapter9() {
  const C = useColors()
  const [gamma] = useState(0.9)
  const [variant, setVariant] = useState<PGVariant>('reinforce')
  const [lr, setLr] = useState(0.4)
  const [episodes, setEpisodes] = useState(1200)
  const [k, setK] = useState(0)
  const [probeS, setProbeS] = useState(6)

  const mdp = useMemo(() => buildGridMDP(classicGrid()), [])
  const env = useMemo(() => mdpToEnv(mdp, { start: 0, horizon: 30 }), [mdp])
  const star = useMemo(() => valueIterationSolve(mdp, gamma), [mdp, gamma])
  const optActions = useMemo(() => argmaxActions(star.policy), [star])
  const pi0 = useMemo(() => uniformPolicy(mdp.nS), [mdp.nS])
  const zeros = useMemo(() => new Array<number>(mdp.nS).fill(0), [mdp.nS])
  const ctx = useFormulaCtx(mdp, pi0, gamma, zeros, probeS)

  const runs = useMemo(
    () =>
      (['reinforce', 'reinforce-baseline'] as PGVariant[]).map((v) => ({
        v,
        res: policyGradient(env, v, {
          gamma,
          alphaTheta: lr,
          alphaW: 0.2,
          episodes,
          seed: 21,
          probes: 40,
        }),
      })),
    [env, gamma, lr, episodes],
  )
  const cur = runs.find((r) => r.v === variant)!.res
  const kk = Math.min(k, cur.snaps.length - 1)
  const snap = cur.snaps[kk]
  const wrong = useMemo(
    () => argmaxActions(snap.policy).filter((a, i) => a !== optActions[i]).length,
    [snap, optActions],
  )

  const varOf = (xs: number[]) => {
    const t = xs.slice(-300)
    const m = t.reduce((a, b) => a + b, 0) / t.length
    return t.reduce((a, b) => a + (b - m) ** 2, 0) / t.length
  }

  const probs = snap.policy[probeS]

  return (
    <div>
      <ChapterHero
        n={9}
        hook="价值只是手段。能不能绕开它，直接优化策略本身？"
        lead={
          <>
            <p>
              前八章都在做同一件事：<strong>先把价值估准，再从价值里读出策略</strong>。
              这条路很成功，但它有两个躲不掉的死角 —— 连续动作空间里的{' '}
              <M>{'\\arg\\max'}</M> 算不出来，而本质随机的最优策略也表达不了。
            </p>
            <p>
              这一章换一条路：把策略本身写成 <M>{'\\pi(a|s,\\theta)'}</M>，
              定义一个标量目标 <M>{'J(\\theta)'}</M>，然后<em>直接对 θ 做梯度上升</em>。
              听上去理所当然，难点在于 —— <strong>目标函数里藏着一个依赖 θ 的状态分布</strong>，
              对它求导可不是链式法则一句话的事。
            </p>
          </>
        }
        gains={[
          '知道把策略参数化之后，「最优」该怎么定义（两个常用目标）',
          '理解策略梯度定理为什么能写成一个可采样的期望',
          '看懂 ∇lnπ 的几何含义：它在往哪个方向推概率',
          '亲手看到 REINFORCE 的方差有多大，以及基线能压下去多少',
          '明白 softmax 参数化为什么天然自带探索',
        ]}
      />

      <ChapterGlance g={GLANCE} />

      {/* ───────────────────── 第 1 幕 ───────────────────── */}
      <Act
        id="a1"
        no="第 1 幕"
        title="把策略变成一组参数"
        goal="softmax 参数化，以及「最优」在参数空间里的两种定义。"
        minutes={9}
        points={[
          <>
            <M>{'\\pi(a|s,\\theta) = \\dfrac{e^{h(s,a,\\theta)}}{\\sum_{a\'} e^{h(s,a\',\\theta)}}'}</M>{' '}
            —— 所有概率恒为正，探索内建。
          </>,
          <>
            两个常用目标：平均状态价值 <M>{'\\bar v_\\pi = \\sum_s d(s)v_\\pi(s)'}</M> 与
            平均单步奖励 <M>{'\\bar r_\\pi'}</M>。折扣情形下两者只差一个常数因子。
          </>,
          <>
            关键难点：<M>{'J(\\theta)'}</M> 里的状态分布 <M>{'d_\\pi'}</M> <strong>也依赖 θ</strong>。
            策略梯度定理的全部技术含量都在处理这一项。
          </>,
        ]}
        stage={() => (
          <div className="space-y-4">
            <Panel title="活的公式：策略梯度定理">
              <LiveFormula node={F_PG} ctx={ctx} />
              <p className="mt-3 border-t border-line pt-3 text-[11.5px] leading-relaxed text-faint">
                注意右边是一个<strong>期望</strong>。凡是期望，就可以用采样估计 ——
                这句话是全章的立足点。
              </p>
            </Panel>

            <Panel title={`状态 s${probeS + 1} 上的动作概率分布`}>
              <div className="space-y-1.5">
                {probs.map((p, a) => (
                  <div key={a} className="flex items-center gap-2.5">
                    <span className="w-8 font-mono text-[12px] text-dim">
                      {ACTIONS[a].glyph}
                    </span>
                    <div className="h-3 flex-1 overflow-hidden rounded bg-surface2">
                      <div
                        className="h-full rounded transition-all duration-300"
                        style={{
                          width: `${p * 100}%`,
                          background: a === optActions[probeS] ? C.value : C.policy,
                        }}
                      />
                    </div>
                    <span className="w-11 text-right font-mono text-[11.5px] text-faint">
                      {fmt(p * 100, 0)}%
                    </span>
                  </div>
                ))}
              </div>
              <p className="mt-3 text-[11.5px] leading-relaxed text-faint">
                绿色那条是真正的最优动作。拖下面的时间轴，看它怎么被一点点顶上去 ——
                <strong>而且其它动作的概率永远不会归零</strong>。
              </p>
              <div className="mt-3">
                <Scrubber k={kk} setK={setK} max={cur.snaps.length - 1} label="快照" fps={6} />
              </div>
            </Panel>

            <Panel title="当前策略">
              <GridWorld
                mdp={mdp}
                policy={snap.policy}
                showValues={false}
                showHeatmap={false}
                cell={50}
                onCellClick={setProbeS}
              />
              <p className="mt-2 text-center text-[11.5px] text-faint">
                点一个格子，换一个观察对象 · 与最优不同的格子：{wrong} / 25
              </p>
            </Panel>
          </div>
        )}
      >
        <Beat id="b1" keep>
          <p>
            第一步是把「策略」从一张表变成一个函数。最常用的写法是 softmax：
          </p>
          <MB>{"\\pi(a \\mid s, \\theta) = \\frac{\\exp\\!\\big(h(s,a,\\theta)\\big)}{\\sum_{a'}\\exp\\!\\big(h(s,a',\\theta)\\big)}"}</MB>
          <p>
            其中 <M>{'h'}</M> 是任意的打分函数（表格、线性、神经网络都行）。
            softmax 有两个我们非常需要的性质：概率自动归一化，
            而且<strong>每个动作的概率都严格大于 0</strong>。
          </p>
          <p>
            后一条意味着 —— <em>探索不再需要外挂</em>。
            前面章节里那个人为塞进去的 ε，在这里被参数化本身吸收掉了。
          </p>
        </Beat>

        <Beat id="b2" keep>
          <p>
            第二步是定义「好」。策略参数化之后，目标必须是一个<strong>标量</strong>，
            这样才能求梯度。两个标准选择：
          </p>
          <MB>
            {String.raw`\bar v_\pi = \sum_{s} d_\pi(s)\, v_\pi(s)
            \qquad\text{与}\qquad
            \bar r_\pi = \sum_{s} d_\pi(s) \sum_a \pi(a|s)\, r(s,a)`}
          </MB>
          <p>
            前者是「平均能拿多少总回报」，后者是「平均每步能拿多少」。
            在折扣情形下可以证明 <M>{'\\bar r_\\pi = (1-\\gamma)\\bar v_\\pi'}</M>，
            所以优化哪个都一样。
          </p>
        </Beat>

        <Beat id="b3" keep>
          <Callout tone="trap" title="真正的难点在哪">
            <p>
              注意 <M>{'d_\\pi(s)'}</M> 这个权重：它是<strong>策略下的状态分布</strong>。
              换了策略，去的地方就变了，权重也就变了。
            </p>
            <p>
              所以 <M>{'\\nabla_\\theta J'}</M> 有两部分：一部分来自
              「同一个状态上动作概率变了」，另一部分来自「去的状态本身变了」。
              后者看起来完全无从下手 —— 你没法对「环境的访问分布」求导。
            </p>
            <p>
              策略梯度定理的漂亮之处，就在于它证明了
              <strong>第二部分可以被完全吸收掉</strong>，
              最终的表达式里只剩下对 <M>{'\\pi'}</M> 的导数。
            </p>
          </Callout>
        </Beat>
      </Act>

      {/* ───────────────────── 第 2 幕 ───────────────────── */}
      <Act
        id="a2"
        no="第 2 幕"
        title="策略梯度定理"
        goal="梯度可以写成一个期望，于是采样就够了。"
        minutes={11}
        points={[
          <>
            <M>{'\\nabla J = \\mathbb{E}[\\nabla\\ln\\pi(A|S,\\theta)\\, q_\\pi(S,A)]'}</M> ——
            右边完全由「可采样的量」组成。
          </>,
          <>
            <M>{'\\nabla\\ln\\pi = \\nabla\\pi/\\pi'}</M>。那个 <M>{'1/\\pi'}</M>
            是用来抵消「按 π 采样」带来的偏置的，不是凭空冒出来的技巧。
          </>,
          <>
            softmax 的得分函数特别干净：
            <M>{'\\nabla_{\\theta_{s,a\'}}\\ln\\pi(a|s) = \\mathbb 1[a\'=a] - \\pi(a\'|s)'}</M>。
          </>,
          <>
            因为要除以 <M>{'\\pi'}</M>，所以策略<strong>必须处处为正</strong> ——
            这又一次说明为什么用 softmax 而不是确定性策略。
          </>,
        ]}
        stage={() => (
          <div className="space-y-4">
            <Panel title="一次更新，概率被推向哪">
              <PushDiagram probs={probs} best={optActions[probeS]} />
              <p className="mt-3 text-[11.5px] leading-relaxed text-faint">
                <M>{'\\nabla\\ln\\pi'}</M> 对被选中的动作是 <M>{'1-\\pi(a)'}</M>（正，往上推），
                对其它动作是 <M>{'-\\pi(a\')'}</M>（负，往下压）。
                乘上 <M>{'q'}</M> 之后：<M>{'q>0'}</M> 就按这个方向走，
                <M>{'q<0'}</M> 就反着走 —— 坏动作被压下去。
              </p>
            </Panel>
            <Panel title="softmax 的得分函数">
              <MB className="!my-1">
                {String.raw`\frac{\partial \ln \pi(a\mid s,\theta)}{\partial \theta_{s,a'}}
                = \mathbb{1}[a'=a] - \pi(a'\mid s,\theta)`}
              </MB>
              <p className="mt-2 text-[11.5px] leading-relaxed text-faint">
                这个式子干净得让人怀疑。它说：把被执行动作的分数加一点，
                把所有动作的分数按当前概率减一点 —— 加加减减刚好保持归一。
              </p>
            </Panel>
          </div>
        )}
      >
        <Beat id="c1" keep>
          <p>定理本身：</p>
          <MB>
            {String.raw`\nabla_\theta J(\theta) = \sum_{s} \eta(s) \sum_{a}
            \nabla_\theta \pi(a\mid s,\theta)\, q_\pi(s,a)`}
          </MB>
          <p>
            这里 <M>{'\\eta'}</M> 是某个状态分布（具体形式取决于选的目标）。
            这个式子已经很好了 —— 里面没有 <M>{'\\nabla d_\\pi'}</M>。
            但它还不能直接采样，因为对 <M>{'a'}</M> 的求和不是期望。
          </p>
        </Beat>

        <Beat id="c2" keep>
          <p>接下来是全章最关键的一步变形。乘一个除一个：</p>
          <MB>
            {String.raw`\sum_a \nabla\pi(a|s)\, q(s,a)
            = \sum_a \pi(a|s)\, \frac{\nabla\pi(a|s)}{\pi(a|s)}\, q(s,a)
            = \mathbb{E}_{A\sim\pi}\!\left[\nabla\ln\pi(A|s)\, q(s,A)\right]`}
          </MB>
          <p>
            于是求和变成了期望，而期望可以用采样估计。整个定理化为
          </p>
          <MB>{'\\nabla_\\theta J = \\mathbb{E}\\left[ \\nabla_\\theta\\ln\\pi(A|S,\\theta)\\; q_\\pi(S,A) \\right]'}</MB>
          <p>
            那个 <M>{'\\ln'}</M> 不是为了好看。<M>{'1/\\pi'}</M> 这个因子的作用是
            <strong>抵消采样偏置</strong>：概率大的动作被抽中得多，
            除以 <M>{'\\pi'}</M> 正好把这份「抽中得多」还回去。
          </p>
        </Beat>

        <Beat id="c3" keep>
          <PredictChoice
            id="ch9-sign"
            question={
              <>
                智能体执行了动作 <M>{'a'}</M>，算出来 <M>{'q(s,a) = -3'}</M>（很差）。
                这一步更新会把 <M>{'\\pi(a|s)'}</M> 怎么样？
              </>
            }
            options={[
              { id: 'a', label: '还是会提高 —— 毕竟它被执行了' },
              { id: 'b', label: '降低，同时把其它动作的概率抬高' },
              { id: 'c', label: '不变 —— 负的 q 会被 softmax 归一化抵消' },
            ]}
            answer="b"
            explain={
              <>
                <p>
                  更新量是 <M>{'\\alpha \\cdot q \\cdot \\nabla\\ln\\pi'}</M>。
                  <M>{'\\nabla\\ln\\pi'}</M> 指向「提高该动作概率」的方向，
                  而 <M>{'q=-3'}</M> 是负数 —— <strong>负号把方向整个翻转了</strong>。
                </p>
                <p>
                  所以坏动作被压下去，而由于 softmax 要归一，
                  被压下去的概率质量会自动分给其它动作。
                </p>
                <p>
                  这也解释了一个常见的工程坑：如果你把所有奖励都加上一个大正数
                  （第 3 章说过这不改变最优策略），那么<strong>所有</strong> q 都变正，
                  于是<em>每个被执行的动作都被鼓励</em>，学习会变得极慢。
                  —— 修正它的办法，就是下一幕的基线。
                </p>
              </>
            }
          />
        </Beat>

        <Beat id="c4">
          <Details summary="展开：为什么策略必须处处为正">
            <p>
              上面那步变形里出现了 <M>{'\\nabla\\pi/\\pi'}</M>。
              如果某个动作的概率恰好是 0，这一项就没有定义。
            </p>
            <p>
              所以策略梯度方法<strong>要求策略是随机的、且处处为正</strong>。
              这正是 softmax 参数化的用武之地，也解释了
              为什么这一族方法叫 stochastic policy gradient。
            </p>
            <p>
              确定性策略也有对应的定理（DPG / DDPG），但推导路径完全不同 ——
              它绕开了对动作的求和，改为对 <M>{'q'}</M> 关于动作求导。
            </p>
          </Details>
        </Beat>
      </Act>

      {/* ───────────────────── 第 3 幕 ───────────────────── */}
      <Act
        id="a3"
        no="第 3 幕"
        title="REINFORCE 与基线"
        goal="用 MC 回报当 q，就是 REINFORCE；减一个基线，方差立刻塌下去。"
        minutes={10}
        points={[
          <>
            REINFORCE：用整条轨迹的真实回报 <M>{'G_t'}</M> 当 <M>{'q_\\pi'}</M> 的估计。
            无偏，但方差惊人。
          </>,
          <>
            对任意只依赖状态的 <M>{'b(s)'}</M>，有{' '}
            <M>{'\\mathbb{E}[\\nabla\\ln\\pi \\cdot b(s)] = 0'}</M> ——
            <strong>减基线不改变期望</strong>。
          </>,
          <>
            但它改变方差。取 <M>{'b(s)=v_\\pi(s)'}</M> 时，
            系数变成 <M>{'G_t-v(s)'}</M>，即<strong>优势</strong>：
            这个动作比平均水平好多少。
          </>,
          <>
            右边两条学习曲线的抖动差距，就是基线值多少钱。
          </>,
        ]}
        stage={() => (
          <div className="space-y-4">
            <Panel title="学习曲线">
              <LineChart
                height={196}
                xLabel="回合"
                series={runs.map((r, i) => ({
                  name: r.v === 'reinforce' ? 'REINFORCE' : 'REINFORCE + 基线',
                  color: [C.qvalue, C.value][i],
                  data: smooth(r.res.episodeReturn, 40),
                  width: r.v === variant ? 2.6 : 1.5,
                }))}
              />
              <div className="mt-3 grid grid-cols-2 gap-2.5">
                {runs.map((r) => (
                  <Stat
                    key={r.v}
                    label={r.v === 'reinforce' ? '无基线 · 回报方差' : '有基线 · 回报方差'}
                    value={fmt(varOf(r.res.episodeReturn), 2)}
                    color={r.v === 'reinforce' ? C.qvalue : C.value}
                  />
                ))}
              </div>
            </Panel>

            <Panel title="更新量的模长（方差的直接证据）">
              <LineChart
                height={176}
                logY
                xLabel="回合"
                series={runs.map((r, i) => ({
                  name: r.v === 'reinforce' ? '无基线' : '有基线',
                  color: [C.qvalue, C.value][i],
                  data: smooth(r.res.gradNorm, 40),
                }))}
              />
            </Panel>

            <Panel title="参数">
              <div className="space-y-3">
                <Seg
                  value={variant}
                  onChange={setVariant}
                  size="sm"
                  options={[
                    { value: 'reinforce', label: 'REINFORCE' },
                    { value: 'reinforce-baseline', label: '+ 基线' },
                  ]}
                />
                <Slider
                  label={<>学习率 <M>{'\\alpha_\\theta'}</M></>}
                  value={lr}
                  min={0.05}
                  max={1.5}
                  step={0.05}
                  onChange={setLr}
                  accent={C.policy}
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
          <p>
            定理里的 <M>{'q_\\pi(s,a)'}</M> 是未知的。最朴素的补法：
            用第 5 章的蒙特卡洛 —— 跑完一整条轨迹，用真实回报 <M>{'G_t'}</M> 顶上去。
          </p>
          <MB>{'\\theta_{t+1} = \\theta_t + \\alpha\\, G_t\\, \\nabla_\\theta \\ln \\pi(a_t \\mid s_t, \\theta_t)'}</MB>
          <p>
            这就是 REINFORCE，1992 年的算法，也是所有策略梯度方法的祖先。
            它无偏，因为 <M>{'\\mathbb{E}[G_t] = q_\\pi(s_t,a_t)'}</M>。
          </p>
        </Beat>

        <Beat id="d2" keep>
          <p>
            无偏归无偏，方差大得离谱。而有一个几乎免费的改进：
            <strong>减去一个只依赖状态的量</strong>。
          </p>
          <MB>{'\\nabla J = \\mathbb{E}\\left[\\nabla\\ln\\pi(A|S)\\left(q_\\pi(S,A) - b(S)\\right)\\right]'}</MB>
          <p>为什么可以随便减？因为多出来的那一项期望恒为 0：</p>
          <MB>
            {String.raw`\mathbb{E}_{A\sim\pi}\!\left[\nabla\ln\pi(A|s)\,b(s)\right]
            = b(s)\sum_a \pi(a|s)\frac{\nabla\pi(a|s)}{\pi(a|s)}
            = b(s)\,\nabla\underbrace{\sum_a \pi(a|s)}_{=1} = 0`}
          </MB>
          <p>
            <strong>期望没变，方差却能显著下降。</strong>
            这是强化学习里少见的「白给」。
          </p>
        </Beat>

        <Beat id="d3" keep>
          <PredictChoice
            id="ch9-baseline"
            question={
              <>
                既然减任意 <M>{'b(s)'}</M> 都不改变期望，那减一个巨大的常数（比如 1000）
                会怎样？
              </>
            }
            options={[
              { id: 'a', label: '没影响，反正期望不变' },
              { id: 'b', label: '方差反而更大 —— 基线选得不好会帮倒忙' },
              { id: 'c', label: '算法会发散' },
            ]}
            answer="b"
            explain={
              <>
                <p>
                  「不改变期望」和「不改变方差」是两回事。
                  方差是 <M>{'\\mathbb{E}[\\|\\nabla\\ln\\pi\\|^2 (q-b)^2]'}</M> 减去期望的平方，
                  它是 <M>{'b'}</M> 的<strong>二次函数</strong>，有唯一的最小值点。
                </p>
                <p>
                  离那个最小值点越远，方差越大。减 1000 会让 <M>{'(q-b)^2'}</M> 巨大，
                  方差爆炸。
                </p>
                <p>
                  理论最优基线是 <M>{'b^*(s)=\\frac{\\mathbb{E}[\\|\\nabla\\ln\\pi\\|^2 q]}{\\mathbb{E}[\\|\\nabla\\ln\\pi\\|^2]}'}</M>，
                  实践中一般直接取 <M>{'b(s)=v_\\pi(s)'}</M> —— 简单、接近最优，
                  而且有清晰的含义：<strong>这个动作比该状态的平均水平好多少</strong>。
                </p>
                <p>
                  这个量 <M>{'q_\\pi(s,a)-v_\\pi(s)'}</M> 有个名字，叫<strong>优势函数</strong>。
                  记住它，下一章它会以另一副面孔出现。
                </p>
              </>
            }
          />
        </Beat>

        <Beat id="d4" keep>
          <Code
            code={`def reinforce(env, gamma=0.9, alpha=0.4, episodes=2000, baseline=True):
    theta = zeros(env.nS, env.nA)      # softmax 的打分
    v     = zeros(env.nS)              # 基线（顺便用 MC 学）

    for _ in range(episodes):
        traj, s = [], env.reset()      # ① 用当前策略采一整条轨迹
        for _ in range(env.horizon):
            p = softmax(theta[s])
            a = sample(p)
            sp, r, done = env.step(s, a)
            traj.append((s, a, r))
            s = sp
            if done: break

        G = 0.0                        # ② 从后往前算回报
        for t, (s, a, r) in enumerate(reversed(traj)):
            G = gamma * G + r
            coef = G - v[s] if baseline else G      # ③ 优势 or 裸回报
            if baseline:
                v[s] += 0.2 * (G - v[s])

            p = softmax(theta[s])                   # ④ ∇lnπ = onehot(a) − π
            for k in range(env.nA):
                theta[s][k] += alpha * coef * ((k == a) - p[k])
    return theta`}
          />
        </Beat>

        <Beat id="d5" keep>
          <Callout tone="trap" title="这一章留下的问题">
            <p>
              REINFORCE 能跑，但它继承了蒙特卡洛的全部毛病：
              <strong>必须等回合结束</strong>，而且<strong>方差大</strong>。
              这两条我们在第 5 章末尾就抱怨过一次了。
            </p>
            <p>
              当时的解药是 TD —— 用估计代替真回报，走一步就更新。
              现在同样的解药能不能再吃一次？
            </p>
            <p>
              也就是说：<em>梯度公式里那个 <M>{'q_\\pi(s,a)'}</M>，
              与其用整条轨迹的回报去估，不如专门养一个函数来估它？</em>
            </p>
            <p>
              那个「专门估 q 的函数」，就是<strong>评论家</strong>。
              而策略本身，就是<strong>演员</strong>。
            </p>
          </Callout>
        </Beat>
      </Act>
    </div>
  )
}

/* ────────────────────────── ∇lnπ 的方向示意 ────────────────────────── */

function PushDiagram({ probs, best }: { probs: number[]; best: number }) {
  const C = useColors()
  const W = 400
  const H = 150
  const bw = W / probs.length
  const grads = probs.map((p, a) => (a === best ? 1 - p : -p))

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
      <line x1={0} x2={W} y1={H / 2} y2={H / 2} stroke="var(--line-strong)" />
      {grads.map((g, a) => {
        const h = Math.abs(g) * (H / 2 - 18)
        const x = a * bw + bw / 2
        const up = g > 0
        return (
          <g key={a}>
            <rect
              x={x - 13}
              y={up ? H / 2 - h : H / 2}
              width={26}
              height={h}
              rx={4}
              fill={up ? C.value : C.danger}
              opacity={0.75}
            />
            <text
              x={x}
              y={up ? H / 2 + 15 : H / 2 - 6}
              textAnchor="middle"
              fontSize={13}
              fill="var(--ink)"
            >
              {ACTIONS[a].glyph}
            </text>
            <text
              x={x}
              y={up ? H / 2 - h - 5 : H / 2 + h + 13}
              textAnchor="middle"
              fontSize={10}
              fill="var(--ink-faint)"
              fontFamily="ui-monospace, monospace"
            >
              {fmt(g, 2)}
            </text>
          </g>
        )
      })}
      <text x={6} y={14} fontSize={10.5} fill={C.value}>
        ↑ 概率被推高
      </text>
      <text x={6} y={H - 6} fontSize={10.5} fill={C.danger}>
        ↓ 概率被压低
      </text>
    </svg>
  )
}
