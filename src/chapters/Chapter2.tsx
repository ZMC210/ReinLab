import { useMemo, useState } from 'react'
import { buildGridMDP, classicGrid } from '../core/mdp'
import { deterministicPolicy, qFromV, uniformPolicy, type Policy } from '../core/policy'
import {
  PPi,
  contractionWitness,
  normInf,
  policyEvaluationDirect,
  policyEvaluationTrace,
  valueIterationSolve,
} from '../core/solvers'
import { runningMean, sampleReturns } from '../core/sample'
import { useBus } from '../highlight/bus'
import { Act, Beat } from '../narrative/Act'
import { ChapterGlance, ChapterHero, type Glance } from '../narrative/ChapterShell'
import { PredictChoice, PredictNumber } from '../narrative/Predict'
import { LiveFormula } from '../formula/LiveFormula'
import { useFormulaCtx } from '../formula/core'
import {
  bellmanElementwise,
  bellmanMatrix,
  qDefinition,
  vFromQ,
  valueAsExpectation,
} from '../formula/bellman'
import { GridWorld } from '../viz/GridWorld'
import { LineChart } from '../viz/LineChart'
import { MatrixHeatmap } from '../viz/MatrixHeatmap'
import { Scrubber } from '../ui/Scrubber'
import { Callout, Code, Details, M, MB, Panel, Seg, Slider, Toggle } from '../ui/prims'
import { fmt, useColors } from '../theme'

const GLANCE: Glance = {
  formula: String.raw`v_\pi(s) = \sum_{a}\pi(a\mid s)\left[
  \underbrace{\sum_{r} p(r\mid s,a)\,r}_{\text{今天的奖励}}
  \; + \;\gamma\underbrace{\sum_{s'} p(s'\mid s,a)\,v_\pi(s')}_{\text{折扣后的明天}}\right]`,
  formulaNote:
    '贝尔曼公式。它只说了一句话：今天的价值 = 今天的奖励 + 折扣后的明天的价值。25 个状态的方程联立起来，整个价值场可以一次解出来。',
  takeaways: [
    <>
      状态价值 <M>{'v_\\pi(s)=\\mathbb{E}[G_t\\mid S_t=s]'}</M>：从 <M>{'s'}</M> 出发按{' '}
      <M>{'\\pi'}</M> 走，<strong>期望</strong>能拿到的总回报。
    </>,
    <>
      整个推导只用到回报的递归性 <M>{'G_t = R_{t+1}+\\gamma G_{t+1}'}</M>，
      两边取条件期望即可。
    </>,
    <>
      矩阵形式 <M>{'v_\\pi = r_\\pi + \\gamma P_\\pi v_\\pi'}</M> 有唯一解
      <M>{'(I-\\gamma P_\\pi)^{-1}r_\\pi'}</M> —— 因为 <M>{'\\gamma<1'}</M>{' '}
      保证了 <M>{'I-\\gamma P_\\pi'}</M> 可逆。
    </>,
    <>
      迭代解法的误差按 <M>{'\\gamma^k'}</M> 指数塌缩。这就是压缩映射定理，
      也是后面几乎所有算法收敛性的源头。
    </>,
  ],
  traps: [
    <>
      把它当成「一个能直接算出 v 的公式」。<strong>两边都有 v</strong> ——
      它是方程，不是赋值语句。
    </>,
    <>
      忘了它是<strong>对某个给定的 π</strong> 成立的。换策略，整组方程全换。
    </>,
    <>
      混淆 <M>{'v_\\pi'}</M> 与 <M>{'q_\\pi'}</M>：前者是「站在这里有多好」，
      后者是「站在这里<em>做这件事</em>有多好」。
    </>,
  ],
}

const F_BELLMAN_FULL = bellmanElementwise('full')
const F_BELLMAN = bellmanElementwise('compact')
const F_MATRIX = bellmanMatrix()
const F_Q = qDefinition()
const F_VQ = vFromQ()
const F_VEXP = valueAsExpectation()

type PolicyKind = 'uniform' | 'naive' | 'optimal'

/** 一个「看起来挺合理其实很糟」的确定性策略：一律往右，撞墙就往下 */
function naiveActions(nS: number, cols: number): number[] {
  return Array.from({ length: nS }, (_, s) => ((s % cols) === cols - 1 ? 2 : 1))
}

export function Chapter2() {
  const C = useColors()
  const mdp = useMemo(() => buildGridMDP(classicGrid()), [])
  const [gamma, setGamma] = useState(0.9)
  const [kind, setKind] = useState<PolicyKind>('uniform')
  const [mode, setMode] = useState<'symbolic' | 'numeric'>('symbolic')
  const [variant, setVariant] = useState<'compact' | 'full'>('compact')
  const [k, setK] = useState(0)
  const [showQ, setShowQ] = useState(false)
  const [logAxis, setLogAxis] = useState(true)

  const focus = useBus((s) => s.focus)
  const setFocus = useBus((s) => s.setFocus)
  // 默认钉在 s9 = 第 2 行第 4 列：它是普通格子，但左边紧邻一个禁区，展开式好看
  const s = focus ?? 8

  const optimal = useMemo(() => valueIterationSolve(mdp, gamma), [mdp, gamma])

  const pi: Policy = useMemo(() => {
    if (kind === 'uniform') return uniformPolicy(mdp.nS)
    if (kind === 'naive') return deterministicPolicy(naiveActions(mdp.nS, mdp.grid.cols))
    return optimal.policy
  }, [kind, mdp, optimal])

  const vStar = useMemo(() => policyEvaluationDirect(mdp, pi, gamma), [mdp, pi, gamma])
  const trace = useMemo(() => policyEvaluationTrace(mdp, pi, gamma, 80), [mdp, pi, gamma])
  const kk = Math.min(k, trace.length - 1)
  const vk = trace[kk]

  const errors = useMemo(() => trace.map((v) => normInf(v, vStar)), [trace, vStar])
  const bound = useMemo(
    () => errors.map((_, i) => (errors[0] || 1) * Math.pow(gamma, i)),
    [errors, gamma],
  )

  const witness = useMemo(() => {
    const a = new Array(mdp.nS).fill(0)
    const b = mdp.grid.cells.map((_, i) => (i % 2 ? 8 : -8))
    return contractionWitness(mdp, pi, gamma, a, b, 60)
  }, [mdp, pi, gamma])

  const P = useMemo(() => PPi(mdp, pi), [mdp, pi])
  const q = useMemo(() => qFromV(mdp, gamma, vStar), [mdp, gamma, vStar])

  const samples = useMemo(
    () => runningMean(sampleReturns(mdp, pi, gamma, s, 300, 260, 7)),
    [mdp, pi, gamma, s],
  )

  const ctx = useFormulaCtx(mdp, pi, gamma, vStar, s)
  const ctxK = useFormulaCtx(mdp, pi, gamma, vk, s)

  const policySeg = (
    <Seg
      size="sm"
      value={kind}
      onChange={setKind}
      options={[
        { value: 'uniform', label: '均匀随机', hint: '每个动作都 1/5' },
        { value: 'naive', label: '一路向右', hint: '看起来有道理的笨策略' },
        { value: 'optimal', label: '最优', hint: '第 3 章才会讲怎么找到它' },
      ]}
    />
  )

  const worldPanel = (v: number[], extra?: React.ReactNode) => (
    <Panel title="世界" right={policySeg}>
      <GridWorld
        mdp={mdp}
        v={v}
        q={showQ ? q : undefined}
        policy={pi}
        showQ={showQ}
        cell={54}
        onCellClick={(x) => setFocus(x === focus ? null : x)}
      />
      <div className="mt-3 flex items-center justify-between gap-3">
        <span className="text-[11.5px] text-faint">
          {focus === null ? '点一个格子把它钉住' : `已钉住 s${focus + 1}，再点一次取消`}
        </span>
        <Toggle label="显示 q(s,a)" checked={showQ} onChange={setShowQ} />
      </div>
      <div className="mt-3 border-t border-line pt-3">
        <Slider
          label={
            <>
              折扣因子 <M>{'\\gamma'}</M>
            </>
          }
          value={gamma}
          min={0.1}
          max={0.99}
          step={0.01}
          onChange={setGamma}
          accent={C.gamma}
        />
      </div>
      {extra}
    </Panel>
  )

  return (
    <div>
      <ChapterHero
        n={2}
        hook="能给策略打分了吗？给我一个 π，它到底值多少分？"
        lead={
          <>
            <p>
              这一章只做一件事：给策略打分。但这一件事，是整个强化学习的支点 ——
              后面八章全部的算法，说到底都是在用不同的方式逼近这一章的那个数。
            </p>
            <p>
              打分的思路会经历一次很漂亮的转折：先用最笨的办法（走一万遍取平均），
              发现它慢得离谱；然后发现<em>一个状态的价值可以用它邻居的价值表示出来</em>，
              于是一个方程组代替了一万次试验。这个转折就叫贝尔曼公式。
            </p>
          </>
        }
        gains={[
          '知道状态价值 v_π(s) 到底是什么的期望',
          '能自己从 G_t = R_{t+1} + γG_{t+1} 推出贝尔曼公式',
          '看懂矩阵形式 v_π = r_π + γP_π v_π，并知道它为什么一定有唯一解',
          '亲眼见到压缩映射：误差以 γᵏ 的速度指数衰减',
          '分清 v_π 与 q_π，知道 q 多问了哪一句',
        ]}
      />

      <ChapterGlance g={GLANCE} />

      {/* ───────────────────────── 第 1 幕 ───────────────────────── */}
      <Act
        id="a1"
        no="第 1 幕"
        title="最笨的打分办法"
        goal="状态价值的定义，以及为什么「多试几次取平均」这条路走不通。"
        minutes={10}
        points={[
          <>
            <M>{'v_\\pi(s) \\doteq \\mathbb{E}[G_t \\mid S_t = s]'}</M> —— 一个数，
            概括「从这里开始，按 π 走下去，能拿多少」。
          </>,
          <>
            最朴素的算法是采样求平均。它<strong>能用但太贵</strong>：
            估准一个状态就要几百条轨迹，而这里有 25 个状态。
          </>,
          <>
            采样法还有个更根本的浪费：<strong>它没利用状态之间的关系</strong>。
            相邻格子的价值明明是相互约束的。
          </>,
        ]}
        stage={() => (
          <div className="space-y-5">
            {worldPanel(vStar)}
            <Panel title={`从 s${s + 1} 出发，采样 300 条轨迹`}>
              <LineChart
                series={[
                  { name: '前 n 条轨迹的平均回报', color: C.reward, data: samples },
                  {
                    name: '真值 v_π',
                    color: C.value,
                    data: samples.map(() => vStar[s]),
                    dashed: true,
                  },
                ]}
                xLabel="已采样的轨迹条数 n"
                height={200}
              />
              <p className="mt-2 text-[11.5px] leading-relaxed text-faint">
                抖了几十上百条才勉强贴住真值 —— 而这只是<strong>一个</strong>状态。
                25 个状态就要重复 25 次。
              </p>
            </Panel>
          </div>
        )}
      >
        <Beat id="b1">
          <p>
            要比较两个策略的好坏，得先有一把尺子。最自然的尺子是：
            <strong>从某个状态出发，按这个策略一直走下去，平均能拿到多少折扣回报。</strong>
            这就是<strong>状态价值</strong>：
          </p>
          <div className="rounded-2xl border border-line bg-surface px-4 py-1">
            <LiveFormula node={F_VEXP} ctx={ctx} />
          </div>
          <p>
            注意这里有两层随机性都被期望吃掉了：策略选动作是随机的，环境的转移和奖励也可能是随机的。
            <M>{'v_\\pi(s)'}</M> 是对<em>所有可能的未来</em>取平均。
          </p>
        </Beat>

        <Beat id="b2">
          <h3>那就真的去平均一下</h3>
          <p>
            定义里写着「期望」，最直白的做法就是去采样：从 <M>{'s'}</M> 出发跑一条轨迹，
            算出它的折扣回报；再跑一条，再算一次；跑够多条，取平均。
          </p>
          <p>
            右边就是这么干的结果。<strong>你会看到它确实在收敛，但收敛得很难看。</strong>
            前几十条轨迹的平均值上蹿下跳，因为单条轨迹的回报方差很大 ——
            运气好一头扎进目标，运气差在禁区里打转。
          </p>
          <Callout tone="trap" title="这条路的三个致命伤">
            <ul>
              <li>
                <strong>慢</strong>：一个状态要几百条轨迹，25 个状态就是上万步模拟。
              </li>
              <li>
                <strong>抖</strong>：方差随任务长度增长，长任务几乎没法用。
              </li>
              <li>
                <strong>浪费</strong>：每条轨迹都途经许多其它状态，
                这些信息全被扔掉了 —— 明明可以顺手更新它们。
              </li>
            </ul>
          </Callout>
        </Beat>

        <Beat id="b3">
          <p>
            第三点尤其刺眼。从 <M>{'s_7'}</M> 出发的轨迹，第二步就到了 <M>{'s_{12}'}</M>；
            那么这条轨迹的后半段，其实正是一条「从 <M>{'s_{12}'}</M> 出发」的轨迹。
          </p>
          <p>
            换句话说：<strong>各个状态的价值之间，一定存在某种关系。</strong>
            如果能把这个关系写下来，我们也许就不需要一遍遍地采样了。
          </p>

          <PredictChoice
            id="ch2-relation"
            question={
              <>
                在下笔之前先猜：<M>{'v_\\pi(s)'}</M> 和它的邻居们的价值{' '}
                <M>{'v_\\pi(s\')'}</M> 之间，最可能是什么关系？
              </>
            }
            options={[
              { id: 'a', label: 'v(s) 等于所有邻居价值的平均。' },
              { id: 'b', label: 'v(s) = 这一步的期望奖励 + γ × 邻居价值的加权平均。' },
              { id: 'c', label: '没有普遍关系，得看具体的网格长什么样。' },
            ]}
            answer="b"
            explain={
              <>
                <p>
                  就是 B，而且它有一个名字：<strong>贝尔曼公式</strong>。
                </p>
                <p>
                  两个要点缺一不可 —— 权重来自「策略选动作的概率 × 环境转移的概率」，
                  而未来那一项必须乘上 <M>{'\\gamma'}</M>，因为它晚了一步。
                  下一幕我们把它一行行推出来。
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
        title="今天 = 今天 + 折扣后的明天"
        goal="从回报的递归性，一步步推出贝尔曼公式；并看清公式里每个符号在世界里是谁。"
        minutes={13}
        points={[
          <>
            起点只有一句：<M>{'G_t = R_{t+1} + \\gamma G_{t+1}'}</M>。
            两边取条件期望，贝尔曼公式就出来了。
          </>,
          <>
            公式里有三层求和：对<strong>动作</strong>（策略的随机性）、
            对<strong>奖励</strong>、对<strong>后继状态</strong>（环境的随机性）。
          </>,
          <>
            把鼠标放在任意一段符号上，世界里对应的格子、箭头、后继会同时亮起来 ——
            <strong>符号与画面的对应关系不需要你去背</strong>。
          </>,
        ]}
        stage={() => (
          <div className="space-y-5">
            <Panel
              title="活的公式"
              right={
                <div className="flex gap-2">
                  <Seg
                    size="sm"
                    value={variant}
                    onChange={setVariant}
                    options={[
                      { value: 'compact', label: '紧凑' },
                      { value: 'full', label: '展开奖励' },
                    ]}
                  />
                </div>
              }
            >
              <LiveFormula node={variant === 'full' ? F_BELLMAN_FULL : F_BELLMAN} ctx={ctx} />
              <p className="mt-3 border-t border-line pt-3 text-[11.5px] leading-relaxed text-faint">
                把鼠标停在公式的任意一段上 —— 它指向的格子、箭头、后继状态会同时亮起来。
                反过来，在网格上划过某个箭头，公式里对应的那一项也会亮。
              </p>
            </Panel>
            {worldPanel(vStar)}
          </div>
        )}
      >
        <Beat id="c1">
          <h3>第一步：把回报拆成两段</h3>
          <p>回报的定义本身就带着递归的影子。把它写开：</p>
          <MB>{'G_t = R_{t+1} + \\gamma R_{t+2} + \\gamma^2 R_{t+3} + \\cdots'}</MB>
          <p>
            从第二项开始，每一项都含有因子 <M>{'\\gamma'}</M>。把它提出来：
          </p>
          <MB>{'G_t = R_{t+1} + \\gamma \\underbrace{\\left( R_{t+2} + \\gamma R_{t+3} + \\cdots \\right)}_{= \\; G_{t+1}} = R_{t+1} + \\gamma G_{t+1}'}</MB>
          <Callout tone="insight">
            <strong>
              <M>{'G_t = R_{t+1} + \\gamma G_{t+1}'}</M>
            </strong>
            {' '}—— 整章的全部内容都从这一行长出来。它说的是：
            这一趟旅程的总收获，等于第一步的收获，加上「从第二步开始的那一趟旅程」的收获打个折。
          </Callout>
        </Beat>

        <Beat id="c2">
          <h3>第二步：两边取期望</h3>
          <p>
            对 <M>{'S_t = s'}</M> 取条件期望，利用期望的线性性：
          </p>
          <MB>{'v_\\pi(s) = \\mathbb{E}[R_{t+1} \\mid S_t = s] + \\gamma\\, \\mathbb{E}[G_{t+1} \\mid S_t = s]'}</MB>
          <p>
            剩下的工作就是把这两项分别算出来。第一项是「当下」，第二项是「未来」。
          </p>

          <Details summary="逐行展开这两项（含关键的一步：马尔可夫性用在哪里）" defaultOpen>
            <p>
              <strong>当下那一项。</strong>先对动作求期望，再对奖励求期望：
            </p>
            <MB>{'\\mathbb{E}[R_{t+1} \\mid S_t = s] = \\sum_{a} \\pi(a\\mid s) \\, \\mathbb{E}[R_{t+1} \\mid S_t=s, A_t=a] = \\sum_{a} \\pi(a\\mid s) \\sum_{r} p(r \\mid s,a)\\, r'}</MB>
            <p>
              <strong>未来那一项。</strong>先按下一个状态分类：
            </p>
            <MB>{'\\mathbb{E}[G_{t+1} \\mid S_t = s] = \\sum_{s\'} \\mathbb{E}[G_{t+1} \\mid S_t=s, S_{t+1}=s\']\\, p(s\' \\mid s)'}</MB>
            <p>
              现在是全场最关键的一步。<M>{'\\mathbb{E}[G_{t+1} \\mid S_t=s, S_{t+1}=s\']'}</M>{' '}
              里带着「我是从 <M>{'s'}</M> 来的」这段历史。
              <strong>马尔可夫性告诉我们，这段历史对未来毫无影响</strong>，于是可以直接扔掉：
            </p>
            <MB>{'\\mathbb{E}[G_{t+1} \\mid S_t=s, S_{t+1}=s\'] = \\mathbb{E}[G_{t+1} \\mid S_{t+1}=s\'] = v_\\pi(s\')'}</MB>
            <p>
              这一扔，右边就出现了 <M>{'v_\\pi'}</M> 自己 —— 递归就是这样诞生的。
              再把 <M>{'p(s\'\\mid s) = \\sum_a \\pi(a\\mid s) p(s\'\\mid s,a)'}</M> 代进去，两项合并即得贝尔曼公式。
            </p>
            <Callout tone="rigor" title="记住这个位置">
              如果哪天你的问题不满足马尔可夫性，出问题的就是这一行 —— 而不是别的地方。
              第 1 章说「马尔可夫性是状态定义的性质」，代价就体现在这里。
            </Callout>
          </Details>
        </Beat>

        <Beat id="c3">
          <h3>第三步：读懂它</h3>
          <p>
            右边那个公式，现在请你<strong>一段一段地把鼠标放上去</strong>。
            这不是装饰性的交互 —— 认知负荷理论里有个「注意力分散效应」：
            当公式、图、文字彼此分离时，工作记忆会被消耗在建立对应关系上。
            现在对应关系由系统替你维持，你的脑力可以全部花在理解结构上。
          </p>
          <ul>
            <li>
              <M>{'\\pi(a\\mid s)'}</M> —— 你自己的选择，是唯一可以改的东西。
            </li>
            <li>
              <M>{'p(s\'\\mid s,a)'}</M>、<M>{'r(s,a)'}</M> —— 环境的脾气，改不了。
            </li>
            <li>
              <M>{'\\gamma'}</M> —— 你有多在乎未来。
            </li>
            <li>
              方括号里的整体 —— 它有个名字叫 <M>{'q_\\pi(s,a)'}</M>，第 6 幕见。
            </li>
          </ul>
          <Callout tone="trap" title="贝尔曼公式不是「算法」">
            它没有告诉你怎么算出 <M>{'v_\\pi'}</M>，它只是<strong>陈述了一组必须同时成立的等式</strong>。
            25 个状态就是 25 个方程，25 个未知数。
            「怎么解」是第 4、5 幕的事，别把「关系」和「求解」混为一谈 ——
            这是初学时最常见的混淆。
          </Callout>
        </Beat>
      </Act>

      {/* ───────────────────────── 第 3 幕 ───────────────────────── */}
      <Act
        id="a3"
        no="第 3 幕"
        title="亲手算一格"
        goal="把符号换成真实数字，确认自己是真的读懂了，而不是看懂了。"
        minutes={8}
        points={[
          <>
            确定性策略下，三层求和塌成一项：
            <M>{"v_\\pi(s) = r(s,\\pi(s)) + \\gamma v_\\pi(s')"}</M>。
          </>,
          <>
            「看懂公式」和「能算出数」之间有一道坎。这一幕就是用来跨的。
          </>,
        ]}
        stage={() => (
          <div className="space-y-5">
            <Panel
              title={`代入数值：s${s + 1}`}
              right={
                <Seg
                  size="sm"
                  value={mode}
                  onChange={setMode}
                  options={[
                    { value: 'symbolic', label: '符号' },
                    { value: 'numeric', label: '代入数值' },
                  ]}
                />
              }
            >
              <LiveFormula node={F_BELLMAN} ctx={ctx} mode={mode} />
              <p className="mt-3 border-t border-line pt-3 text-[11.5px] leading-relaxed text-faint">
                展开式里的每一个数字仍然是活的。把鼠标放在某个绿色的数上，
                看看它是网格里的哪一格。
              </p>
            </Panel>
            {worldPanel(vStar)}
          </div>
        )}
      >
        <Beat id="d1">
          <p>
            把右上角切到<strong>「代入数值」</strong>，公式会围绕当前钉住的状态逐项展开。
            每一行对应一个动作，行内是 <M>{'\\pi(a|s)'}</M>、<M>{'r(s,a)'}</M>、
            <M>{'\\gamma'}</M> 和后继状态的价值。
          </p>
          <p>
            在网格上换一个格子点一下，展开式会立刻跟着换。
            <em>建议你在均匀随机策略下点几个不同的格子，感受一下「五个动作各占 0.2」是怎么摊出来的。</em>
          </p>
        </Beat>

        <Beat id="d2">
          <PredictNumber
            id="ch2-hand-calc"
            question={
              <>
                切换到<strong>「一路向右」</strong>策略、<M>{'\\gamma = 0.9'}</M>。
                此时目标格 <M>{'s_{18}'}</M>（第 4 行第 3 列）本身的价值 <M>{'v_\\pi(s_{18})'}</M> 是多少？
                提示：这个策略在那里会一直向右走出去，不会停留。
              </>
            }
            min={-10}
            max={10}
            step={0.1}
            truth={policyEvaluationDirect(mdp, deterministicPolicy(naiveActions(mdp.nS, mdp.grid.cols)), 0.9)[17]}
            tolerance={0.6}
            explain={
              <>
                <p>
                  很多人会脱口而出「10」，因为第 1 章算过{' '}
                  <M>{'1 + 0.9 + 0.9^2 + \\cdots = 10'}</M>。但那个 10 的前提是
                  <strong>一直停在目标格上</strong>。
                </p>
                <p>
                  「一路向右」的策略根本不会停 —— 它进了目标拿到 <M>{'+1'}</M> 之后就继续往右，
                  然后撞墙、往下、再往右……价值当然完全不同。
                </p>
                <p>
                  <strong>价值是策略的函数，不是格子的属性。</strong>
                  同一个格子，换个策略就换个价值。这句话请记牢，
                  它是第 3 章「最优」这个概念得以成立的前提。
                </p>
              </>
            }
          />
        </Beat>

        <Beat id="d3">
          <h3>写成代码就是三行</h3>
          <p>
            贝尔曼公式落到代码上朴素得让人意外。给定 <code>pi</code>、<code>P</code>、
            <code>R</code> 和当前的估计 <code>v</code>，做一次「贝尔曼更新」：
          </p>
          <Code
            lang="python"
            code={`def bellman_backup(pi, P, R, gamma, v):
    """v_new[s] = Σ_a π(a|s) [ r(s,a) + γ Σ_s' p(s'|s,a) v(s') ]"""
    q = R + gamma * P @ v          # q[s,a]，P 的形状是 (nS, nA, nS)
    return (pi * q).sum(axis=1)    # 按 π 加权求和，得到 v_new[s]`}
          />
          <p>
            两行。<strong>第一行是方括号里的内容，第二行是外面那个求和。</strong>
            对照着公式再看一遍这两行 —— 如果能一一对上，这一幕就通过了。
          </p>
        </Beat>
      </Act>

      {/* ───────────────────────── 第 4 幕 ───────────────────────── */}
      <Act
        id="a4"
        no="第 4 幕"
        title="25 个方程一起看：矩阵形式"
        goal="把逐状态的公式打包成向量方程，并搞清楚它为什么一定有唯一解。"
        minutes={10}
        points={[
          <>
            <M>{'v_\\pi = r_\\pi + \\gamma P_\\pi v_\\pi'}</M>：把 25 个标量方程
            叠成一个向量方程，形状立刻清楚了。
          </>,
          <>
            <M>{'P_\\pi'}</M> 每行和为 1，所以它的谱半径是 1；
            <M>{'\\gamma P_\\pi'}</M> 的谱半径是 <M>{'\\gamma<1'}</M>，
            于是 <M>{'I-\\gamma P_\\pi'}</M> <strong>一定可逆</strong>。
          </>,
          <>
            解析解 <M>{'(I-\\gamma P_\\pi)^{-1}r_\\pi'}</M> 优雅但不实用：
            求逆是 <M>{'O(n^3)'}</M>，状态一多就崩。
          </>,
        ]}
        stage={() => (
          <div className="space-y-5">
            <Panel title="矩阵形式">
              <LiveFormula node={F_MATRIX} ctx={ctx} mode={mode} />
              <div className="mt-3 flex justify-end">
                <Seg
                  size="sm"
                  value={mode}
                  onChange={setMode}
                  options={[
                    { value: 'symbolic', label: '递归形式' },
                    { value: 'numeric', label: '解析解' },
                  ]}
                />
              </div>
            </Panel>
            <Panel title="P_π —— 把鼠标放在某一行上">
              <div className="flex flex-wrap items-start gap-5">
                <MatrixHeatmap m={P} size={13} label={`${mdp.nS} × ${mdp.nS}`} />
                <p className="max-w-[190px] text-[11.5px] leading-relaxed text-faint">
                  矩阵的第 <span className="text-brand">s</span> 行，
                  就是网格上「从格子 s 出发走一步」会落到哪里的概率分布。
                  <strong>一行 = 一次出发。</strong>
                  悬停时网格上的起点和落点会一起亮。
                </p>
              </div>
            </Panel>
            {worldPanel(vStar)}
          </div>
        )}
      >
        <Beat id="e1">
          <p>
            25 个状态就有 25 条贝尔曼公式。把它们摞起来写成向量，形式一下子清爽了：
          </p>
          <MB>{'v_\\pi = r_\\pi + \\gamma P_\\pi v_\\pi'}</MB>
          <p>其中</p>
          <ul>
            <li>
              <M>{'v_\\pi \\in \\mathbb{R}^{n}'}</M>：所有状态的价值摞成的列向量；
            </li>
            <li>
              <M>{'[r_\\pi]_s = \\sum_a \\pi(a\\mid s) r(s,a)'}</M>：策略 <M>{'\\pi'}</M> 下每个状态的平均即时奖励；
            </li>
            <li>
              <M>{'[P_\\pi]_{s,s\'} = \\sum_a \\pi(a\\mid s) p(s\'\\mid s,a)'}</M>：把动作「积掉」之后的状态转移矩阵。
            </li>
          </ul>
          <p>
            注意 <M>{'P_\\pi'}</M> 是把策略和环境<em>揉在一起</em>之后的产物 ——
            环境固定时，换策略就是换 <M>{'P_\\pi'}</M> 和 <M>{'r_\\pi'}</M>。
          </p>
        </Beat>

        <Beat id="e2">
          <h3>直接解出来</h3>
          <p>移项，把含 <M>{'v_\\pi'}</M> 的项归到一边：</p>
          <MB>{'(I - \\gamma P_\\pi)\\, v_\\pi = r_\\pi \\quad\\Longrightarrow\\quad v_\\pi = (I - \\gamma P_\\pi)^{-1} r_\\pi'}</MB>
          <p>
            一步到位。<strong>只要那个逆存在。</strong>而它确实存在，理由干净利落。
          </p>

          <Details summary="为什么 I − γP_π 一定可逆" defaultOpen>
            <p>
              <M>{'P_\\pi'}</M> 是一个随机矩阵：每一行非负且和为 1。这意味着它的
              无穷范数 <M>{'\\|P_\\pi\\|_\\infty = 1'}</M>，从而所有特征值的模都不超过 1。
            </p>
            <p>
              于是 <M>{'\\gamma P_\\pi'}</M> 的特征值模都不超过 <M>{'\\gamma < 1'}</M>，
              所以 <M>{'1'}</M> 不可能是 <M>{'\\gamma P_\\pi'}</M> 的特征值，
              也就是说 <M>{'I - \\gamma P_\\pi'}</M> 不会奇异。
            </p>
            <p>
              更有意思的是它的逆有一个级数展开（Neumann 级数）：
            </p>
            <MB>{'(I - \\gamma P_\\pi)^{-1} = I + \\gamma P_\\pi + \\gamma^2 P_\\pi^2 + \\gamma^3 P_\\pi^3 + \\cdots'}</MB>
            <p>
              把它代回 <M>{'v_\\pi = (I-\\gamma P_\\pi)^{-1} r_\\pi'}</M> 你会发现，
              第 <M>{'k'}</M> 项正是「<M>{'k'}</M> 步之后的期望奖励，折扣 <M>{'\\gamma^k'}</M>」。
              <strong>这个矩阵求逆，其实就是把所有未来一层层加起来。</strong>
              解析解和定义在这里握了个手。
            </p>
          </Details>

          <Callout tone="trap" title="那为什么后面还要折腾迭代算法">
            求逆的代价是 <M>{'O(n^3)'}</M>，<M>{'n'}</M> 是状态数。
            网格世界 <M>{'n=25'}</M> 无所谓，围棋 <M>{'n \\approx 10^{170}'}</M> 就是笑话。
            更要命的是：<strong>求逆需要知道 <M>{'P_\\pi'}</M></strong>，
            而现实里我们通常拿不到环境模型。这两条，一条通向第 4 章，一条通向第 5 章。
          </Callout>
        </Beat>
      </Act>

      {/* ───────────────────────── 第 5 幕 ───────────────────────── */}
      <Act
        id="a5"
        no="第 5 幕"
        title="迭代求解，和压缩映射的现场证据"
        goal="看着误差以 γᵏ 的速度指数塌缩 —— 这就是压缩映射定理的样子。"
        minutes={12}
        points={[
          <>
            实用解法：把方程改写成迭代 <M>{'v_{k+1} = r_\\pi + \\gamma P_\\pi v_k'}</M>，
            从任意初值出发反复代入。
          </>,
          <>
            误差满足 <M>{'\\|v_k-v_\\pi\\|_\\infty \\le \\gamma^k\\|v_0-v_\\pi\\|_\\infty'}</M>
            —— 半对数图上是一条<strong>直线</strong>，斜率就是 <M>{'\\log\\gamma'}</M>。
          </>,
          <>
            γ 越大，未来看得越远，但收敛越慢。这个权衡会贯穿整门课。
          </>,
        ]}
        stage={() => (
          <div className="space-y-5">
            <Panel title={`第 ${kk} 次迭代时的世界`}>
              <GridWorld
                mdp={mdp}
                v={vk}
                policy={pi}
                cell={54}
                onCellClick={(x) => setFocus(x === focus ? null : x)}
              />
              <div className="mt-4">
                <Scrubber k={kk} setK={setK} max={trace.length - 1} />
              </div>
              <div className="mt-3 flex items-center justify-between">
                <span className="font-mono text-[12px] text-dim">
                  ‖v<sub>k</sub> − v<sub>π</sub>‖<sub>∞</sub> ={' '}
                  <span style={{ color: C.accent }}>{fmt(errors[kk], 4)}</span>
                </span>
                {policySeg}
              </div>
            </Panel>

            <Panel title={`第 ${kk} 步的贝尔曼更新，代入真实数值`}>
              <LiveFormula node={F_BELLMAN} ctx={ctxK} mode="numeric" />
              <p className="mt-3 border-t border-line pt-3 text-[11.5px] leading-relaxed text-faint">
                这就是「算法调试器」：拖时间轴，公式里的每一个数都跟着变。
                <strong>你看到的不是收敛的结果，而是收敛发生的过程。</strong>
              </p>
            </Panel>

            <Panel
              title="误差随迭代的衰减"
              right={<Toggle label="对数纵轴" checked={logAxis} onChange={setLogAxis} />}
            >
              <LineChart
                series={[
                  { name: '实际误差 ‖v_k − v_π‖∞', color: C.accent, data: errors },
                  { name: '理论上界 γᵏ·‖v_0 − v_π‖∞', color: C.gamma, data: bound, dashed: true },
                ]}
                xLabel="迭代次数 k"
                logY={logAxis}
                marker={kk}
                height={210}
              />
              <p className="mt-2 text-[11.5px] leading-relaxed text-faint">
                对数纵轴上是一条直线 —— <strong>直线就是指数衰减的签名</strong>，
                斜率由 γ 决定。把 γ 拖大，直线会变平（收敛变慢）。
              </p>
            </Panel>

            <Panel title="两个任意初值之间的距离">
              <LineChart
                series={[
                  { name: '‖v_k^A − v_k^B‖∞', color: C.qvalue, data: witness.distances },
                  { name: 'γᵏ · 初始距离', color: C.gamma, data: witness.bound, dashed: true },
                ]}
                xLabel="迭代次数 k"
                logY={logAxis}
                height={180}
              />
              <p className="mt-2 text-[11.5px] leading-relaxed text-faint">
                一个从全 0 出发，一个从 ±8 交替出发。它们之间的距离被死死压在 γᵏ 之下 ——
                <strong>这就是「压缩」两个字的字面意思。</strong>
              </p>
            </Panel>
          </div>
        )}
      >
        <Beat id="f1">
          <p>
            既然不想求逆，就换个办法：<strong>随便猜一个 <M>{'v_0'}</M>，然后反复代入。</strong>
          </p>
          <MB>{'v_{k+1} = r_\\pi + \\gamma P_\\pi v_k, \\qquad k = 0, 1, 2, \\dots'}</MB>
          <p>
            右边用旧的估计，算出一个新的估计。听上去像是在原地打转，
            但右边的时间轴会告诉你发生了什么：拖动它，看着整个价值场从一片漆黑逐渐点亮，
            最后稳定下来不再动。
          </p>
          <Callout tone="question" title="先猜一下">
            这个过程有可能<strong>不收敛</strong>吗？比如卡在两个值之间反复横跳，
            或者越来越大？在往下读之前，先自己判断一下。
          </Callout>
        </Beat>

        <Beat id="f2">
          <h3>不会。而且收敛得非常快</h3>
          <p>
            把 <M>{'v_{k+1} = r_\\pi + \\gamma P_\\pi v_k'}</M> 减去{' '}
            <M>{'v_\\pi = r_\\pi + \\gamma P_\\pi v_\\pi'}</M>，常数项抵消：
          </p>
          <MB>{'v_{k+1} - v_\\pi = \\gamma P_\\pi (v_k - v_\\pi)'}</MB>
          <p>
            两边取无穷范数，利用 <M>{'\\|P_\\pi\\|_\\infty = 1'}</M>：
          </p>
          <MB>{'\\| v_{k+1} - v_\\pi \\|_\\infty \\le \\gamma \\, \\| v_k - v_\\pi \\|_\\infty'}</MB>
          <p>
            <strong>每迭代一次，误差至少缩小到原来的 <M>{'\\gamma'}</M> 倍。</strong>
            递推下去就是 <M>{'\\|v_k - v_\\pi\\|_\\infty \\le \\gamma^k \\|v_0 - v_\\pi\\|_\\infty \\to 0'}</M>。
          </p>
          <p>
            右边第二张图就是这条不等式的现场录像：实线是实际误差，虚线是理论上界{' '}
            <M>{'\\gamma^k'}</M>。<em>实线永远压在虚线下面，一次都没越界。</em>
          </p>
        </Beat>

        <Beat id="f3">
          <h3>这有个名字：压缩映射</h3>
          <p>
            把 <M>{'f(v) = r_\\pi + \\gamma P_\\pi v'}</M> 看成一个「把向量搬到另一个向量」的映射。
            上面证明的其实是：
          </p>
          <MB>{'\\| f(v_1) - f(v_2) \\|_\\infty \\le \\gamma \\| v_1 - v_2 \\|_\\infty'}</MB>
          <p>
            <strong>任意两个点，经过 <M>{'f'}</M> 之后都会靠得更近。</strong>
            这样的映射叫压缩映射，Banach 不动点定理保证它有且只有一个不动点，
            而且从任何初值出发反复迭代都会收敛到它。
          </p>
          <p>
            右边第三张图就是在验证这件事：两条从完全不同初值出发的轨迹，
            它们之间的距离被 <M>{'\\gamma^k'}</M> 死死压住。
          </p>
          <Callout tone="insight" title="记住这把锤子">
            压缩映射会在这本书里出现<strong>至少三次</strong>：这里（策略评估）、
            第 3 章（贝尔曼最优公式有唯一解）、第 4 章（值迭代收敛）。
            每一次，靠的都是同一个 <M>{'\\gamma < 1'}</M>。
            <em>先把它认熟，后面能省下很多力气。</em>
          </Callout>

          <PredictNumber
            id="ch2-iters"
            question={
              <>
                取 <M>{'\\gamma = 0.9'}</M>。要让误差降到初始误差的{' '}
                <M>{'1/1000'}</M>，大约需要迭代多少次？
              </>
            }
            min={0}
            max={200}
            step={1}
            truth={Math.ceil(Math.log(0.001) / Math.log(0.9))}
            tolerance={12}
            unit=" 次"
            explain={
              <>
                <p>
                  解 <M>{'0.9^k \\le 10^{-3}'}</M>，得{' '}
                  <M>{'k \\ge \\dfrac{\\ln 10^{-3}}{\\ln 0.9} \\approx 66'}</M>。
                </p>
                <p>
                  这个估算很实用：<strong>误差每降一个数量级，需要{' '}
                  <M>{'\\ln 10 / \\ln(1/\\gamma)'}</M> 次迭代</strong>。
                  <M>{'\\gamma = 0.99'}</M> 时这个数字会暴涨到约 229 次 ——
                  所以「智能体更有远见」在计算上是要付学费的。
                  把右边的 γ 拖到 0.99 再看那条对数直线，它会明显变平。
                </p>
              </>
            }
          />
        </Beat>

        <Beat id="f4">
          <h3>算法长这样</h3>
          <Code
            lang="python"
            code={`def policy_evaluation(pi, P, R, gamma, tol=1e-8, max_iter=10_000):
    """迭代法求解 v_π = r_π + γ P_π v_π"""
    n = P.shape[0]
    v = np.zeros(n)                       # 初值随便取，收敛性与它无关
    for k in range(max_iter):
        q = R + gamma * P @ v             # (nS, nA)
        v_new = (pi * q).sum(axis=1)      # 一次贝尔曼更新
        if np.max(np.abs(v_new - v)) < tol:
            return v_new, k
        v = v_new
    return v, max_iter`}
          />
          <p>
            注意 <code>v = np.zeros(n)</code> 这一行的注释：
            <strong>初值真的可以随便取。</strong>压缩映射不在乎你从哪里出发，
            它只保证你一定会到同一个地方。这是理论给出的、非常实在的工程许可。
          </p>
        </Beat>
      </Act>

      {/* ───────────────────────── 第 6 幕 ───────────────────────── */}
      <Act
        id="a6"
        no="第 6 幕"
        title="动作价值 q_π：多问一句「如果我这一步不听话呢」"
        goal="v 和 q 的分工，以及为什么后面所有算法都围着 q 转。"
        minutes={9}
        points={[
          <>
            <M>{'q_\\pi(s,a)'}</M>：在 <M>{'s'}</M> <strong>强行</strong>做 <M>{'a'}</M>，
            之后再回到 π。它比 v 多问了「这一步不听话会怎样」。
          </>,
          <>
            两者互相表示：<M>{'v_\\pi(s)=\\sum_a \\pi(a|s) q_\\pi(s,a)'}</M>，
            <M>{"q_\\pi(s,a)=r(s,a)+\\gamma\\sum_{s'}p(s'|s,a)v_\\pi(s')"}</M>。
          </>,
          <>
            <strong>为什么后面全在学 q</strong>：只有 q 能在不知道环境模型的情况下
            直接告诉你「该选哪个动作」。用 v 选动作还得额外知道 <M>{'p'}</M>。
          </>,
        ]}
        stage={() => (
          <div className="space-y-5">
            <Panel title="v 与 q 的关系">
              <LiveFormula node={F_Q} ctx={ctx} />
              <div className="my-2 border-t border-line" />
              <LiveFormula node={F_VQ} ctx={ctx} />
            </Panel>
            {worldPanel(vStar)}
          </div>
        )}
      >
        <Beat id="g1">
          <p>
            <M>{'v_\\pi(s)'}</M> 回答的是「待在这个状态有多好」。
            但要<strong>改进</strong>策略，这个问题问得不够细 ——
            我们真正想知道的是「在这个状态<em>做这个动作</em>有多好」。
          </p>
          <p>于是定义动作价值：</p>
          <MB>{'q_\\pi(s,a) = \\mathbb{E}\\!\\left[G_t \\mid S_t = s,\\ A_t = a\\right]'}</MB>
          <p>
            差别只有一处：<strong>第一步的动作被强行指定为 <M>{'a'}</M>，从第二步起才回到策略 <M>{'\\pi'}</M>。</strong>
            这一点「不听话」正是策略改进的全部空间所在。
          </p>
        </Beat>

        <Beat id="g2">
          <h3>两条互相翻译的等式</h3>
          <p>
            打开右边世界面板里的「显示 q(s,a)」，每个格子的四周会出现五个粉色小数字。
            它们和格子中间那个绿色的 <M>{'v'}</M> 之间，有两条对偶的关系：
          </p>
          <MB>{'v_\\pi(s) = \\sum_a \\pi(a\\mid s)\\, q_\\pi(s,a)'}</MB>
          <MB>{'q_\\pi(s,a) = r(s,a) + \\gamma \\sum_{s\'} p(s\'\\mid s,a)\\, v_\\pi(s\')'}</MB>
          <p>
            上面一条：<strong>v 是 q 按策略加权的平均</strong>。
            下面一条：<strong>q 是即时奖励加上后继状态 v 的折扣期望</strong>。
            两条合起来，就是贝尔曼公式 —— 把第二条代进第一条，你会原样得到第 2 幕那个式子。
          </p>

          <PredictChoice
            id="ch2-vq"
            question={
              <>
                在均匀随机策略下，某个状态的 <M>{'v_\\pi(s) = 2.0'}</M>，
                而它五个动作的 <M>{'q_\\pi(s,a)'}</M> 分别是{' '}
                <M>{'3.5,\\ 2.0,\\ 1.0,\\ 2.5,\\ 1.0'}</M>。这组数说明了什么？
              </>
            }
            options={[
              { id: 'a', label: '数据有矛盾，v 应该等于 q 的最大值 3.5。' },
              {
                id: 'b',
                label: '一切正常：v 是这五个 q 的平均值，而且有动作比平均值更好 —— 策略有改进空间。',
              },
              { id: 'c', label: '一切正常，但说明不了任何关于策略好坏的事。' },
            ]}
            answer="b"
            explain={
              <>
                <p>
                  <M>{'(3.5+2.0+1.0+2.5+1.0)/5 = 2.0'}</M>，与 <M>{'v_\\pi(s)'}</M> 完全吻合。
                </p>
                <p>
                  更重要的是那个 <M>{'3.5 > 2.0'}</M>。它意味着：
                  <strong>如果把这个状态的策略改成「总是选第一个动作」，价值就会上升。</strong>
                </p>
                <p>
                  这个观察有个名字叫<strong>策略改进定理</strong>，它是第 3、4 章的引擎。
                  你已经提前摸到它了 —— <em>只要 q 里有比 v 大的项，当前策略就不是最优的</em>。
                </p>
              </>
            }
          />
        </Beat>

        <Beat id="g3">
          <Callout tone="intuition" title="这一章的一句话">
            贝尔曼公式把「对无穷多条未来轨迹求平均」这件不可能的事，
            换成了「解一个有限维的线性方程组」。代价是你必须知道环境模型；
            回报是你从此拥有了一把能给任何策略打分的尺子。
          </Callout>
          <p>
            现在你能给<strong>任意一个</strong>策略打分了。
            那么下一个问题就无法回避了：<strong>所有策略里，最好的那个长什么样？</strong>
          </p>
          <p>
            注意这个问题比看上去麻烦。策略空间大得吓人（25 个状态、5 个动作，
            光是确定性策略就有 <M>{'5^{25} \\approx 3 \\times 10^{17}'}</M> 个），
            总不能一个个打分比大小。而且「更好」到底是什么意思 —— 
            如果策略 A 在某些状态更好、策略 B 在另一些状态更好呢？
          </p>
        </Beat>
      </Act>
    </div>
  )
}
