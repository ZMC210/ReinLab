import { useMemo, useState } from 'react'
import {
  RM_ROOT,
  STEP_CHECK,
  STEP_LABEL,
  gradientDescentDemo,
  meanEstimation,
  robbinsMonro,
  sampleCloud,
  stepSize,
  type GDKind,
  type StepRule,
} from '../core/approx'
import { Act, Beat } from '../narrative/Act'
import { ChapterGlance, ChapterHero, type Glance } from '../narrative/ChapterShell'
import { PredictChoice } from '../narrative/Predict'
import { LiveFormula } from '../formula/LiveFormula'
import { useFormulaCtx } from '../formula/core'
import { incrementalMean } from '../formula/bellman'
import { buildGridMDP, symmetricGrid } from '../core/mdp'
import { uniformPolicy } from '../core/policy'
import { LineChart } from '../viz/LineChart'
import { Callout, Code, Details, M, MB, Panel, Seg, Slider, Stat } from '../ui/prims'
import { fmt, useColors } from '../theme'

const F_INC = incrementalMean()

const GLANCE: Glance = {
  formula: String.raw`w_{k+1} = w_k + \alpha_k \left( \tilde{x}_k - w_k \right)
  \qquad\text{或更一般地}\qquad
  w_{k+1} = w_k - \alpha_k \, \tilde{g}(w_k, \eta_k)`,
  formulaNote:
    '「旧估计 + 步长 × 误差」。这一行是整本书后半部分的骨架 —— TD、Sarsa、Q-learning、策略梯度，全都是它换了个误差项。',
  takeaways: [
    <>
      增量式均值 <M>{'w_{k+1}=w_k+\\frac1k(x_k-w_k)'}</M> 与批量求平均<strong>数学上完全等价</strong>，
      但它不需要存历史数据，而且随时可以用。
    </>,
    <>
      Robbins-Monro 定理：只观测到 <M>{'g(w)'}</M> 的带噪值，也能求出 <M>{'g(w)=0'}</M> 的根。
      条件是步长满足 <M>{'\\sum\\alpha_k=\\infty'}</M> 且 <M>{'\\sum\\alpha_k^2<\\infty'}</M>。
    </>,
    <>
      两个条件各管一件事：<M>{'\\sum\\alpha_k=\\infty'}</M> 保证<strong>走得到</strong>
      （不会半路停下），<M>{'\\sum\\alpha_k^2<\\infty'}</M> 保证<strong>停得住</strong>（噪声被压下去）。
    </>,
    <>
      SGD 是 RM 的特例（令 <M>{'g=\\nabla J'}</M>）；而均值估计又是 SGD 的特例。
      三者是同一件事的三种说法。
    </>,
  ],
  traps: [
    <>
      以为常数步长「不收敛」所以没用。恰恰相反：非平稳环境里常数步长是<strong>刻意的选择</strong>，
      因为它让估计一直保持对新数据的敏感。
    </>,
    <>
      把 <M>{'\\alpha_k=1/k^2'}</M> 当成「更保守所以更安全」。它连真值都走不到 ——
      步长衰减得太快，总位移是有限的。
    </>,
    <>
      SGD 的随机性看起来像缺点。但在离最优点远的时候，
      SGD 的方向<strong>和批量梯度几乎一样</strong>，所以前期它几乎不吃亏。
    </>,
  ],
}

export function Chapter6() {
  const C = useColors()
  const [rule, setRule] = useState<StepRule>('inv-k')
  const [n, setN] = useState(200)
  const [gd, setGd] = useState<GDKind>('sgd')
  const [alpha, setAlpha] = useState(0.12)
  const [batch, setBatch] = useState(8)

  /* 用一个只有 9 格的小世界给「活的公式」当上下文 */
  const tinyMdp = useMemo(() => buildGridMDP(symmetricGrid()), [])
  const tinyPi = useMemo(() => uniformPolicy(tinyMdp.nS), [tinyMdp.nS])
  const zeros = useMemo(() => new Array<number>(tinyMdp.nS).fill(0), [tinyMdp.nS])
  const ctx = useFormulaCtx(tinyMdp, tinyPi, 0.9, zeros, 0)

  /* ── 均值估计 ── */
  const me = useMemo(() => meanEstimation(rule, 400, { mean: 3, noise: 1.4, w0: 10 }), [rule])
  const meAll = useMemo(
    () =>
      (['inv-k', 'const', 'inv-sqrt', 'inv-k2'] as StepRule[]).map((r) => ({
        r,
        w: meanEstimation(r, 400, { mean: 3, noise: 1.4, w0: 10 }).w,
      })),
    [],
  )

  /* ── Robbins-Monro ── */
  const rm = useMemo(() => robbinsMonro(rule, 400, { w0: 3, noise: 1.2 }), [rule])
  const rmErr = Math.abs(rm.w[Math.min(n, rm.w.length - 1)] - RM_ROOT)

  /* 步长的两个和，直接算出来给学生看 */
  const sums = useMemo(() => {
    let s1 = 0
    let s2 = 0
    for (let k = 1; k <= 2000; k++) {
      const a = stepSize(rule, k)
      s1 += a
      s2 += a * a
    }
    return { s1, s2 }
  }, [rule])

  /* ── 梯度下降三兄弟 ── */
  const cloud = useMemo(() => sampleCloud(120), [])
  const truth = useMemo(
    () => ({
      x: cloud.reduce((a, d) => a + d.x, 0) / cloud.length,
      y: cloud.reduce((a, d) => a + d.y, 0) / cloud.length,
    }),
    [cloud],
  )
  const paths = useMemo(
    () =>
      (['bgd', 'mbgd', 'sgd'] as GDKind[]).map((kind) => ({
        kind,
        path: gradientDescentDemo(kind, { data: cloud, steps: 120, alpha, batch }).path,
      })),
    [cloud, alpha, batch],
  )
  const distTo = (p: [number, number][]) =>
    p.map(([x, y]) => Math.hypot(x - truth.x, y - truth.y))

  return (
    <div>
      <ChapterHero
        n={6}
        hook="必须等一整条轨迹结束才能更新，而且估计值抖得厉害。"
        lead={
          <>
            <p>
              这一章是全书唯一一章<strong>不出现 MDP</strong> 的。
              它要办的是一张许可证：凭什么「拿一个带噪声的样本，往前挪一小步」，
              重复无穷多次之后就能落在正确答案上？
            </p>
            <p>
              听上去像数学插曲，其实是全书的枢纽。
              办下这张证之后，第 7 章只需要把它套回贝尔曼公式，
              时序差分算法就<em>直接掉出来了</em>。
            </p>
          </>
        }
        gains={[
          '把「求平均」改写成增量式，并看出它就是后面所有更新式的模板',
          '理解 Robbins-Monro 定理在保证什么，两个步长条件各自管什么',
          '亲手把步长换成 1/k²、常数、1/√k，看它们各自怎么失败',
          '看清 SGD、MBGD、BGD 只是「一次用几个样本」的区别',
          '明白为什么 SGD 在远离最优点时几乎不吃亏',
        ]}
      />

      <ChapterGlance g={GLANCE} />

      {/* ───────────────────── 第 1 幕 ───────────────────── */}
      <Act
        id="a1"
        no="第 1 幕"
        title="把求平均改写一遍"
        goal="增量式均值：一个看起来平平无奇的恒等变形，却是后半本书的模板。"
        minutes={7}
        points={[
          <>
            <M>{'w_{k+1} = w_k + \\frac1k(x_k - w_k)'}</M> 与{' '}
            <M>{'w_{k+1}=\\frac1k\\sum_{i\\le k} x_i'}</M> <strong>完全等价</strong>，
            不是近似。
          </>,
          <>
            好处是<strong>不必存历史数据</strong>，而且中途任何时刻都有一个可用的估计。
          </>,
          <>
            把 <M>{'1/k'}</M> 换成一般的 <M>{'\\alpha_k'}</M>，就打开了通往整个随机近似理论的门。
          </>,
        ]}
        stage={() => (
          <div className="space-y-4">
            <Panel title="活的公式">
              <LiveFormula node={F_INC} ctx={ctx} />
              <p className="mt-3 border-t border-line pt-3 text-[11.5px] leading-relaxed text-faint">
                「新估计 = 旧估计 + 步长 × 误差」。把这三块记住 ——
                之后 TD、Sarsa、Q-learning、策略梯度，全是在换那个「误差」。
              </p>
            </Panel>
            <Panel title="增量式 vs 批量：两条线完全重合">
              <LineChart
                height={190}
                xLabel="样本序号 k"
                series={[
                  { name: '增量式（α=1/k）', color: C.accent, data: meanEstimation('inv-k', 200, { mean: 3, noise: 1.4, w0: 10 }).w },
                  {
                    name: '批量求平均',
                    color: C.value,
                    data: meanEstimation('inv-k', 200, { mean: 3, noise: 1.4, w0: 10 }).batch,
                    dashed: true,
                  },
                ]}
              />
              <p className="mt-2 text-[11.5px] leading-relaxed text-faint">
                两条线叠在一起看不出差别 —— 因为它们本来就是同一个数列的两种写法。
              </p>
            </Panel>
          </div>
        )}
      >
        <Beat id="b1" keep>
          <p>
            上一章末尾出现了这一行：<code>q += (G - q) / cnt</code>。
            现在把它单独拎出来。设 <M>{'w_{k}'}</M> 是前 <M>{'k-1'}</M> 个样本的平均：
          </p>
          <MB>{'w_{k} = \\frac{1}{k-1}\\sum_{i=1}^{k-1} x_i'}</MB>
          <p>那么加入第 <M>{'k'}</M> 个样本之后</p>
          <MB>
            {String.raw`w_{k+1} = \frac{1}{k}\sum_{i=1}^{k} x_i
            = \frac{1}{k}\left[(k-1)w_k + x_k\right]
            = w_k + \frac{1}{k}\left(x_k - w_k\right)`}
          </MB>
          <p>
            这是<strong>恒等变形</strong>，不是近似。但换个写法之后，整件事的气质变了：
            它从「统计一批数据」变成了「<em>一个不断自我修正的过程</em>」。
          </p>
        </Beat>

        <Beat id="b2" keep>
          <Callout tone="insight" title="三块积木">
            <p>
              <M>{'\\underbrace{w_{k+1}}_{\\text{新估计}} = \\underbrace{w_k}_{\\text{旧估计}} + \\underbrace{\\alpha_k}_{\\text{挪多远}}\\underbrace{(x_k - w_k)}_{\\text{往哪挪}}'}</M>
            </p>
            <p>
              后面每一个算法都长这个样子，区别只在「误差」这一块写的是什么：
            </p>
            <ul>
              <li>误差 = <M>{'x_k - w_k'}</M> → 均值估计</li>
              <li>误差 = <M>{"r + \\gamma v(s') - v(s)"}</M> → TD</li>
              <li>误差 = <M>{"r + \\gamma\\max_a q(s',a) - q(s,a)"}</M> → Q-learning</li>
              <li>误差 = <M>{'\\nabla\\ln\\pi \\cdot q'}</M> → 策略梯度</li>
            </ul>
            <p>
              所以这一幕看着简单，实际上是在给你一个<strong>读后面所有算法的模板</strong>。
            </p>
          </Callout>
        </Beat>
      </Act>

      {/* ───────────────────── 第 2 幕 ───────────────────── */}
      <Act
        id="a2"
        no="第 2 幕"
        title="步长：两个条件，两种失败"
        goal="Robbins-Monro 的两个步长条件，各自防的是哪一种失败。"
        minutes={11}
        points={[
          <>
            <M>{'\\sum_k \\alpha_k = \\infty'}</M>：总位移必须无穷 ——
            否则从任意初值出发都<strong>走不到</strong>真值。
          </>,
          <>
            <M>{'\\sum_k \\alpha_k^2 < \\infty'}</M>：噪声的累积影响必须有限 ——
            否则会一直被推着<strong>抖个不停</strong>。
          </>,
          <>
            <M>{'\\alpha_k = 1/k'}</M> 恰好卡在两个条件的交集里，
            这不是巧合，是调和级数的性质。
          </>,
          <>
            工程上常用常数步长，明知不收敛也用 —— 因为在非平稳环境里，
            <strong>「忘掉旧数据」本身就是需求</strong>。
          </>,
        ]}
        stage={() => (
          <div className="space-y-4">
            <Panel title="换一个步长规则">
              <Seg
                value={rule}
                onChange={setRule}
                size="sm"
                options={(['inv-k', 'const', 'inv-sqrt', 'inv-k2'] as StepRule[]).map((r) => ({
                  value: r,
                  label: STEP_LABEL[r],
                }))}
              />
              <div className="mt-3 grid grid-cols-2 gap-2.5">
                <Stat
                  label={<>Σαₖ（前 2000 项）</>}
                  value={fmt(sums.s1, 1)}
                  color={STEP_CHECK[rule].sum ? C.value : C.danger}
                  hint="要发散才好"
                />
                <Stat
                  label={<>Σαₖ²（前 2000 项）</>}
                  value={fmt(sums.s2, 2)}
                  color={STEP_CHECK[rule].sq ? C.value : C.danger}
                  hint="要收敛才好"
                />
              </div>
              <p className="mt-3 text-[12px] leading-relaxed text-dim">
                {STEP_CHECK[rule].verdict}
              </p>
            </Panel>

            <Panel title="均值估计：真值 = 3，初值 = 10">
              <LineChart
                height={186}
                xLabel="样本序号 k"
                marker={Math.min(n, 399)}
                series={[
                  { name: STEP_LABEL[rule], color: C.accent, data: me.w },
                  { name: '真值', color: C.value, data: new Array(401).fill(3), dashed: true },
                ]}
              />
            </Panel>

            <Panel title="Robbins-Monro：求 w³ − 5 = 0 的根">
              <LineChart
                height={186}
                xLabel="迭代 k"
                marker={Math.min(n, rm.w.length - 1)}
                series={[
                  { name: 'w_k', color: C.qvalue, data: rm.w },
                  {
                    name: `真根 ≈ ${fmt(RM_ROOT, 3)}`,
                    color: C.value,
                    data: new Array(rm.w.length).fill(RM_ROOT),
                    dashed: true,
                  },
                ]}
              />
              <div className="mt-3">
                <Slider
                  label="看到第几步"
                  value={n}
                  min={10}
                  max={399}
                  step={1}
                  onChange={setN}
                  format={(v) => String(v)}
                  hint={`此刻误差 = ${fmt(rmErr, 4)}`}
                />
              </div>
            </Panel>

            <Panel title="四条规则同框">
              <LineChart
                height={186}
                xLabel="样本序号 k"
                yMin={2}
                yMax={10.5}
                series={meAll.map((m, i) => ({
                  name: STEP_LABEL[m.r].split('（')[0],
                  color: [C.accent, C.reward, C.policy, C.danger][i],
                  data: m.w,
                  dashed: m.r !== rule,
                  width: m.r === rule ? 2.6 : 1.4,
                }))}
              />
            </Panel>
          </div>
        )}
      >
        <Beat id="c1" keep>
          <p>
            现在放开手脚：把 <M>{'1/k'}</M> 换成任意的 <M>{'\\alpha_k'}</M>。
            问题立刻变得不平凡 —— <strong>什么样的步长序列能保证收敛？</strong>
          </p>
          <p>Robbins 与 Monro 在 1951 年给出了答案。他们考虑的问题更一般：</p>
          <MB>{'\\text{求 } g(w) = 0 \\text{ 的根，但你只能观测到 } \\tilde g(w,\\eta) = g(w) + \\eta'}</MB>
          <p>
            连 <M>{'g'}</M> 的表达式都不知道，只能把 <M>{'w'}</M> 递进去，
            拿回一个带噪声的读数。这正是强化学习的处境。
          </p>
        </Beat>

        <Beat id="c2" keep>
          <p>RM 算法本身只有一行：</p>
          <MB>{'w_{k+1} = w_k - \\alpha_k \\, \\tilde g(w_k, \\eta_k)'}</MB>
          <p>而收敛的充分条件是三条，其中两条是关于步长的：</p>
          <MB>
            {String.raw`\sum_{k=1}^{\infty}\alpha_k = \infty
            \qquad\text{且}\qquad
            \sum_{k=1}^{\infty}\alpha_k^2 < \infty`}
          </MB>
        </Beat>

        <Beat id="c3" keep>
          <PredictChoice
            id="ch6-step"
            question={
              <>
                把步长换成 <M>{'\\alpha_k = 1/k^2'}</M>（衰减得比 1/k 更快、看起来更稳），
                会发生什么？
              </>
            }
            options={[
              { id: 'a', label: '收敛得更快 —— 步子小，更稳' },
              { id: 'b', label: '收敛得更慢，但最终还是能到' },
              { id: 'c', label: '根本到不了真值，会卡在半路上' },
            ]}
            answer="c"
            explain={
              <>
                <p>
                  在右边把规则切成 <M>{'\\alpha_k=1/k^2'}</M> 看看 —— 曲线从 10 出发，
                  掉到 8 附近就<strong>不动了</strong>。
                </p>
                <p>
                  原因很朴素：<M>{'\\sum 1/k^2 = \\pi^2/6 \\approx 1.64'}</M> 是<strong>有限的</strong>。
                  也就是说这个算法这辈子能挪动的总距离不超过某个常数乘以 1.64。
                  如果初值离真值比这还远，它<em>物理上就走不到</em>。
                </p>
                <p>
                  这就是 <M>{'\\sum\\alpha_k=\\infty'}</M> 的含义：
                  它不是什么技术条件，它是在说「你得有足够的燃料飞到任何地方」。
                </p>
              </>
            }
          />
        </Beat>

        <Beat id="c4">
          <Details summary="展开：另一个条件 Σαₖ² < ∞ 在防什么">
            <p>
              把更新式展开，<M>{'w_{k+1}'}</M> 里累积的噪声大致是{' '}
              <M>{'\\sum_k \\alpha_k \\eta_k'}</M>。若噪声独立、方差有界，
              这个和的方差正比于 <M>{'\\sum_k\\alpha_k^2'}</M>。
            </p>
            <p>
              所以 <M>{'\\sum\\alpha_k^2<\\infty'}</M> 说的是：
              <strong>噪声的总影响必须是有限的</strong>，否则估计会被噪声永远推着走，
              停不下来。
            </p>
            <p>
              常数步长就属于这一类：<M>{'\\sum \\alpha^2 = \\infty'}</M>，
              于是它只能收敛到真值附近的一个<strong>邻域</strong>里来回抖 ——
              把右边切到「α = 0.1」，那条曲线到了 3 附近就一直在抖，正是这个道理。
            </p>
            <p>
              有意思的是，工程上大量使用常数步长，<strong>明知它不收敛</strong>。
              因为在非平稳问题里（环境会变、策略在变），
              「永远对新数据保持敏感」比「收敛到某个旧答案」更值钱。
            </p>
          </Details>
        </Beat>

        <Beat id="c5">
          <Callout tone="rigor" title="第三个条件">
            <p>
              完整的 RM 定理还要求 <M>{'g(w)'}</M> 单调递增且梯度有界：
              <M>{'0 < c_1 \\le \\nabla_w g(w) \\le c_2'}</M>。
              它保证根唯一，而且函数不会陡到让算法跳过去。
            </p>
            <p>
              这个条件在强化学习里往往可以放宽 —— 后面 TD 的收敛性证明用的是随机近似的
              变体（ODE 方法），但两个步长条件是共通的。
            </p>
          </Callout>
        </Beat>
      </Act>

      {/* ───────────────────── 第 3 幕 ───────────────────── */}
      <Act
        id="a3"
        no="第 3 幕"
        title="SGD：一次只看一个样本"
        goal="BGD / MBGD / SGD 只差「每步用几个样本」，而 SGD 在远处几乎不吃亏。"
        minutes={9}
        points={[
          <>
            目标 <M>{'J(w)=\\mathbb{E}[f(w,X)]'}</M>，
            真梯度算不出来（分布未知），就用样本梯度顶上去。
          </>,
          <>
            <strong>BGD</strong> 用全部样本、<strong>MBGD</strong> 用一小批、
            <strong>SGD</strong> 只用一个。三者是同一个式子里 <M>{'m=n,\\,m,\\,1'}</M>。
          </>,
          <>
            关键观察：<strong>离最优点远时，SGD 的方向和真梯度几乎一致</strong>；
            只有靠近最优点，随机性才开始主导。所以 SGD 前期不吃亏、后期才抖。
          </>,
        ]}
        stage={() => (
          <div className="space-y-4">
            <Panel title="下降路径（目标：走到样本云的中心）">
              <ScatterPath cloud={cloud} paths={paths} truth={truth} active={gd} />
              <div className="mt-3">
                <Seg
                  value={gd}
                  onChange={setGd}
                  size="sm"
                  options={[
                    { value: 'bgd', label: 'BGD（全部 120 个）' },
                    { value: 'mbgd', label: `MBGD（${batch} 个）` },
                    { value: 'sgd', label: 'SGD（1 个）' },
                  ]}
                />
              </div>
            </Panel>

            <Panel title="到最优点的距离">
              <LineChart
                height={180}
                logY
                xLabel="迭代步"
                series={paths.map((p, i) => ({
                  name: p.kind.toUpperCase(),
                  color: [C.value, C.reward, C.qvalue][i],
                  data: distTo(p.path),
                  width: p.kind === gd ? 2.6 : 1.3,
                  dashed: p.kind !== gd,
                }))}
              />
            </Panel>

            <Panel title="参数">
              <div className="space-y-3">
                <Slider
                  label={<>学习率 <M>{'\\alpha'}</M></>}
                  value={alpha}
                  min={0.01}
                  max={0.4}
                  step={0.01}
                  onChange={setAlpha}
                  accent={C.gamma}
                />
                <Slider
                  label="mini-batch 大小"
                  value={batch}
                  min={1}
                  max={40}
                  step={1}
                  onChange={setBatch}
                  format={(v) => String(v)}
                  hint="调到 1 就是 SGD，调到 120 就是 BGD"
                />
              </div>
            </Panel>
          </div>
        )}
      >
        <Beat id="d1" keep>
          <p>
            把 RM 用在一个具体问题上：最小化 <M>{'J(w) = \\mathbb{E}[f(w, X)]'}</M>。
            最优点满足 <M>{'\\nabla J(w) = 0'}</M>，这正好是一个「求根」问题，
            于是 RM 直接适用 —— 只要能拿到 <M>{'\\nabla J'}</M> 的带噪估计。
          </p>
          <p>
            而最省事的带噪估计就是<strong>只用一个样本</strong>：
          </p>
          <MB>{'w_{k+1} = w_k - \\alpha_k \\nabla_w f(w_k, x_k)'}</MB>
          <p>这就是随机梯度下降。它是 RM 的特例，不是另一套理论。</p>
        </Beat>

        <Beat id="d2" keep>
          <p>三兄弟的差别只有一个数字：</p>
          <MB>
            {String.raw`w_{k+1} = w_k - \alpha_k \cdot \frac{1}{m}\sum_{i=1}^{m}\nabla_w f(w_k, x_i),
            \qquad m = \begin{cases} n & \text{BGD} \\ m & \text{MBGD} \\ 1 & \text{SGD}\end{cases}`}
          </MB>
          <p>
            右边的散点图里，三条路径同时在往样本云的中心走。
            切换看看：SGD 的路是歪歪扭扭的，BGD 是笔直的 ——
            但它们<strong>到达的速度差得没有想象中大</strong>。
          </p>
        </Beat>

        <Beat id="d3" keep>
          <PredictChoice
            id="ch6-sgd"
            question="SGD 每步只用一个样本，方向必然很不准。那它在什么时候最不准？"
            options={[
              { id: 'a', label: '一直都不准，全程都在乱走' },
              { id: 'b', label: '刚开始最不准，离最优点越近越准' },
              { id: 'c', label: '刚开始几乎和真梯度一样，越靠近最优点越乱' },
            ]}
            answer="c"
            explain={
              <>
                <p>
                  设最优点是 <M>{'w^*=\\mathbb{E}[X]'}</M>。SGD 的方向是{' '}
                  <M>{'w_k - x_k'}</M>，真梯度是 <M>{'w_k - \\mathbb{E}[X]'}</M>。
                  两者的相对偏差是
                </p>
                <MB>{'\\delta_k = \\frac{\\left| (w_k - x_k) - (w_k - \\mathbb{E}[X]) \\right|}{\\left| w_k - \\mathbb{E}[X] \\right|} = \\frac{\\left| \\mathbb{E}[X] - x_k \\right|}{\\left| w_k - w^* \\right|}'}</MB>
                <p>
                  分子是样本自身的波动，与 <M>{'w_k'}</M> 无关。所以当{' '}
                  <M>{'w_k'}</M> 离 <M>{'w^*'}</M> <strong>很远</strong>时，
                  分母很大，相对偏差趋近 0 —— <em>SGD 的方向和真梯度几乎一样</em>。
                </p>
                <p>
                  这解释了 SGD 为什么在实践中这么好用：
                  <strong>前期几乎白拿了 n 倍的速度，代价只在最后收敛阶段的抖动上。</strong>
                  右边那张对数距离图能看得很清楚：三条线前半段几乎重合，后半段才分开。
                </p>
              </>
            }
          />
        </Beat>

        <Beat id="d4" keep>
          <Code
            code={`# 三兄弟其实是同一个函数
def gradient_descent(data, alpha, steps, m):
    w = init()
    for k in range(1, steps + 1):
        batch = data if m == len(data) else random.sample(data, m)
        grad  = sum(grad_f(w, x) for x in batch) / m
        w     = w - alpha * grad          # α 也可以按 1/k 衰减
    return w

# m = len(data)  -> BGD
# m = 8          -> MBGD
# m = 1          -> SGD`}
          />
        </Beat>

        <Beat id="d5" keep>
          <Callout tone="insight" title="这张许可证要拿去干什么">
            <p>
              回头看第 2 章的贝尔曼公式：
              <M>{"v(s) = \\mathbb{E}\\left[R + \\gamma v(S') \\mid s\\right]"}</M>。
              它本质上是一个「求根」问题：找一个 <M>{'v'}</M> 使得{' '}
              <M>{"g(v) = v(s) - \\mathbb{E}[R+\\gamma v(S')] = 0"}</M>。
            </p>
            <p>
              而我们能观测到的，恰好是它的带噪版本 ——
              一次真实的转移 <M>{"(s, r, s')"}</M> 给出{' '}
              <M>{"\\tilde g = v(s) - (r + \\gamma v(s'))"}</M>。
            </p>
            <p>
              把 RM 算法套上去：
              <M>{"v(s) \\leftarrow v(s) - \\alpha\\left[v(s) - (r+\\gamma v(s'))\\right]"}</M>。
            </p>
            <p>
              <strong>这就是 TD 算法。</strong>下一章不需要任何新想法，
              只需要把这一步写清楚。
            </p>
          </Callout>
        </Beat>
      </Act>
    </div>
  )
}

/* ────────────────────────── 散点 + 下降路径 ────────────────────────── */

function ScatterPath({
  cloud,
  paths,
  truth,
  active,
}: {
  cloud: { x: number; y: number }[]
  paths: { kind: GDKind; path: [number, number][] }[]
  truth: { x: number; y: number }
  active: GDKind
}) {
  const C = useColors()
  const W = 420
  const H = 300
  const lim = 10
  const tx = (x: number) => ((x + lim) / (2 * lim)) * W
  const ty = (y: number) => H - ((y + lim) / (2 * lim)) * H
  const colors: Record<GDKind, string> = { bgd: C.value, mbgd: C.reward, sgd: C.qvalue }

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
      <rect x={0} y={0} width={W} height={H} fill="var(--surface-2)" rx={10} />
      {cloud.map((d, i) => (
        <circle key={i} cx={tx(d.x)} cy={ty(d.y)} r={2} fill={C.state} opacity={0.28} />
      ))}
      {paths.map((p) => (
        <polyline
          key={p.kind}
          points={p.path.map(([x, y]) => `${tx(x)},${ty(y)}`).join(' ')}
          fill="none"
          stroke={colors[p.kind]}
          strokeWidth={p.kind === active ? 2.2 : 1}
          opacity={p.kind === active ? 1 : 0.35}
          strokeLinejoin="round"
        />
      ))}
      <circle cx={tx(truth.x)} cy={ty(truth.y)} r={5} fill="none" stroke={C.accent} strokeWidth={2} />
      <text x={tx(truth.x) + 9} y={ty(truth.y) + 4} fontSize={10.5} fill="var(--ink-faint)">
        E[X]
      </text>
    </svg>
  )
}
