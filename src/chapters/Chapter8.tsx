import { useMemo, useState } from 'react'
import { buildGridMDP, classicGrid } from '../core/mdp'
import { uniformPolicy } from '../core/policy'
import { policyEvaluationDirect } from '../core/solvers'
import { mdpToEnv } from '../core/env'
import {
  FEATURE_LABEL,
  bairdCounterexample,
  featureDim,
  leastSquaresFit,
  semiGradientTD,
  valuesFromW,
  type FeatureKind,
} from '../core/fa'
import { Act, Beat } from '../narrative/Act'
import { ChapterGlance, ChapterHero, type Glance } from '../narrative/ChapterShell'
import { PredictChoice } from '../narrative/Predict'
import { GridWorld } from '../viz/GridWorld'
import { LineChart } from '../viz/LineChart'
import { Callout, Code, Details, M, MB, Panel, Seg, Slider, Stat, Toggle } from '../ui/prims'
import { fmt, useColors } from '../theme'

const GLANCE: Glance = {
  formula: String.raw`\hat v(s, w) \approx v_\pi(s), \qquad
  w \leftarrow w + \alpha \Big[ r + \gamma \hat v(s', w) - \hat v(s, w) \Big] \nabla_w \hat v(s, w)`,
  formulaNote:
    '把那张表换成一个带参数的函数。更新式几乎没变，只是「改一个格子」变成了「改一组参数」—— 于是一次更新会同时影响所有相似的状态。这既是泛化能力的来源，也是不稳定的来源。',
  takeaways: [
    <>
      表格法是函数近似的<strong>特例</strong>：取 one-hot 特征时，
      半梯度 TD 逐字退化成第 7 章的 TD。
    </>,
    <>
      「半梯度」的半：TD 目标 <M>{"r+\\gamma\\hat v(s',w)"}</M> 里也含 <M>{'w'}</M>，
      但求导时<strong>假装它是常数</strong>。这一步不严谨，却是可行性的来源。
    </>,
    <>
      近似能力有<strong>硬上限</strong>：真实的 <M>{'v_\\pi'}</M> 若不在函数空间里，
      再多数据也只能逼近投影。选特征 = 选这个上限。
    </>,
    <>
      致命三位一体：<strong>函数近似 + 自举 + 离策略</strong>，
      三者同时出现时算法可以指数发散 —— 哪怕真值就在近似空间内。
    </>,
  ],
  traps: [
    <>
      以为「用神经网络就叫深度强化学习」。真正的分水岭是
      <strong>从「更新一个格子」变成「更新一组共享参数」</strong>，
      泛化和干扰是同一枚硬币的两面。
    </>,
    <>
      把发散归咎于学习率太大。Baird 反例里<strong>任意小的学习率都会发散</strong>，
      问题出在更新方向本身。
    </>,
    <>
      忘了目标函数其实是<strong>加权</strong>的平方误差，
      权重是状态的访问分布 —— 离策略之所以危险，正是因为它把这个权重换掉了。
    </>,
  ],
}

const FEATURES: FeatureKind[] = ['poly1', 'poly2', 'poly3', 'fourier', 'tabular']

export function Chapter8() {
  const C = useColors()
  const [gamma] = useState(0.9)
  const [kind, setKind] = useState<FeatureKind>('poly2')
  const [alpha, setAlpha] = useState(0.02)
  const [bootstrap, setBootstrap] = useState(true)
  const [offPolicy, setOffPolicy] = useState(true)
  const [approx, setApprox] = useState(true)
  const [triadAlpha, setTriadAlpha] = useState(0.01)

  const mdp = useMemo(() => buildGridMDP(classicGrid()), [])
  const pi = useMemo(() => uniformPolicy(mdp.nS), [mdp.nS])
  const vTrue = useMemo(() => policyEvaluationDirect(mdp, pi, gamma), [mdp, pi, gamma])
  const env = useMemo(() => mdpToEnv(mdp, { horizon: 40 }), [mdp])

  /* 最小二乘拟合：近似能力的上限 */
  const fit = useMemo(() => leastSquaresFit(mdp, kind, vTrue), [mdp, kind, vTrue])
  const vFit = useMemo(() => valuesFromW(mdp, kind, fit), [mdp, kind, fit])
  const rmse = (a: number[]) =>
    Math.sqrt(a.reduce((acc, x, i) => acc + (x - vTrue[i]) ** 2, 0) / a.length)

  const ceiling = useMemo(
    () =>
      FEATURES.map((k) => {
        const w = leastSquaresFit(mdp, k, vTrue)
        return rmse(valuesFromW(mdp, k, w))
      }),
    [mdp, vTrue],
  )

  /* 半梯度 TD 学出来的 */
  const learned = useMemo(
    () => semiGradientTD(env, mdp, pi, kind, { gamma, alpha, episodes: 500, seed: 8 }, vTrue),
    [env, mdp, pi, kind, gamma, alpha, vTrue],
  )
  const vLearned = useMemo(() => valuesFromW(mdp, kind, learned.w), [mdp, kind, learned])

  /* 致命三位一体 */
  const triad = useMemo(
    () =>
      bairdCounterexample({
        bootstrap,
        offPolicy,
        approx,
        alpha: triadAlpha,
        gamma: 0.99,
        steps: 600,
      }),
    [bootstrap, offPolicy, approx, triadAlpha],
  )
  const diverged = triad.norm[triad.norm.length - 1] > 1e4
  const triadCount = [bootstrap, offPolicy, approx].filter(Boolean).length

  return (
    <div>
      <ChapterHero
        n={8}
        hook="状态一多，那张 Q 表格就装不下了。"
        lead={
          <>
            <p>
              前七章的所有算法都依赖一张表。表的大小是 <M>{'|\\mathcal S|'}</M> 或{' '}
              <M>{'|\\mathcal S|\\times|\\mathcal A|'}</M>。
              一旦状态空间大到装不下 —— 或者干脆是连续的 —— 这条路就断了。
            </p>
            <p>
              替代方案听起来平淡无奇：<strong>用一个带参数的函数去拟合价值</strong>。
              但这一步的后果远超「省内存」：一次更新不再只改一个格子，
              而是<em>同时改变所有相似状态的估计</em>。
              泛化能力和不稳定性，从此成为同一件事的两面。
            </p>
          </>
        }
        gains={[
          '把价值估计从「查表」变成「拟合一条曲线」',
          '理解「半梯度」里那个半字省掉了什么，以及为什么必须省',
          '看清近似能力的硬上限：选特征就是在选天花板',
          '亲手打开/关闭三个开关，让 Baird 反例发散给你看',
          '知道 DQN 的两个补丁分别在补哪个洞',
        ]}
      />

      <ChapterGlance g={GLANCE} />

      {/* ───────────────────── 第 1 幕 ───────────────────── */}
      <Act
        id="a1"
        no="第 1 幕"
        title="从查表到拟合"
        goal="用参数化函数代替表格，以及由此带来的天花板。"
        minutes={10}
        points={[
          <>
            目标函数 <M>{'J(w) = \\mathbb{E}\\left[(v_\\pi(S) - \\hat v(S,w))^2\\right]'}</M>，
            期望按状态的<strong>访问分布</strong>加权。
          </>,
          <>
            线性近似 <M>{'\\hat v(s,w) = \\phi(s)^\\top w'}</M> 里，
            <M>{'\\phi'}</M> 决定了函数空间；<strong>真值不在这个空间里，就永远拟合不上</strong>。
          </>,
          <>
            表格法 = 取 one-hot 特征的线性近似。它的天花板是 0，
            代价是参数量等于状态数、而且完全不泛化。
          </>,
        ]}
        stage={() => (
          <div className="space-y-4">
            <Panel
              title="真值 vs 最优拟合"
              right={
                <span className="font-mono text-[11.5px]" style={{ color: C.accent }}>
                  RMSE = {fmt(rmse(vFit), 4)}
                </span>
              }
            >
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <div className="mb-1.5 text-center text-[11px] text-faint">真值 v_π</div>
                  <GridWorld mdp={mdp} v={vTrue} showPolicy={false} cell={44} quiet />
                </div>
                <div>
                  <div className="mb-1.5 text-center text-[11px] text-faint">
                    {FEATURE_LABEL[kind]}
                  </div>
                  <GridWorld mdp={mdp} v={vFit} showPolicy={false} cell={44} quiet />
                </div>
              </div>
              <div className="mt-3">
                <Seg
                  value={kind}
                  onChange={setKind}
                  size="sm"
                  options={FEATURES.map((k) => ({ value: k, label: FEATURE_LABEL[k].split('（')[0] }))}
                />
              </div>
              <div className="mt-3 grid grid-cols-2 gap-2.5">
                <Stat label="参数个数" value={featureDim(mdp, kind)} color={C.policy} />
                <Stat label="状态个数" value={mdp.nS} color={C.state} />
              </div>
            </Panel>

            <Panel title="近似能力的天花板（最小二乘 RMSE）">
              <LineChart
                height={172}
                xLabel="一次 → 二次 → 三次 → 傅里叶 → 表格"
                marker={FEATURES.indexOf(kind)}
                series={[{ name: '可达到的最小误差', color: C.danger, data: ceiling }]}
              />
              <p className="mt-2 text-[11.5px] leading-relaxed text-faint">
                这条线是<strong>用无穷多数据也降不下去</strong>的误差。
                它只取决于你选了什么特征。
              </p>
            </Panel>
          </div>
        )}
      >
        <Beat id="b1" keep>
          <p>
            把表格 <M>{'v(s)'}</M> 换成一个函数 <M>{'\\hat v(s, w)'}</M>，
            其中 <M>{'w'}</M> 是一组参数，而且<strong>参数个数远少于状态个数</strong>。
            最简单的形式是线性的：
          </p>
          <MB>{'\\hat v(s, w) = \\phi(s)^\\top w = \\sum_{i} \\phi_i(s)\\, w_i'}</MB>
          <p>
            <M>{'\\phi(s)'}</M> 叫特征向量，是人（或者神经网络）设计的。
            比如在网格里可以取 <M>{'\\phi(s) = [1,\\, x,\\, y,\\, x^2,\\, y^2,\\, xy]^\\top'}</M> ——
            六个参数，描述整个 25 格的价值曲面。
          </p>
        </Beat>

        <Beat id="b2" keep>
          <p>拟合的目标很自然，就是最小化平方误差：</p>
          <MB>{'J(w) = \\mathbb{E}_{S\\sim d}\\left[\\left(v_\\pi(S) - \\hat v(S,w)\\right)^2\\right]'}</MB>
          <p>
            注意下标那个 <M>{'d'}</M> —— 期望是按<strong>状态的访问分布</strong>加权的。
            经常去的地方拟合得准一点，很少去的地方糊一点，是划算的。
            这个不起眼的权重，会在第 4 幕成为整章最大的麻烦。
          </p>
        </Beat>

        <Beat id="b3" keep>
          <PredictChoice
            id="ch8-ceiling"
            question={
              <>
                用一次多项式（3 个参数）去拟合这个 5×5 世界的 <M>{'v_\\pi'}</M>，
                如果给它<strong>无穷多</strong>的数据，误差能降到 0 吗？
              </>
            }
            options={[
              { id: 'a', label: '能 —— 数据足够多总能拟合上' },
              { id: 'b', label: '不能 —— 有一个与数据量无关的下界' },
              { id: 'c', label: '取决于学习率调得好不好' },
            ]}
            answer="b"
            explain={
              <>
                <p>
                  一次多项式张成的函数空间只有 3 维，而真实的 <M>{'v_\\pi'}</M>
                  是 25 维空间里的一个点。<strong>它几乎必然不在那个 3 维平面上。</strong>
                </p>
                <p>
                  能做到的最好，是把它<em>正交投影</em>到那个平面上，
                  而投影的残差就是右边那条曲线的第一个点：{fmt(ceiling[0], 3)}。
                  这个数与数据量、学习率、算法都无关，只与特征有关。
                </p>
                <p>
                  所以「选特征」不是调参，它是在<strong>决定这个算法能力的上限</strong>。
                  深度学习真正改变的事情，是让这个空间大到几乎不再是瓶颈。
                </p>
              </>
            }
          />
        </Beat>
      </Act>

      {/* ───────────────────── 第 2 幕 ───────────────────── */}
      <Act
        id="a2"
        no="第 2 幕"
        title="半梯度：省掉的那一半"
        goal="TD 目标里也含 w，但求导时假装它是常数 —— 为什么必须这么干。"
        minutes={9}
        points={[
          <>
            真梯度需要对 <M>{"\\gamma\\hat v(s',w)"}</M> 也求导；
            半梯度<strong>故意不求</strong>，把 TD 目标当成固定标签。
          </>,
          <>
            这么做的理由很实际：真梯度对应的目标函数（贝尔曼残差）
            需要<strong>两个独立样本</strong>才能无偏估计，采样时拿不到。
          </>,
          <>
            线性 + on-policy 时，半梯度 TD 收敛到<strong>TD 不动点</strong>，
            它比最小二乘的最优解略差，但差得有界。
          </>,
        ]}
        stage={() => (
          <div className="space-y-4">
            <Panel title="学出来的 vs 最优拟合 vs 真值">
              <div className="grid grid-cols-3 gap-2">
                <div>
                  <div className="mb-1.5 text-center text-[10.5px] text-faint">真值</div>
                  <GridWorld mdp={mdp} v={vTrue} showPolicy={false} showValues={false} cell={34} quiet />
                </div>
                <div>
                  <div className="mb-1.5 text-center text-[10.5px] text-faint">最小二乘</div>
                  <GridWorld mdp={mdp} v={vFit} showPolicy={false} showValues={false} cell={34} quiet />
                </div>
                <div>
                  <div className="mb-1.5 text-center text-[10.5px] text-faint">半梯度 TD</div>
                  <GridWorld mdp={mdp} v={vLearned} showPolicy={false} showValues={false} cell={34} quiet />
                </div>
              </div>
              <div className="mt-3 grid grid-cols-2 gap-2.5">
                <Stat label="最小二乘 RMSE" value={fmt(rmse(vFit), 3)} color={C.value} />
                <Stat
                  label="半梯度 TD RMSE"
                  value={fmt(rmse(vLearned), 3)}
                  color={rmse(vLearned) > rmse(vFit) * 1.5 ? C.reward : C.accent}
                />
              </div>
            </Panel>

            <Panel title="学习曲线">
              <LineChart
                height={180}
                xLabel="回合"
                series={[
                  { name: '半梯度 TD 的 RMSE', color: C.accent, data: learned.err },
                  {
                    name: '最小二乘天花板',
                    color: C.value,
                    data: new Array(learned.err.length).fill(rmse(vFit)),
                    dashed: true,
                  },
                ]}
              />
              <div className="mt-3">
                <Slider
                  label={<>步长 <M>{'\\alpha'}</M></>}
                  value={alpha}
                  min={0.002}
                  max={0.1}
                  step={0.002}
                  onChange={setAlpha}
                  accent={C.gamma}
                  hint="太大就会在天花板附近剧烈抖动，甚至发散"
                />
              </div>
            </Panel>
          </div>
        )}
      >
        <Beat id="c1" keep>
          <p>
            用 SGD 最小化 <M>{'J(w)'}</M>，梯度是
          </p>
          <MB>{'-\\tfrac12\\nabla_w J = \\mathbb{E}\\left[\\left(v_\\pi(S) - \\hat v(S,w)\\right)\\nabla_w \\hat v(S,w)\\right]'}</MB>
          <p>
            但 <M>{'v_\\pi(S)'}</M> 我们不知道 —— 它正是要求的东西。
            于是照第 7 章的老办法，用 TD 目标顶替它：
          </p>
          <MB>{"w \\leftarrow w + \\alpha\\left[ r + \\gamma\\hat v(s',w) - \\hat v(s,w) \\right]\\nabla_w \\hat v(s,w)"}</MB>
          <p>
            对线性近似而言 <M>{'\\nabla_w\\hat v = \\phi(s)'}</M>，
            所以整个算法就是「TD 误差 × 特征向量」。
          </p>
        </Beat>

        <Beat id="c2" keep>
          <Callout tone="trap" title="它其实不是任何函数的梯度">
            <p>
              上式里 <M>{"\\hat v(s',w)"}</M> 明明也依赖 <M>{'w'}</M>，
              可我们求导时把它当成了常数。所以这个「梯度」并不是 <M>{'J'}</M> 的真梯度，
              甚至<strong>不是任何目标函数的梯度</strong>。这就是「半梯度」这个名字的来历。
            </p>
            <p>
              为什么不用真梯度？因为真梯度对应的目标是贝尔曼残差{' '}
              <M>{"\\mathbb{E}[(r+\\gamma\\hat v(S')-\\hat v(S))^2]"}</M>，
              而它的无偏梯度估计需要<strong>从同一个状态独立采样两次</strong>
              （所谓 double sampling 问题）—— 真实环境里做不到。
            </p>
            <p>
              所以半梯度不是偷懒，是<strong>唯一可行的选择</strong>。
              代价我们在第 4 幕会付。
            </p>
          </Callout>
        </Beat>

        <Beat id="c3">
          <Details summary="展开：TD 不动点与最优拟合差多远">
            <p>
              on-policy 线性情形下，半梯度 TD 收敛到满足下式的 <M>{'w_{TD}'}</M>：
            </p>
            <MB>{'\\Phi^\\top D(\\Phi w_{TD} - r_\\pi - \\gamma P_\\pi \\Phi w_{TD}) = 0'}</MB>
            <p>它与最优投影 <M>{'w_{LS}'}</M> 的误差满足经典的界：</p>
            <MB>{'\\|\\hat v_{w_{TD}} - v_\\pi\\|_D \\le \\frac{1}{1-\\gamma}\\,\\|\\Pi v_\\pi - v_\\pi\\|_D'}</MB>
            <p>
              也就是说，TD 学到的东西<strong>最差不会比最优拟合差过 1/(1−γ) 倍</strong>。
              γ 越接近 1，这个保证越松 —— 又一次，γ 在幕后决定一切。
            </p>
          </Details>
        </Beat>
      </Act>

      {/* ───────────────────── 第 3 幕 ───────────────────── */}
      <Act
        id="a3"
        no="第 3 幕"
        title="致命三位一体"
        goal="函数近似 + 自举 + 离策略，三者齐聚时算法会发散 —— 亲手把它们一个个打开。"
        minutes={11}
        points={[
          <>
            三个要素单独出现或两两组合都<strong>安全</strong>；三者齐聚才出事。
          </>,
          <>
            Baird 反例里，<strong>真值就在近似空间内</strong>（存在 w 使误差为 0），
            算法却仍然指数发散。这说明问题不在近似能力。
          </>,
          <>
            根因：离策略把状态分布 <M>{'d'}</M> 换掉了，
            于是「TD 更新是收缩映射」这个前提不再成立。
          </>,
          <>
            补救方向：去掉自举（MC）、去掉离策略（on-policy）、
            或者改用真正的梯度法（GTD 系列）。DQN 走的是第四条路：<strong>工程加固</strong>。
          </>,
        ]}
        stage={() => (
          <div className="space-y-4">
            <Panel
              title="Baird 反例"
              right={
                <span
                  className="rounded-md px-2 py-0.5 font-mono text-[11px]"
                  style={{
                    color: diverged ? C.danger : C.value,
                    background: `color-mix(in srgb, ${diverged ? 'var(--danger)' : 'var(--value)'} 12%, transparent)`,
                  }}
                >
                  {diverged ? '发散' : '稳定'}
                </span>
              }
            >
              <LineChart
                height={200}
                logY
                xLabel="更新步"
                series={[{ name: '‖w‖₂', color: diverged ? C.danger : C.value, data: triad.norm }]}
              />
              <div className="mt-4 space-y-2.5">
                <Toggle label="函数近似（关掉 = 表格）" checked={approx} onChange={setApprox} />
                <Toggle label="自举（关掉 = 用真回报，蒙特卡洛式）" checked={bootstrap} onChange={setBootstrap} />
                <Toggle label="离策略（关掉 = on-policy）" checked={offPolicy} onChange={setOffPolicy} />
              </div>
              <div className="mt-3">
                <Slider
                  label={<>步长 <M>{'\\alpha'}</M></>}
                  value={triadAlpha}
                  min={0.001}
                  max={0.05}
                  step={0.001}
                  onChange={setTriadAlpha}
                  accent={C.gamma}
                  hint="调到最小试试 —— 三个开关全开时，再小的步长也救不了"
                />
              </div>
              <div className="mt-3">
                <Stat
                  label="已打开的危险因素"
                  value={`${triadCount} / 3`}
                  color={triadCount === 3 ? C.danger : triadCount === 2 ? C.reward : C.value}
                />
              </div>
            </Panel>
          </div>
        )}
      >
        <Beat id="d1" keep>
          <p>
            到此为止一切都还算顺利。但把函数近似和 Q-learning 拼在一起时，
            一个隐患就成型了。Sutton 给它起了个名字：<strong>致命三位一体</strong>。
          </p>
          <ul>
            <li><strong>函数近似</strong> —— 参数共享，改一个状态会牵动别的</li>
            <li><strong>自举</strong> —— 用估计更新估计（TD 而非 MC）</li>
            <li><strong>离策略</strong> —— 采数据的策略 ≠ 要评估的策略</li>
          </ul>
          <p>
            单独一个都没事，两两组合也基本没事。<em>三个凑齐，算法可以直接炸。</em>
          </p>
        </Beat>

        <Beat id="d2" keep>
          <PredictChoice
            id="ch8-triad"
            question={
              <>
                Baird 反例里，<strong>存在</strong>一组参数让近似误差恰好为 0。
                那把学习率调到极小，算法总该慢慢收敛过去吧？
              </>
            }
            options={[
              { id: 'a', label: '会 —— 学习率足够小，梯度下降总能收敛' },
              { id: 'b', label: '不会 —— 参数会指数发散，学习率只影响发散速度' },
              { id: 'c', label: '会震荡但不发散' },
            ]}
            answer="b"
            explain={
              <>
                <p>
                  右边把三个开关全打开，然后把 <M>{'\\alpha'}</M> 拖到最小 ——
                  曲线仍然一路向上（注意是对数坐标）。
                </p>
                <p>
                  原因在于半梯度不是任何目标函数的梯度。它对应的迭代矩阵
                  <strong>特征值的实部可以为正</strong>，
                  这时无论步长多小，迭代都是不稳定的。步长只改变发散的<em>速度</em>，
                  不改变<em>方向</em>。
                </p>
                <p>
                  然后关掉任意一个开关试试 —— 曲线立刻变平。
                  这就是「三位一体」这个名字的含义：
                  <strong>缺一不可，也缺一即安全。</strong>
                </p>
              </>
            }
          />
        </Beat>

        <Beat id="d3">
          <Details summary="展开：为什么离策略会破坏收敛">
            <p>
              on-policy 时，状态按 <M>{'d_\\pi'}</M> 分布被访问，
              可以证明 <M>{'\\Pi T_\\pi'}</M>（投影 + 贝尔曼算子）在 <M>{'\\|\\cdot\\|_{d_\\pi}'}</M>
              范数下是压缩映射，于是不动点存在且迭代收敛。
            </p>
            <p>
              离策略时，数据按行为策略 <M>{'d_b'}</M> 分布，
              但我们要评估的是 <M>{'\\pi_t'}</M>。投影用的是 <M>{'d_b'}</M> 的度量，
              贝尔曼算子用的是 <M>{'\\pi_t'}</M> 的动力学 ——
              <strong>两个算子在不同的度量下工作，复合之后不再是压缩映射</strong>。
            </p>
            <p>
              这就是全部原因。表格法之所以没事，是因为投影算子是恒等映射，
              度量不匹配的问题不存在。
            </p>
          </Details>
        </Beat>
      </Act>

      {/* ───────────────────── 第 4 幕 ───────────────────── */}
      <Act
        id="a4"
        no="第 4 幕"
        title="DQN：不解决问题，绕开问题"
        goal="经验回放与目标网络，两个补丁各补哪个洞。"
        minutes={8}
        points={[
          <>
            <strong>目标网络</strong>：把 TD 目标里的 <M>{'w'}</M> 冻结一段时间，
            让「半梯度」名副其实地变成真梯度（对一个固定标签做回归）。
          </>,
          <>
            <strong>经验回放</strong>：打散样本相关性，同时把每份数据用很多次 ——
            这依赖 Q-learning 的 off-policy 性质。
          </>,
          <>
            两个补丁都<strong>没有解决</strong>三位一体的理论问题，
            只是让它在实践中不容易爆。这是工程胜利，不是理论胜利。
          </>,
        ]}
        stage={() => (
          <div className="space-y-4">
            <Panel title="从表格到 DQN，一路上换掉了什么">
              <div className="space-y-2.5 text-[12.5px] leading-relaxed">
                {[
                  ['第 7 章 Q-learning', 'q 是一张表；改一格只影响一格', C.state],
                  ['+ 线性近似', 'q = φ(s,a)ᵀw；改 w 影响所有相似状态', C.value],
                  ['+ 神经网络', 'φ 也交给网络自己学，不再手工设计', C.policy],
                  ['+ 目标网络', 'TD 目标用旧参数 w⁻，每 C 步同步一次', C.reward],
                  ['+ 经验回放', '从缓冲区随机采样，打散时间相关性', C.qvalue],
                  ['= DQN', '2015 年，Atari 上超过人类', C.accent],
                ].map(([a, b, col]) => (
                  <div key={a as string} className="flex gap-3 rounded-lg border border-line bg-surface2 px-3 py-2">
                    <span className="w-[104px] shrink-0 font-mono text-[11.5px]" style={{ color: col as string }}>
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
          <Code
            code={`class DQN:
    def __init__(self):
        self.q  = Net()          # 在线网络
        self.q_ = copy(self.q)   # 目标网络：参数被冻结
        self.buf = ReplayBuffer(capacity=100_000)

    def learn_step(self, C=1000):
        s, a, r, sp, done = self.buf.sample(batch=32)     # ① 经验回放

        with no_grad():
            # ② 目标网络：这里的参数是旧的，所以它真的是一个「常数标签」
            y = r + gamma * (1 - done) * self.q_(sp).max(dim=1)

        loss = mse(self.q(s).gather(a), y)                 # 变成了普通的回归
        loss.backward(); opt.step()

        if self.steps % C == 0:
            self.q_.load_state_dict(self.q.state_dict())   # 每 C 步同步一次`}
            lang="python"
          />
        </Beat>

        <Beat id="e2" keep>
          <Callout tone="insight" title="目标网络的真正作用">
            <p>
              把 TD 目标里的参数冻住之后，<M>{'y'}</M> 就成了一个
              <strong>与当前 w 无关的常数</strong>。于是那个「半梯度」
              突然变成了如假包换的真梯度 —— 只不过是对一个
              <em>暂时固定的回归问题</em> 求梯度。
            </p>
            <p>
              这就是它稳定训练的机制：不是消除了三位一体，
              而是把「追着自己跑」变成了「每 C 步追一个静止的靶子」。
            </p>
          </Callout>
        </Beat>

        <Beat id="e3" keep>
          <Callout tone="trap" title="这一章留下的问题">
            <p>
              到这里，价值方法这一条线基本走完了：从表格到函数近似，
              从 Q-learning 到 DQN。但请注意，
              <strong>我们始终在绕道</strong> ——
              先估价值，再从价值里读出策略（<M>{'\\arg\\max_a q'}</M>）。
            </p>
            <p>
              这个绕道有两个硬伤。第一，<M>{'\\arg\\max'}</M> 在
              <strong>连续动作空间</strong>里无法计算。
              第二，从 q 读出来的策略必然是确定性的（或 ε-贪心那种粗糙的随机），
              而有些任务的最优策略<strong>本质上是随机的</strong>（比如石头剪刀布）。
            </p>
            <p>
              于是下一章要问一个大胆的问题：
              <em>价值只是手段。能不能把策略本身当成参数化的对象，直接对它求梯度？</em>
            </p>
          </Callout>
        </Beat>
      </Act>
    </div>
  )
}
