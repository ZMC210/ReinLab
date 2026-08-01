export interface ChapterMeta {
  n: number
  id: string
  title: string
  /** 这一章解决了什么 */
  solves: string
  /** 这一章解决完之后，还欠着什么 —— 它就是下一章存在的理由 */
  cliffhanger: string
  ready: boolean
}

/**
 * 全书主线。
 *
 * 体系感的来源不是「章节目录」，而是这样一条链：
 * 每一章都是被上一章留下的缺陷逼出来的。学生任何时候都应该能看到
 * 「我在哪、我为什么在这、下一步为什么必须去那」。
 */
export const CHAPTERS: ChapterMeta[] = [
  {
    n: 1,
    id: 'ch1',
    title: '基本概念',
    solves: '把「一个智能体在环境里学着做事」翻译成数学：状态、动作、策略、回报、MDP。',
    cliffhanger: '目标写清楚了。可给我一个策略，我怎么算出它到底值多少分？',
    ready: true,
  },
  {
    n: 2,
    id: 'ch2',
    title: '贝尔曼公式',
    solves: '价值的递归定义：今天的价值 = 今天的奖励 + 折扣后的明天的价值。策略从此可以被打分。',
    cliffhanger: '能给策略打分了。可全世界这么多策略，怎么找到最好的那一个？',
    ready: true,
  },
  {
    n: 3,
    id: 'ch3',
    title: '贝尔曼最优公式',
    solves: '在贝尔曼公式里加一个 max，得到刻画「最优」的方程，并证明它的解存在且唯一。',
    cliffhanger: '方程有了，但多了 max 之后它是非线性的，没法一步解出来。怎么办？',
    ready: true,
  },
  {
    n: 4,
    id: 'ch4',
    title: '值迭代与策略迭代',
    solves: '用迭代算法把最优方程真的解出来，并看清两个算法其实是同一条谱系的两端。',
    cliffhanger: '算法要用到 p(s′|s,a) 和 r(s,a) —— 可现实里根本没人给我模型。',
    ready: true,
  },
  {
    n: 5,
    id: 'ch5',
    title: '蒙特卡洛方法',
    solves: '不知道期望，就去采样。用大量轨迹的平均值代替模型给出的期望。',
    cliffhanger: '必须等一整条轨迹结束才能更新，而且估计值抖得厉害。',
    ready: true,
  },
  {
    n: 6,
    id: 'ch6',
    title: '随机近似与随机梯度下降',
    solves: '「用一个个样本增量式地逼近期望」这件事本身的数学许可证：RM 算法与 SGD。',
    cliffhanger: '这张许可证如果用回贝尔曼公式，会长成什么算法？',
    ready: true,
  },
  {
    n: 7,
    id: 'ch7',
    title: '时序差分方法',
    solves: '把随机近似用在贝尔曼公式上，得到在线、单步、无模型的 TD / Sarsa / Q-learning。',
    cliffhanger: '状态一多，那张 Q 表格就装不下了。',
    ready: true,
  },
  {
    n: 8,
    id: 'ch8',
    title: '值函数近似',
    solves: '用一个带参数的函数代替表格，让方法能推广到没见过的状态。',
    cliffhanger: '价值只是手段。能不能绕开它，直接优化策略本身？',
    ready: true,
  },
  {
    n: 9,
    id: 'ch9',
    title: '策略梯度方法',
    solves: '把策略参数化，对目标函数直接求梯度，做梯度上升。',
    cliffhanger: '梯度公式里那个 q_π(s,a)，谁来估？',
    ready: true,
  },
  {
    n: 10,
    id: 'ch10',
    title: 'Actor-Critic 方法',
    solves: '演员负责动作，评论家负责打分 —— 前面九章在这里合流。',
    cliffhanger: '到这里，你已经拿到了现代强化学习的全部地基。',
    ready: true,
  },
]

export function StorylineMap({
  current,
  onGo,
}: {
  current?: number
  onGo?: (c: ChapterMeta) => void
}) {
  return (
    <div className="relative">
      <div className="absolute top-3 bottom-3 left-[19px] w-px bg-gradient-to-b from-brand/40 via-line2 to-transparent" />
      {CHAPTERS.map((c, i) => {
        const isCur = c.n === current
        return (
          <div key={c.id} className="relative pb-1">
            <button
              onClick={() => c.ready && onGo?.(c)}
              disabled={!c.ready}
              className={`group flex w-full items-start gap-4 rounded-2xl px-2 py-3 text-left transition-colors ${
                c.ready ? 'hover:bg-surface' : 'cursor-default opacity-45'
              }`}
            >
              <span
                className={`relative z-10 mt-0.5 flex h-10 w-10 shrink-0 items-center justify-center rounded-full border font-mono text-[13px] transition-all ${
                  isCur
                    ? 'border-brand/70 bg-brand/15 text-brand ring-2 ring-brand/25'
                    : c.ready
                      ? 'border-line bg-bg text-dim group-hover:border-brand/40'
                      : 'border-line bg-bg text-faint'
                }`}
              >
                {c.n}
              </span>
              <span className="min-w-0 flex-1">
                <span className="flex items-baseline gap-2.5">
                  <span
                    className={`font-serif text-[17px] font-bold ${
                      isCur ? 'text-brand' : 'text-ink'
                    }`}
                  >
                    {c.title}
                  </span>
                  {!c.ready && (
                    <span className="rounded border border-line px-1.5 py-px text-[10px] text-faint">
                      建设中
                    </span>
                  )}
                </span>
                <span className="mt-1 block text-[13.5px] leading-relaxed text-dim">
                  {c.solves}
                </span>
              </span>
            </button>

            {i < CHAPTERS.length - 1 && (
              <div className="relative z-10 mb-1 ml-[52px] flex items-start gap-2 pb-2">
                <span className="mt-[7px] text-[10px] text-warn">↓</span>
                <span className="text-[12.5px] leading-relaxed text-warn italic">
                  {c.cliffhanger}
                </span>
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

export function StorylineRail({ current }: { current: number }) {
  return (
    <div className="flex items-center gap-1">
      {CHAPTERS.map((c) => (
        <span
          key={c.id}
          title={`${c.n}. ${c.title}`}
          className={`h-1 rounded-full transition-all ${
            c.n === current ? 'w-6 bg-brand' : 'w-2.5 bg-line2'
          }`}
        />
      ))}
    </div>
  )
}
