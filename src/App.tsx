import { Suspense, lazy, useEffect, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { CHAPTERS, StorylineMap, StorylineRail } from './narrative/Storyline'
import { FrameworkMap } from './narrative/FrameworkMap'
import { useProgress } from './narrative/progress'
import { useApplyTheme, useSettings } from './theme'
import { Seg } from './ui/prims'
import { Home } from './pages/Home'

/** 章节按需加载：十章全量打进首屏会让首次进入明显变慢 */
const CHAPTER_VIEWS: Record<string, React.LazyExoticComponent<() => React.ReactElement>> = {
  ch1: lazy(() => import('./chapters/Chapter1').then((m) => ({ default: m.Chapter1 }))),
  ch2: lazy(() => import('./chapters/Chapter2').then((m) => ({ default: m.Chapter2 }))),
  ch3: lazy(() => import('./chapters/Chapter3').then((m) => ({ default: m.Chapter3 }))),
  ch4: lazy(() => import('./chapters/Chapter4').then((m) => ({ default: m.Chapter4 }))),
  ch5: lazy(() => import('./chapters/Chapter5').then((m) => ({ default: m.Chapter5 }))),
  ch6: lazy(() => import('./chapters/Chapter6').then((m) => ({ default: m.Chapter6 }))),
  ch7: lazy(() => import('./chapters/Chapter7').then((m) => ({ default: m.Chapter7 }))),
  ch8: lazy(() => import('./chapters/Chapter8').then((m) => ({ default: m.Chapter8 }))),
  ch9: lazy(() => import('./chapters/Chapter9').then((m) => ({ default: m.Chapter9 }))),
  ch10: lazy(() => import('./chapters/Chapter10').then((m) => ({ default: m.Chapter10 }))),
}

function useHashRoute() {
  const [route, setRoute] = useState(() => location.hash.replace(/^#\/?/, '') || 'home')
  useEffect(() => {
    const on = () => {
      setRoute(location.hash.replace(/^#\/?/, '') || 'home')
      window.scrollTo({ top: 0, behavior: 'instant' as ScrollBehavior })
    }
    addEventListener('hashchange', on)
    return () => removeEventListener('hashchange', on)
  }, [])
  return route
}

function ReadingProgress() {
  const [p, setP] = useState(0)
  useEffect(() => {
    const on = () => {
      const h = document.documentElement
      const max = h.scrollHeight - h.clientHeight
      setP(max > 0 ? h.scrollTop / max : 0)
    }
    on()
    addEventListener('scroll', on, { passive: true })
    addEventListener('resize', on)
    return () => {
      removeEventListener('scroll', on)
      removeEventListener('resize', on)
    }
  }, [])
  return (
    <div className="fixed inset-x-0 top-0 z-50 h-[2px] bg-transparent">
      <div
        className="h-full bg-gradient-to-r from-brand to-alt transition-[width] duration-150"
        style={{ width: `${p * 100}%` }}
      />
    </div>
  )
}

function TopBar({ current, onOpenMap }: { current: number; onOpenMap: () => void }) {
  const answers = useProgress((s) => s.answers)
  const done = Object.keys(answers).length
  const right = Object.values(answers).filter(Boolean).length
  const { theme, setTheme, skim, setSkim } = useSettings()

  return (
    <header className="sticky top-0 z-40 border-b border-line bg-bg/85 backdrop-blur-xl">
      <div className="mx-auto flex h-14 max-w-6xl items-center gap-5 px-6">
        <a href="#/" className="flex items-center gap-2.5">
          <span className="flex h-6 w-6 items-center justify-center rounded-md bg-gradient-to-br from-brand to-alt font-mono text-[11px] font-bold text-white">
            RL
          </span>
          <span className="font-serif text-[15px] font-bold text-ink">ReinLab</span>
        </a>

        <div className="ml-2 hidden items-center gap-3 lg:flex">
          <StorylineRail current={current} />
        </div>

        <div className="ml-auto flex items-center gap-2.5">
          {done > 0 && (
            <span className="hidden font-mono text-[11.5px] text-faint xl:inline">
              下注 {done} 次 · 命中 {right}
            </span>
          )}
          <Seg
            size="sm"
            value={skim ? 'skim' : 'full'}
            onChange={(v) => setSkim(v === 'skim')}
            options={[
              { value: 'full', label: '精读', hint: '完整叙事，推荐第一遍' },
              { value: 'skim', label: '速读', hint: '只留要点、公式、交互和练习' },
            ]}
          />
          <button
            onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
            title={theme === 'dark' ? '切到浅色' : '切到深色'}
            className="flex h-8 w-8 items-center justify-center rounded-lg border border-line text-[13px] text-dim transition-colors hover:border-brand/40 hover:text-brand"
          >
            {theme === 'dark' ? '☀' : '☾'}
          </button>
          <button
            onClick={onOpenMap}
            className="rounded-lg border border-line px-3 py-1.5 text-[12.5px] text-dim transition-colors hover:border-brand/40 hover:text-brand"
          >
            地图
          </button>
        </div>
      </div>
    </header>
  )
}

function MapDrawer({
  open,
  onClose,
  current,
}: {
  open: boolean
  onClose: () => void
  current: number
}) {
  const [tab, setTab] = useState<'frame' | 'line'>('frame')
  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 z-50 bg-black/35 backdrop-blur-sm"
          />
          <motion.aside
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: 'spring', damping: 32, stiffness: 300 }}
            className="fixed top-0 right-0 z-50 h-full w-full max-w-[880px] overflow-y-auto border-l border-line bg-bg px-6 py-6"
          >
            <div className="mb-5 flex items-center justify-between gap-4">
              <Seg
                value={tab}
                onChange={setTab}
                options={[
                  { value: 'frame', label: '全景图 · 算法谱系' },
                  { value: 'line', label: '主线 · 章节链条' },
                ]}
              />
              <button
                onClick={onClose}
                className="rounded-lg border border-line px-2.5 py-1 text-[12px] text-dim hover:text-ink"
              >
                关闭
              </button>
            </div>
            {tab === 'frame' ? (
              <FrameworkMap
                compact
                current={current > 0 ? `ch${current}` : undefined}
                onGo={(id) => {
                  location.hash = `#/${id}`
                  onClose()
                }}
              />
            ) : (
              <div className="max-w-[520px]">
                <p className="mb-4 text-[12.5px] text-faint">
                  每一章，都是被上一章留下的缺陷逼出来的。
                </p>
                <StorylineMap
                  current={current}
                  onGo={(c) => {
                    location.hash = `#/${c.id}`
                    onClose()
                  }}
                />
              </div>
            )}
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  )
}

function ChapterFooter({ current }: { current: number }) {
  const cur = CHAPTERS.find((c) => c.n === current)
  const next = CHAPTERS.find((c) => c.n === current + 1)
  if (!cur) return null
  return (
    <footer className="mx-auto max-w-3xl px-6 py-20">
      <div className="card rounded-3xl p-8">
        <div className="text-[11px] tracking-[.2em] text-warn uppercase">这一章留下的问题</div>
        <p className="mt-3 font-serif text-[21px] leading-relaxed text-ink italic">
          「{cur.cliffhanger}」
        </p>
        {next && (
          <div className="mt-6 flex items-center gap-3">
            {next.ready ? (
              <a
                href={`#/${next.id}`}
                className="rounded-xl bg-brand px-4 py-2 text-[13.5px] font-medium text-white transition-opacity hover:opacity-90"
              >
                进入第 {next.n} 章 · {next.title}
              </a>
            ) : (
              <span className="rounded-xl border border-line px-4 py-2 text-[13.5px] text-faint">
                第 {next.n} 章 · {next.title} · 建设中
              </span>
            )}
            <a href="#/" className="text-[13px] text-faint hover:text-dim">
              回到目录
            </a>
          </div>
        )}
      </div>
    </footer>
  )
}

function Loading() {
  return (
    <div className="mx-auto max-w-3xl px-6 py-40 text-center">
      <div className="mx-auto h-6 w-6 animate-spin rounded-full border-2 border-line border-t-brand" />
      <p className="mt-4 text-[13px] text-faint">正在装载这一章的世界…</p>
    </div>
  )
}

export function App() {
  useApplyTheme()
  const route = useHashRoute()
  const [mapOpen, setMapOpen] = useState(false)

  const chapterNo = route.startsWith('ch') ? Number(route.slice(2)) : 0
  const View = CHAPTER_VIEWS[route]

  return (
    <div className="min-h-screen">
      <ReadingProgress />
      <TopBar current={chapterNo} onOpenMap={() => setMapOpen(true)} />
      <MapDrawer open={mapOpen} onClose={() => setMapOpen(false)} current={chapterNo} />

      <main>
        {route === 'home' && <Home />}
        {View && (
          <Suspense fallback={<Loading />}>
            <View />
          </Suspense>
        )}
        {route !== 'home' && !View && (
          <div className="mx-auto max-w-3xl px-6 py-32 text-center">
            <p className="text-dim">没有这一页。</p>
            <a href="#/" className="mt-4 inline-block text-brand">
              回到目录
            </a>
          </div>
        )}
      </main>

      {chapterNo > 0 && <ChapterFooter current={chapterNo} />}
    </div>
  )
}
