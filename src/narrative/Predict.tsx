import { useState, type ReactNode } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { fmt } from '../theme'
import { useProgress } from './progress'

/**
 * 「先下注，再揭晓」。
 *
 * 依据是生成效应与 productive failure：先让学生做出一个可能错的承诺，
 * 再给出正解，长期记忆与迁移都显著优于被动阅读。
 * 所以这里刻意做成「不选就不给看答案」—— 摩擦是故意的。
 */

function Shell({
  question,
  children,
  revealed,
  correct,
  onReveal,
  canReveal,
  explain,
}: {
  question: ReactNode
  children: ReactNode
  revealed: boolean
  correct: boolean | null
  onReveal: () => void
  canReveal: boolean
  explain: ReactNode
}) {
  return (
    <div className="my-8 overflow-hidden rounded-2xl border border-warn/25 bg-gradient-to-b from-warn/[.06] to-transparent">
      <div className="flex items-center gap-2 border-b border-warn/15 px-5 py-2.5">
        <span className="relative flex h-1.5 w-1.5">
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-warn opacity-70" />
          <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-warn" />
        </span>
        <span className="text-[11px] font-medium tracking-[.18em] text-warn/90 uppercase">
          先下注 · 再揭晓
        </span>
      </div>

      <div className="px-5 py-4">
        <div className="text-[15.5px] leading-[1.8] font-medium text-ink">{question}</div>
        <div className="mt-4">{children}</div>

        {!revealed && (
          <button
            disabled={!canReveal}
            onClick={onReveal}
            className={`mt-4 rounded-xl px-4 py-2 text-[13px] font-medium transition-all ${
              canReveal
                ? 'bg-warn text-white hover:opacity-90'
                : 'cursor-not-allowed border border-line bg-surface text-faint'
            }`}
          >
            {canReveal ? '揭晓答案' : '先做出你的选择'}
          </button>
        )}

        <AnimatePresence>
          {revealed && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              transition={{ duration: 0.45, ease: [0.2, 0.7, 0.2, 1] }}
              className="overflow-hidden"
            >
              <div className="mt-4 border-t border-line pt-4">
                {correct !== null && (
                  <div
                    className={`mb-2.5 inline-flex items-center gap-2 rounded-lg px-2.5 py-1 text-[12px] font-medium ${
                      correct
                        ? 'bg-ok/12 text-ok'
                        : 'bg-bad/12 text-bad'
                    }`}
                  >
                    {correct ? '猜对了' : '猜错了 —— 这恰恰是最值钱的一刻'}
                  </div>
                )}
                <div className="text-[14.5px] leading-[1.9] text-dim">{explain}</div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}

export function PredictChoice({
  id,
  question,
  options,
  answer,
  explain,
}: {
  id: string
  question: ReactNode
  options: { id: string; label: ReactNode }[]
  answer: string
  explain: ReactNode
}) {
  const [picked, setPicked] = useState<string | null>(null)
  const [revealed, setRevealed] = useState(false)
  const record = useProgress((s) => s.record)

  return (
    <Shell
      question={question}
      revealed={revealed}
      correct={revealed ? picked === answer : null}
      canReveal={picked !== null}
      onReveal={() => {
        setRevealed(true)
        record(id, picked === answer)
      }}
      explain={explain}
    >
      <div className="flex flex-col gap-2">
        {options.map((o) => {
          const isPicked = picked === o.id
          const isAnswer = o.id === answer
          const tone = !revealed
            ? isPicked
              ? 'border-warn/55 bg-warn/10 text-ink'
              : 'border-line bg-surface text-dim hover:border-line'
            : isAnswer
              ? 'border-ok/50 bg-ok/10 text-ok'
              : isPicked
                ? 'border-bad/40 bg-bad/8 text-bad'
                : 'border-line bg-transparent text-faint'
          return (
            <button
              key={o.id}
              disabled={revealed}
              onClick={() => setPicked(o.id)}
              className={`rounded-xl border px-4 py-2.5 text-left text-[14px] leading-relaxed transition-all ${tone}`}
            >
              {o.label}
            </button>
          )
        })}
      </div>
    </Shell>
  )
}

export function PredictNumber({
  id,
  question,
  min,
  max,
  step = 0.1,
  truth,
  tolerance,
  unit = '',
  explain,
}: {
  id: string
  question: ReactNode
  min: number
  max: number
  step?: number
  truth: number
  tolerance: number
  unit?: string
  explain: ReactNode
}) {
  const [guess, setGuess] = useState<number | null>(null)
  const [revealed, setRevealed] = useState(false)
  const record = useProgress((s) => s.record)
  const ok = guess !== null && Math.abs(guess - truth) <= tolerance

  const pos = (x: number) => `${((x - min) / (max - min)) * 100}%`

  return (
    <Shell
      question={question}
      revealed={revealed}
      correct={revealed ? ok : null}
      canReveal={guess !== null}
      onReveal={() => {
        setRevealed(true)
        record(id, ok)
      }}
      explain={explain}
    >
      <div className="rounded-xl border border-line bg-surface2 px-4 pt-4 pb-3">
        <div className="relative h-10">
          <div className="absolute top-4 h-1 w-full rounded bg-surface" />
          {guess !== null && (
            <div
              className="absolute top-1.5 -translate-x-1/2 transition-all"
              style={{ left: pos(guess) }}
            >
              <div className="h-6 w-[3px] rounded bg-warn" />
              <div className="mt-0.5 -translate-x-1/2 font-mono text-[11px] whitespace-nowrap text-warn">
                你猜 {fmt(guess)}
                {unit}
              </div>
            </div>
          )}
          {revealed && (
            <motion.div
              initial={{ opacity: 0, scale: 0.7 }}
              animate={{ opacity: 1, scale: 1 }}
              className="absolute top-1.5 -translate-x-1/2"
              style={{ left: pos(Math.min(max, Math.max(min, truth))) }}
            >
              <div className="h-6 w-[3px] rounded bg-ok" />
              <div className="mt-0.5 -translate-x-1/2 font-mono text-[11px] whitespace-nowrap text-ok">
                真相 {fmt(truth)}
                {unit}
              </div>
            </motion.div>
          )}
        </div>
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={guess ?? (min + max) / 2}
          disabled={revealed}
          onChange={(e) => setGuess(Number(e.target.value))}
          className="ui-range mt-6 w-full"
          style={
            {
              '--pct': `${(((guess ?? (min + max) / 2) - min) / (max - min)) * 100}%`,
              '--sl': 'var(--reward)',
            } as React.CSSProperties
          }
        />
        <div className="mt-1 flex justify-between font-mono text-[10.5px] text-faint">
          <span>{fmt(min)}{unit}</span>
          <span>{fmt(max)}{unit}</span>
        </div>
      </div>
    </Shell>
  )
}
