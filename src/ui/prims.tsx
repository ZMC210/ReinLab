import { useMemo, useState, type ReactNode } from 'react'
import katex from 'katex'

/* ────────────────────────── 行内 / 独立公式 ────────────────────────── */

export function M({ children }: { children: string }) {
  const html = useMemo(
    () => katex.renderToString(children, { throwOnError: false, trust: true, strict: false }),
    [children],
  )
  return <span className="mx-[.12em]" dangerouslySetInnerHTML={{ __html: html }} />
}

export function MB({ children, className = '' }: { children: string; className?: string }) {
  const html = useMemo(
    () =>
      katex.renderToString(children, {
        displayMode: true,
        throwOnError: false,
        trust: true,
        strict: false,
      }),
    [children],
  )
  return (
    <div
      className={`my-4 overflow-x-auto text-ink ${className}`}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  )
}

/* ────────────────────────── 提示块 ────────────────────────── */

type CalloutTone = 'question' | 'insight' | 'trap' | 'intuition' | 'rigor'

const TONES: Record<CalloutTone, { label: string; accent: string; tint: string }> = {
  question: { label: '先别急，想一下', accent: 'var(--gamma)', tint: '10' },
  insight: { label: '关键', accent: 'var(--accent)', tint: '12' },
  trap: { label: '这里容易掉坑', accent: 'var(--reward)', tint: '12' },
  intuition: { label: '一句话直觉', accent: 'var(--value)', tint: '10' },
  rigor: { label: '严格地说', accent: 'var(--policy)', tint: '10' },
}

export function Callout({
  tone = 'insight',
  title,
  children,
}: {
  tone?: CalloutTone
  title?: string
  children: ReactNode
}) {
  const t = TONES[tone]
  return (
    <div
      className="my-6 rounded-2xl border px-5 py-4"
      style={{
        borderColor: `color-mix(in srgb, ${t.accent} 32%, transparent)`,
        background: `color-mix(in srgb, ${t.accent} ${t.tint}%, transparent)`,
      }}
    >
      <div className="mb-2 flex items-center gap-2">
        <span className="h-1.5 w-1.5 rounded-full" style={{ background: t.accent }} />
        <span className="text-[11px] font-semibold tracking-[.16em] uppercase" style={{ color: t.accent }}>
          {title ?? t.label}
        </span>
      </div>
      <div className="prose-body text-[15px] leading-[1.85]">{children}</div>
    </div>
  )
}

/* ────────────────────────── 要点卡 ────────────────────────── */

/**
 * 「本幕要点」。
 *
 * 叙事负责让人理解，要点负责让人带走。二者必须并存 ——
 * 只有故事，读完抓不住核心；只有要点，又退化成干巴巴的讲义。
 */
export function KeyPoints({ items, title = '本幕要点' }: { items: ReactNode[]; title?: string }) {
  return (
    <div className="my-6 rounded-2xl border border-brand/30 bg-brand/[.055] px-5 py-4">
      <div className="mb-2.5 flex items-center gap-2">
        <span className="text-[11px] font-semibold tracking-[.16em] text-brand uppercase">
          {title}
        </span>
      </div>
      <ol className="space-y-2">
        {items.map((it, i) => (
          <li key={i} className="flex gap-3 text-[14.5px] leading-[1.8] text-ink">
            <span className="mt-[3px] font-mono text-[11px] text-brand">{i + 1}</span>
            <span>{it}</span>
          </li>
        ))}
      </ol>
    </div>
  )
}

/* ────────────────────────── 滑块 ────────────────────────── */

export function Slider({
  label,
  value,
  min,
  max,
  step = 0.01,
  onChange,
  format,
  hint,
  accent = 'var(--accent)',
}: {
  label: ReactNode
  value: number
  min: number
  max: number
  step?: number
  onChange: (v: number) => void
  format?: (v: number) => string
  hint?: string
  accent?: string
}) {
  const pct = ((value - min) / (max - min)) * 100
  return (
    <div className="w-full">
      <div className="mb-1 flex items-baseline justify-between gap-3">
        <span className="text-[13px] text-dim">{label}</span>
        <span className="font-mono text-[13px] tabular-nums" style={{ color: accent }}>
          {format ? format(value) : value.toFixed(2)}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="ui-range w-full"
        style={{ '--pct': `${pct}%`, '--sl': accent } as React.CSSProperties}
      />
      {hint && <div className="mt-0.5 text-[11.5px] leading-relaxed text-faint">{hint}</div>}
    </div>
  )
}

/* ────────────────────────── 分段选择 / 开关 ────────────────────────── */

export function Seg<T extends string>({
  options,
  value,
  onChange,
  size = 'md',
}: {
  options: { value: T; label: ReactNode; hint?: string }[]
  value: T
  onChange: (v: T) => void
  size?: 'sm' | 'md'
}) {
  return (
    <div className="inline-flex rounded-xl border border-line bg-surface2 p-1">
      {options.map((o) => (
        <button
          key={o.value}
          onClick={() => onChange(o.value)}
          title={o.hint}
          className={`rounded-lg whitespace-nowrap transition-all ${
            size === 'sm' ? 'px-2.5 py-1 text-[11.5px]' : 'px-3.5 py-1.5 text-[13px]'
          } ${
            value === o.value
              ? 'bg-surface font-medium text-brand shadow-sm ring-1 ring-brand/30'
              : 'text-faint hover:text-ink'
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}

export function Toggle({
  label,
  checked,
  onChange,
}: {
  label: ReactNode
  checked: boolean
  onChange: (v: boolean) => void
}) {
  return (
    <button onClick={() => onChange(!checked)} className="group flex items-center gap-2.5 text-[13px] text-dim">
      <span
        className={`relative h-[18px] w-8 rounded-full transition-colors ${
          checked ? 'bg-brand' : 'bg-surface3'
        }`}
      >
        <span
          className={`absolute top-[2px] h-3.5 w-3.5 rounded-full bg-white shadow transition-transform ${
            checked ? 'translate-x-[16px]' : 'translate-x-[2px]'
          }`}
        />
      </span>
      <span className="group-hover:text-ink">{label}</span>
    </button>
  )
}

/* ────────────────────────── 面板 ────────────────────────── */

export function Panel({
  title,
  right,
  children,
  className = '',
}: {
  title?: ReactNode
  right?: ReactNode
  children: ReactNode
  className?: string
}) {
  return (
    <div className={`card overflow-hidden rounded-2xl ${className}`}>
      {(title || right) && (
        <div className="flex items-center justify-between gap-3 border-b border-line px-4 py-2.5">
          <div className="text-[11px] font-semibold tracking-[.14em] text-faint uppercase">
            {title}
          </div>
          {right}
        </div>
      )}
      <div className="p-4">{children}</div>
    </div>
  )
}

/* ────────────────────────── 折叠：推导 ────────────────────────── */

export function Details({
  summary,
  children,
  defaultOpen = false,
}: {
  summary: ReactNode
  children: ReactNode
  defaultOpen?: boolean
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div
      className="my-5 overflow-hidden rounded-2xl border transition-colors"
      style={{
        borderColor: open
          ? 'color-mix(in srgb, var(--policy) 38%, transparent)'
          : 'color-mix(in srgb, var(--policy) 22%, transparent)',
        background: 'color-mix(in srgb, var(--policy) 5%, transparent)',
      }}
    >
      <button onClick={() => setOpen(!open)} className="flex w-full items-center gap-3 px-5 py-3.5 text-left">
        <span
          className="text-[10px] transition-transform duration-300"
          style={{ color: 'var(--policy)', transform: open ? 'rotate(90deg)' : 'none' }}
        >
          ▶
        </span>
        <span className="text-[13.5px] font-medium text-ink">{summary}</span>
        <span className="ml-auto text-[11px] text-faint">{open ? '收起' : '展开推导'}</span>
      </button>
      <div
        className="grid transition-[grid-template-rows] duration-500 ease-out"
        style={{ gridTemplateRows: open ? '1fr' : '0fr' }}
      >
        <div className="overflow-hidden">
          <div className="prose-body border-t border-line px-5 py-4">{children}</div>
        </div>
      </div>
    </div>
  )
}

/* ────────────────────────── 代码 ────────────────────────── */

const PY_KW =
  /\b(def|return|for|in|while|if|elif|else|import|from|as|class|lambda|with|not|and|or|None|True|False|break|continue|yield|pass|range|len|max|min|sum|abs|enumerate|zip|print)\b/g

export function Code({ code, lang = 'python' }: { code: string; lang?: string }) {
  const html = useMemo(() => {
    const esc = code.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    return esc
      .replace(/(#[^\n]*)/g, '<span class="tok-c">$1</span>')
      .replace(/(&quot;|")([^"\n]*)("|&quot;)/g, '<span class="tok-s">$1$2$3</span>')
      .replace(/'([^'\n]*)'/g, "<span class=\"tok-s\">'$1'</span>")
      .replace(PY_KW, '<span class="tok-k">$1</span>')
      .replace(/\b(\d+\.?\d*)\b/g, '<span class="tok-n">$1</span>')
  }, [code])

  return (
    <div className="code-shell my-5 overflow-hidden rounded-2xl border border-line">
      <div className="flex items-center gap-2 border-b border-line px-4 py-2">
        <span className="h-2 w-2 rounded-full" style={{ background: 'var(--danger)', opacity: 0.5 }} />
        <span className="h-2 w-2 rounded-full" style={{ background: 'var(--reward)', opacity: 0.5 }} />
        <span className="h-2 w-2 rounded-full" style={{ background: 'var(--value)', opacity: 0.5 }} />
        <span className="ml-2 font-mono text-[11px] text-faint">{lang}</span>
      </div>
      <pre className="overflow-x-auto px-4 py-3.5 font-mono text-[12.5px] leading-[1.75] text-ink">
        <code dangerouslySetInnerHTML={{ __html: html }} />
      </pre>
    </div>
  )
}

/* ────────────────────────── 统计小卡 ────────────────────────── */

export function Stat({
  label,
  value,
  color,
  hint,
}: {
  label: ReactNode
  value: ReactNode
  color?: string
  hint?: string
}) {
  return (
    <div className="rounded-xl border border-line bg-surface2 p-3 text-center" title={hint}>
      <div className="text-[11px] leading-snug text-faint">{label}</div>
      <div className="mt-1 font-mono text-[22px] leading-none" style={{ color: color ?? 'var(--ink)' }}>
        {value}
      </div>
    </div>
  )
}
