import { useEffect } from 'react'
import { create } from 'zustand'

/**
 * 全站唯一的语义配色。
 *
 * 一个符号在公式里是什么颜色，它在世界里就是什么颜色 —— 这不是装饰，
 * 而是为了消除认知负荷理论里的「注意力分散效应」。
 *
 * 浅色主题的色值全部压深过，保证在白底上的对比度达到可读标准；
 * SVG 与 KaTeX 需要字面色值，所以这里保留 JS 版调色板，
 * 与 index.css 里的 CSS 变量一一对应。
 */
export interface Palette {
  state: string
  policy: string
  reward: string
  gamma: string
  value: string
  qvalue: string
  prob: string
  accent: string
  danger: string
  target: string
  /** 网格：普通格底色 */
  cell: string
  cellForbidden: string
  cellTarget: string
  /** 网格：格线 */
  cellLine: string
  /** 网格：格内数字 */
  cellText: string
  /** 网格：数字描边（保证在任何底色上都看得清） */
  cellTextHalo: string
  axis: string
  grid: string
}

export const LIGHT: Palette = {
  state: '#334e68',
  policy: '#7c3aed',
  reward: '#c2740a',
  gamma: '#0369a1',
  value: '#047857',
  qvalue: '#be185d',
  prob: '#be123c',
  accent: '#0e7490',
  danger: '#e11d48',
  target: '#a16207',
  cell: '#dee2ea',
  cellForbidden: '#f2d8de',
  cellTarget: '#f2e7c4',
  cellLine: 'rgba(15,23,42,.16)',
  cellText: '#16213a',
  cellTextHalo: 'rgba(255,255,255,.8)',
  axis: 'rgba(15,23,42,.3)',
  grid: 'rgba(15,23,42,.09)',
}

export const DARK: Palette = {
  state: '#cbd5e1',
  policy: '#c084fc',
  reward: '#fbbf24',
  gamma: '#38bdf8',
  value: '#34d399',
  qvalue: '#f472b6',
  prob: '#fb7185',
  accent: '#22d3ee',
  danger: '#f43f5e',
  target: '#facc15',
  cell: '#131c2e',
  cellForbidden: '#3b1220',
  cellTarget: '#3b330f',
  cellLine: 'rgba(255,255,255,.1)',
  cellText: '#f2f6fb',
  cellTextHalo: 'rgba(0,0,0,.6)',
  axis: 'rgba(255,255,255,.22)',
  grid: 'rgba(255,255,255,.07)',
}

export type ThemeMode = 'light' | 'dark'

/* ───────────────────────── 全局设置 ───────────────────────── */

interface Settings {
  theme: ThemeMode
  /** 速读模式：只留要点、公式、交互和下注题，收起叙事段落 */
  skim: boolean
  setTheme: (t: ThemeMode) => void
  setSkim: (s: boolean) => void
}

const read = <T,>(key: string, fallback: T): T => {
  try {
    const v = localStorage.getItem(key)
    return v === null ? fallback : (JSON.parse(v) as T)
  } catch {
    return fallback
  }
}
const write = (key: string, v: unknown) => {
  try {
    localStorage.setItem(key, JSON.stringify(v))
  } catch {
    /* 隐私模式下写不进去也无所谓 */
  }
}

export const useSettings = create<Settings>((set) => ({
  theme: read<ThemeMode>('rl-theme', 'light'),
  skim: read<boolean>('rl-skim', false),
  setTheme: (theme) => {
    write('rl-theme', theme)
    set({ theme })
  },
  setSkim: (skim) => {
    write('rl-skim', skim)
    set({ skim })
  },
}))

/** 把主题写到 <html data-theme> 上，CSS 变量随之切换 */
export function useApplyTheme() {
  const theme = useSettings((s) => s.theme)
  useEffect(() => {
    document.documentElement.dataset.theme = theme
  }, [theme])
}

export function useColors(): Palette {
  return useSettings((s) => (s.theme === 'dark' ? DARK : LIGHT))
}

/* ───────────────────────── 数值 → 颜色 ───────────────────────── */

type Stop = [number, [number, number, number]]

const RAMP_LIGHT: Stop[] = [
  [0.0, [214, 220, 230]],
  [0.34, [166, 205, 222]],
  [0.6, [116, 190, 205]],
  [0.82, [229, 199, 130]],
  [1.0, [222, 156, 58]],
]

const RAMP_DARK: Stop[] = [
  [0.0, [30, 41, 90]],
  [0.35, [14, 116, 144]],
  [0.6, [34, 211, 238]],
  [0.82, [163, 230, 53]],
  [1.0, [250, 204, 21]],
]

/** 价值的发散色带：冷（低）→ 中性 → 暖（高） */
export function valueColor(x: number, lo: number, hi: number, mode: ThemeMode = 'light'): string {
  const stops = mode === 'dark' ? RAMP_DARK : RAMP_LIGHT
  const span = Math.max(hi - lo, 1e-6)
  const t = Math.min(1, Math.max(0, (x - lo) / span))
  let i = 0
  while (i < stops.length - 2 && t > stops[i + 1][0]) i++
  const [t0, c0] = stops[i]
  const [t1, c1] = stops[i + 1]
  const u = (t - t0) / (t1 - t0)
  const ch = (k: number) => Math.round(c0[k] + (c1[k] - c0[k]) * u)
  return `rgb(${ch(0)}, ${ch(1)}, ${ch(2)})`
}

export function fmt(x: number, digits = 2): string {
  if (!Number.isFinite(x)) return x > 0 ? '∞' : '-∞'
  const r = Number(x.toFixed(digits))
  return Object.is(r, -0) ? '0' : String(r)
}
