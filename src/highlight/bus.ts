import { useMemo } from 'react'
import { create } from 'zustand'

/**
 * 语义高亮总线。
 *
 * 公式片段、网格世界、数值表格、矩阵热力图 …… 全都只是这条总线上的
 * 发布者兼订阅者。于是「公式 ↔ 世界」的双向绑定不需要任何特殊逻辑，
 * 而且以后再加第五、第六个视图时是零成本接入。
 */
export type Entity =
  | { kind: 'state'; s: number }
  | { kind: 'value'; s: number }
  | { kind: 'policy'; s: number }
  | { kind: 'action'; s: number; a: number }
  | { kind: 'qvalue'; s: number; a: number }
  | { kind: 'reward'; s: number; a: number }
  | { kind: 'transition'; s: number; a: number; sp: number }
  | { kind: 'gamma' }

interface BusState {
  hovered: Entity[]
  /** 被点击「钉住」的状态：公式会围绕它代入数值展开 */
  focus: number | null
  setHovered: (e: Entity[]) => void
  clearHovered: () => void
  setFocus: (s: number | null) => void
}

export const useBus = create<BusState>((set) => ({
  hovered: [],
  focus: null,
  setHovered: (hovered) => set({ hovered }),
  clearHovered: () => set({ hovered: [] }),
  setFocus: (focus) => set({ focus }),
}))

export interface HighlightSets {
  /** 被直接指涉的状态 */
  states: Set<number>
  /** 作为「后继状态」被指涉的状态 */
  successors: Set<number>
  /** `${s}:${a}` */
  actions: Set<string>
  /** `${s}:${a}` —— 奖励被指涉的 (s,a) */
  rewards: Set<string>
  gamma: boolean
  any: boolean
}

const EMPTY: HighlightSets = {
  states: new Set(),
  successors: new Set(),
  actions: new Set(),
  rewards: new Set(),
  gamma: false,
  any: false,
}

export function computeSets(hovered: Entity[], nA = 5): HighlightSets {
  if (hovered.length === 0) return EMPTY
  const states = new Set<number>()
  const successors = new Set<number>()
  const actions = new Set<string>()
  const rewards = new Set<string>()
  let gamma = false

  for (const e of hovered) {
    switch (e.kind) {
      case 'state':
      case 'value':
        states.add(e.s)
        break
      case 'policy':
        states.add(e.s)
        for (let a = 0; a < nA; a++) actions.add(`${e.s}:${a}`)
        break
      case 'action':
      case 'qvalue':
        states.add(e.s)
        actions.add(`${e.s}:${e.a}`)
        break
      case 'reward':
        states.add(e.s)
        actions.add(`${e.s}:${e.a}`)
        rewards.add(`${e.s}:${e.a}`)
        break
      case 'transition':
        states.add(e.s)
        actions.add(`${e.s}:${e.a}`)
        successors.add(e.sp)
        break
      case 'gamma':
        gamma = true
        break
    }
  }
  return { states, successors, actions, rewards, gamma, any: true }
}

export function useHighlight(nA = 5): HighlightSets {
  const hovered = useBus((s) => s.hovered)
  return useMemo(() => computeSets(hovered, nA), [hovered, nA])
}
