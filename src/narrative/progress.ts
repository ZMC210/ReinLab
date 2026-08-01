import { create } from 'zustand'

/** 学生答过的题：id → 第一次是否答对 */
type Answers = Record<string, boolean>

const KEY = 'reinlab-progress-v1'

function load(): Answers {
  try {
    return JSON.parse(localStorage.getItem(KEY) ?? '{}')
  } catch {
    return {}
  }
}

interface ProgressState {
  answers: Answers
  record: (id: string, correct: boolean) => void
  reset: () => void
}

export const useProgress = create<ProgressState>((set, get) => ({
  answers: load(),
  record: (id, correct) => {
    if (id in get().answers) return
    const next = { ...get().answers, [id]: correct }
    set({ answers: next })
    try {
      localStorage.setItem(KEY, JSON.stringify(next))
    } catch {
      /* 隐私模式下写不进去也无所谓 */
    }
  },
  reset: () => {
    set({ answers: {} })
    try {
      localStorage.removeItem(KEY)
    } catch {
      /* ignore */
    }
  },
}))
