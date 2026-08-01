import { useMemo } from 'react'
import type { Entity } from '../highlight/bus'
import type { MDP } from '../core/mdp'
import type { Policy } from '../core/policy'
import { useColors, type Palette } from '../theme'

/**
 * 「活的公式」的地基。
 *
 * 关键设计：公式不是字符串，是一棵表达式树。同一棵树同时支持三件事 ——
 *   1. 渲染成 LaTeX（每个节点带 htmlId，于是 DOM 与树节点一一对应）
 *   2. 求值（给定 MDP / 策略 / 当前 v）
 *   3. 声明自己「指向世界里的什么」（广播到语义高亮总线）
 * 这三件事绑在一起，才能做到悬停公式点亮网格、点击网格展开公式。
 */

export type FMode = 'symbolic' | 'numeric'

export interface FCtx {
  mdp: MDP
  pi: Policy
  gamma: number
  v: number[]
  /** 当前聚焦的状态 */
  s: number
  /** 公式里的着色必须跟着主题走，所以调色板是上下文的一部分 */
  colors: Palette
}

/** 各章统一用它构造上下文，免得每处都要记得把调色板塞进去 */
export function useFormulaCtx(
  mdp: MDP,
  pi: Policy,
  gamma: number,
  v: number[],
  s: number,
): FCtx {
  const colors = useColors()
  return useMemo(() => ({ mdp, pi, gamma, v, s, colors }), [mdp, pi, gamma, v, s, colors])
}

export interface RenderArgs {
  mode: FMode
  ctx: FCtx
  /** 渲染一个子节点；sub 用来生成稳定的路径 id */
  child: (n: FNode, sub: string) => string
}

export interface FNode {
  /** 悬停卡片的标题 */
  name?: string
  /** 悬停卡片的一句话解释 */
  desc?: string | ((ctx: FCtx) => string)
  /** 这个片段在世界里指向什么 */
  entities?: (ctx: FCtx) => Entity[]
  /** 这个片段当前的数值 */
  value?: (ctx: FCtx) => number
  render: (a: RenderArgs) => string
}

export type Registry = Map<string, FNode>

const interactive = (n: FNode) => Boolean(n.entities || n.desc || n.value)

export function renderFormula(
  root: FNode,
  uid: string,
  mode: FMode,
  ctx: FCtx,
): { tex: string; registry: Registry } {
  const registry: Registry = new Map()

  const walk = (n: FNode, path: string): string => {
    const child = (c: FNode, sub: string) => walk(c, `${path}x${sub}`)
    const body = n.render({ mode, ctx, child })
    if (!interactive(n)) return body
    registry.set(path, n)
    return `\\htmlId{${path}}{${body}}`
  }

  return { tex: walk(root, uid), registry }
}

export function nodeDesc(n: FNode, ctx: FCtx): string | undefined {
  return typeof n.desc === 'function' ? n.desc(ctx) : n.desc
}
