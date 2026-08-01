/**
 * 两套主题的令牌巡检。
 *
 * 组件只允许通过语义变量取色。这个脚本核对两件事：
 *   1. 每个被 var(--x) 引用的令牌，浅色和深色都定义了（少一个就会在那套主题下变透明/黑）
 *   2. 两套主题定义的令牌集合完全一致（防止改一边忘另一边）
 * 顺带扫一遍 src 里有没有绕过变量直接写死的颜色。
 *
 * 用法：node scripts/verify-theme.mjs
 */
import fs from 'node:fs'
import path from 'node:path'

const css = fs.readFileSync('src/index.css', 'utf8').replace(/\r\n/g, '\n')

/** 抽出某个选择器块里定义的自定义属性 */
function tokensIn(selector) {
  const at = css.search(selector)
  if (at < 0) throw new Error(`找不到选择器：${selector}`)
  const open = css.indexOf('{', at)
  let depth = 0
  let end = open
  for (let i = open; i < css.length; i++) {
    if (css[i] === '{') depth++
    else if (css[i] === '}' && --depth === 0) {
      end = i
      break
    }
  }
  const body = css.slice(open + 1, end)
  return new Set([...body.matchAll(/^\s*(--[\w-]+)\s*:/gm)].map((m) => m[1]))
}

const light = tokensIn(/:root,\s*\[data-theme='light'\]/)
const dark = tokensIn(/\[data-theme='dark'\]\s*\{/)
const theme = tokensIn(/@theme inline/)

let failed = 0
const fail = (msg) => {
  failed++
  console.log(`  \x1b[31m✗ ${msg}\x1b[0m`)
}
const ok = (msg) => console.log(`  \x1b[32m✓\x1b[0m ${msg}`)

console.log('\n\x1b[1m主题令牌对称性\x1b[0m')
const onlyLight = [...light].filter((t) => !dark.has(t))
const onlyDark = [...dark].filter((t) => !light.has(t))
if (onlyLight.length) fail(`只有浅色定义了：${onlyLight.join(', ')}`)
if (onlyDark.length) fail(`只有深色定义了：${onlyDark.join(', ')}`)
if (!onlyLight.length && !onlyDark.length) ok(`两套主题各定义 ${light.size} 个令牌，完全对称`)

console.log('\n\x1b[1m引用的令牌是否都有定义\x1b[0m')
// 组件里通过 Tailwind 别名或 var() 引用的令牌
const files = []
;(function walk(dir) {
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name)
    if (e.isDirectory()) walk(p)
    else if (/\.(tsx?|css)$/.test(e.name)) files.push(p)
  }
})('src')

// 局部变量：在使用处自己定义的（如滑块的 --sl、--pct），不算主题令牌
const LOCAL = new Set(['--sl', '--pct', '--tw-content'])
const defined = new Set([...light, ...dark, ...theme, ...LOCAL])
const missing = new Map()

for (const f of files) {
  const src = fs.readFileSync(f, 'utf8')
  for (const m of src.matchAll(/var\((--[\w-]+)/g)) {
    const t = m[1]
    if (t.startsWith('--font-') || t.startsWith('--color-')) continue
    if (!defined.has(t)) {
      if (!missing.has(t)) missing.set(t, new Set())
      missing.get(t).add(f)
    }
  }
}
if (missing.size) {
  for (const [t, where] of missing) fail(`${t} 未定义，被 ${[...where].join(', ')} 引用`)
} else {
  ok('所有 var() 引用都能解析到定义')
}

console.log('\n\x1b[1m有没有绕过变量写死颜色\x1b[0m')
const HEX = /#[0-9a-fA-F]{3,8}\b/
const RGBA = /\brgba?\(/
const PALETTE =
  /\b(?:text|bg|border|from|via|to|ring|fill|stroke|decoration|divide|outline|shadow|accent)-(?:slate|zinc|gray|neutral|stone|amber|emerald|rose|violet|sky|cyan|indigo|teal|lime|orange|red|green|blue|purple|pink|fuchsia|yellow)-\d{2,3}\b/
const ALLOW = new Set(['src/index.css', 'src/theme.ts'].map((p) => path.normalize(p)))

const offenders = []
for (const f of files) {
  if (ALLOW.has(path.normalize(f))) continue
  const src = fs.readFileSync(f, 'utf8')
  src.split('\n').forEach((line, i) => {
    if (HEX.test(line) || RGBA.test(line) || PALETTE.test(line)) {
      offenders.push(`${f}:${i + 1}  ${line.trim().slice(0, 90)}`)
    }
  })
}
if (offenders.length) {
  for (const o of offenders) fail(o)
} else {
  ok(`扫描 ${files.length} 个文件，颜色全部走语义令牌（调色板只在 index.css 与 theme.ts 里）`)
}

console.log(
  `\n${failed === 0 ? '\x1b[1;32m两套主题巡检通过\x1b[0m' : `\x1b[1;31m${failed} 处问题\x1b[0m`}\n`,
)
if (failed > 0) process.exit(1)
