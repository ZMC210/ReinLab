// 单独验证正文里最容易踩坑的 LaTeX 片段能被 KaTeX 正确渲染
import katex from 'katex'

const CASES = {
  'htmlId 嵌套': '\\htmlId{f1_}{\\htmlId{f1_xa}{v_\\pi}(\\htmlId{f1_xb}{s})}',
  'aligned + htmlId': '\\begin{aligned} \\htmlId{f2_}{v_\\pi(s_9)} &= 0.2 \\cdot [0 + 0.9 \\cdot 3.51] \\\\[2pt] &+ 0.2 \\cdot [-1 + 0.9 \\cdot 2.1] \\\\[4pt] &= 1.23 \\end{aligned}',
  'textcolor 十六进制': '\\textcolor{#c084fc}{\\pi}(a \\mid s)\\textcolor{#fbbf24}{r}(s,a)',
  '中文 text': '\\pi_1 \\ge \\pi_2 \\Longleftrightarrow v_{\\pi_1}(s) \\ge v_{\\pi_2}(s) \\;\\; \\text{对每一个 } s \\text{ 都成立}',
  scriptstyle中文: '{\\scriptstyle \\textcolor{#64748b}{(\\text{概率为 0 的 4 个动作已略去})}}',
  xrightarrow: 's_1 \\xrightarrow{\\;a_2,\\, r=0\\;} s_2 \\xrightarrow{\\;a_3,\\, r=-1\\;} s_7 \\cdots',
  argmax: '\\pi^*(a\\mid s), \\quad a^* = \\arg\\max_a q(s,a)',
  cases: '\\pi^*(a \\mid s) = \\begin{cases} 1, & a = a^* \\\\ 0, & \\text{否则} \\end{cases}',
  underbrace中文: '\\underbrace{-1}_{\\text{近道的罚，就在眼前}} \\quad \\text{vs} \\quad \\underbrace{(\\gamma^{k} - \\gamma^{k+2})\\cdot v}_{\\text{远路的代价}}',
  范数与期望: '\\left\\| f(v_1) - f(v_2) \\right\\|_{\\infty} \\le \\gamma \\left\\| v_1 - v_2 \\right\\|_{\\infty}, \\quad v_\\pi(s) = \\mathbb{E}\\!\\left[ G_t \\mid S_t = s \\right]',
  dfrac: 'k \\ge \\dfrac{\\ln 10^{-3}}{\\ln 0.9} \\approx 66',
  mathcal求和: 'v(s) = \\max_{a \\in \\mathcal{A}} \\sum_{s\' \\in \\mathcal{S}} p(s\'\\mid s,a) v(s\'), \\quad \\max_{\\pi(s) \\in \\Pi}',
  Neumann: '(I - \\gamma P_\\pi)^{-1} = I + \\gamma P_\\pi + \\gamma^2 P_\\pi^2 + \\cdots',
}

let fatal = 0
let warned = 0

for (const [name, tex] of Object.entries(CASES)) {
  const notes = []
  try {
    const html = katex.renderToString(tex, {
      displayMode: true,
      throwOnError: true, // 故意开启：任何解析问题都要暴露出来
      trust: true,
      strict: (code, msg) => {
        notes.push(`${code}: ${msg}`)
        return 'ignore'
      },
      output: 'html',
    })
    const ids = [...html.matchAll(/id="(f\d+_[^"]*)"/g)].map((m) => m[1])
    console.log(
      `✓ ${name}` +
        (ids.length ? `  [htmlId: ${ids.join(', ')}]` : '') +
        (notes.length ? `\n    宽松处理: ${notes.join(' | ')}` : ''),
    )
    if (notes.length) warned++
  } catch (e) {
    fatal++
    console.log(`✗ ${name}\n    ${e.message}`)
  }
}

console.log(`\n致命错误 ${fatal} 个，宽松处理 ${warned} 个（宽松处理不影响渲染）`)
process.exit(fatal ? 1 : 0)
