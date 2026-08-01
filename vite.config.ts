import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    open: true,
    host: '0.0.0.0',
    // 校验脚本的产物落在 scripts/.out，别让它触发整页刷新
    watch: { ignored: ['**/scripts/.out/**', '**/dist/**'] },
  },
})
