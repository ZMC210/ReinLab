<p align="center">
  <img src="docs/banner.jpg" alt="ReinLab — 看得见的强化学习" width="100%" />
</p>

<p align="center">
  <strong>ReinLab</strong> · 看得见的强化学习<br/>
  <em>把公式演出来。从贝尔曼方程一路走到 Actor-Critic。</em>
</p>

<p align="center">
  <a href="https://github.com/ZMC210/ReinLab/stargazers"><img src="https://img.shields.io/github/stars/ZMC210/ReinLab?style=flat-square&color=c2740a" alt="stars" /></a>
  <a href="#怎么学"><img src="https://img.shields.io/badge/chapters-10%2F10-047857?style=flat-square" alt="chapters" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square" alt="license" /></a>
</p>

---

## 这是什么

**ReinLab** 是一套交互式强化学习教程。

学 RL 时最常见的尴尬是：公式认得，却对不上世界里发生了什么；算法名字背得出，却说不清每一步在改什么。ReinLab 把抽象过程**演给你看** ——

- 鼠标放到公式任意一段，网格里对应的格子 / 箭头会亮起来  
- 拖动 γ、ε、学习率，价值场和策略当场重排  
- 核心概念前先下注再揭晓，猜错没关系，被纠正的那下记得最牢  

主线对齐《强化学习的数学原理》（赵世钰），从基本概念走到 Actor-Critic，十章全部可交互。

<p align="center">
  <img src="docs/screenshot-live-formula.png" alt="贝尔曼最优公式与网格世界同屏" width="100%" />
</p>
<p align="center">
  <sub>公式和世界绑在一起：看符号，同时看见它指向的那一格。</sub>
</p>

---

## 怎么学

不用按「定义 → 定理 → 证明」硬啃。每一章按幕推进，大概 8–12 分钟一幕：

1. 先遇到一个具体困惑（比如：怎么给策略打分？）  
2. 试笨办法，看它为什么不够用  
3. 引出公式 / 算法，并在世界里把它演出来  
4. 用一两道预测题确认自己真懂了  

顶栏可以切 **精读 / 速读**：第一遍建议精读；复习时开速读，只留要点、公式和交互。章首有一页纸速览，赶时间也能先抓住核心。

打开 **地图**，能看到整条算法谱系，以及「这一章为什么必须在上一章后面」。

<p align="center">
  <img src="docs/screenshot-framework-map.png" alt="强化学习算法谱系全景图" width="100%" />
</p>

---

## 十章内容

| # | 章节 | 学完你能说清什么 |
| -: | --- | --- |
| 1 | 基本概念 | 状态、动作、策略、回报、MDP 各自指画面上的什么 |
| 2 | 贝尔曼公式 | 今天的价值 = 今天的奖励 + 折扣后的明天；为什么一定有解 |
| 3 | 贝尔曼最优公式 | 加一个 max 之后，「最好」意味着什么，以及最优策略为何可确定 |
| 4 | 值迭代与策略迭代 | 两个算法其实是同一条谱系的两端，滑块能亲眼看到 |
| 5 | 蒙特卡洛 | 没有模型时如何用采样估价值；探索要付什么代价 |
| 6 | 随机近似与 SGD | 为什么「一步步挪」最终能挪到正确答案 |
| 7 | 时序差分 | Sarsa 与 Q-learning 差在哪；悬崖上谁摔得更少 |
| 8 | 值函数近似 | 表格装不下时怎么办；什么时候近似会发散 |
| 9 | 策略梯度 | 直接优化策略本身；基线如何让更新更稳 |
| 10 | Actor-Critic | 演员与评论家如何分工，前面九章在这里合流 |

每一章都在回答上一章留下的问题，顺着读下去不会丢线。

---

## 本地打开

```bash
git clone https://github.com/ZMC210/ReinLab.git
cd ReinLab
npm install
npm run dev
```

浏览器打开 http://localhost:5173 ，从第 1 章开始即可。

---

## 致谢

- 主线参考：[赵世钰《强化学习的数学原理》](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning)  
- 悬崖行走等经典设定来自 Sutton & Barto  

## License

[MIT](LICENSE)
