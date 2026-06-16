# POET-X

Implements POET-X, a scalable orthogonal equivalence transformation that trains a network by reparameterizing each weight matrix as a fixed random matrix sandwiched between two trainable orthogonal matrices.

POET reparameterizes a weight $W$ as $W_{RP} = R\,W_0\,P$, where $W_0$ is frozen at its random initialization and only the orthogonal matrices $R$ and $P$ are learned. This preserves the singular-value spectrum of $W_0$ throughout training, which stabilizes optimization. POET-X makes this practical at scale by giving $R$ and $P$ a block-diagonal structure under random permutations, and by reconstructing each orthogonal block from a skew-symmetric matrix $Q$ via a truncated Cayley (Neumann-series) map. The trainable parameters are the upper-triangular entries of the per-block $Q$ matrices, updated by a standard first-order optimizer; the orthogonal factors are rebuilt from $Q$ at every step. After training, $R$ and $P$ are folded back into a single weight $W_{RP}$, so inference has no overhead.

$$
\begin{aligned}
W_{RP} &= R\,W_0\,P, \qquad R^\top R = I,\ \ P^\top P = I, \\
R &= \Psi_R^\top\,\mathrm{Diag}\!\big(G^1,\dots,G^{\lceil m/b\rceil}\big)\,\Psi_R, \\
G &= \mathrm{Cayley}(Q) = (I+Q)(I-Q)^{-1} \approx I + 2\big(Q + Q^2 + Q^3\big) + Q^4, \\
Q &= -Q^\top, \\
Q_{t+1} &= Q_t - \gamma\,\widehat{m}_t,\qquad m_t = \beta_1 m_{t-1} + (1-\beta_1)\,g_t .
\end{aligned}
$$

where $W_0\in\mathbb{R}^{m\times n}$ is the frozen random weight, $R\in\mathbb{R}^{m\times m}$ and $P\in\mathbb{R}^{n\times n}$ are the trainable orthogonal factors, $\Psi_R$ is a fixed random permutation, $G^k$ is a $b\times b$ orthogonal block built from a skew-symmetric $Q$ of block size $b$, $g_t$ is the gradient with respect to $Q$, $m_t$ its first moment with decay $\beta_1$, $\gamma$ the learning rate, and $\widehat{m}_t$ the base optimizer's bias-corrected update direction ($P$ is parameterized identically with its own permutation and blocks).

Reference: Zeju Qiu, Lixin Liu, Adrian Weller, Han Shi, Weiyang Liu, "POET-X: Memory-efficient LLM Training by Scaling Orthogonal Transformation", ICML 2026. https://arxiv.org/abs/2603.05500

---
[Back to the Canon](../index.md)
