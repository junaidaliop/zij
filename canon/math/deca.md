# DECA

Implements DECA, a decentralized full-parameter fine-tuning method that runs block-wise Adam with consensus-corrected moment estimates.

DECA partitions the model into $B$ disjoint blocks and, in each communication round, optimizes one block at a time with a server-free Adam variant. For an active block it runs an inner loop of $R$ steps that combines two ingredients: standard Adam-style moments built from fresh local gradients, and *block-wise moment approximations* (BMAs) that fold in a consensus-derived discrepancy signal. After each local Adam step a client averages its block with its neighbors, and the resulting change — the difference between the local and aggregated model — is injected back into the first and second moments. This steers the next step toward network-wide agreement while keeping it aligned with the local objective, mitigating client drift on non-IID data.

For client $i$ on active block $k$, the inner step $r$ (block/round indices dropped) is:

$$
\begin{aligned}
g_r &= \nabla_k F_i(\theta_r, \zeta), \quad \zeta \sim D_i \\
m_{r+\frac12} &= \alpha_1 m_r + (1-\alpha_1)\, g_r, \qquad v_{r+\frac12} = \alpha_2 v_r + (1-\alpha_2)\, g_r \odot g_r \\
\hat m_r &= m_{r+\frac12} / (1-\alpha_1^{\,r+1}), \qquad \hat v_r = v_{r+\frac12} / (1-\alpha_2^{\,r+1}) \\
\theta_{r+\frac12} &= \theta_r - \gamma \cdot \hat m_r / \big(\sqrt{\hat v_r} + \epsilon\big) \\
\theta_{r+1} &= \textstyle\sum_{j \in N_i} w_{i,j}\, \theta_{j,\,r+\frac12} \\
h_r &= \theta_{r+1} - \theta_r \\
m_{r+1} &= \beta_1 m_{r+\frac12} + (1-\beta_1)\, h_r, \qquad v_{r+1} = \beta_2 v_{r+\frac12} + (1-\beta_2)\, h_r \odot h_r
\end{aligned}
$$

where $\theta$ are the local block parameters, $\gamma$ the learning rate, $g_r$ the local block gradient on data $D_i$, $m$/$v$ the first- and second-order BMAs, $\alpha_1,\alpha_2$ the local-gradient decay rates, $\beta_1,\beta_2$ the consensus-signal decay rates, $\epsilon$ the stability constant, $N_i$ and $w_{i,j}$ the neighborhood and mixing weights of client $i$, and $h_r$ the consensus-derived discrepancy signal between the local and aggregated block.

Reference: Yunsheng Yuan, Shaowei Li, Kai Wang, Zhongyuan Sun, Zheng Zhang, Kai Han, Jun Luo, Feng Li, "DECA: Decentralizing Block-Wise Adam for Efficient LLM Full-Parameter Fine-Tuning on Non-IID Data", arXiv 2026. https://arxiv.org/abs/2606.03209

---
[Back to the Canon](../README.md)
