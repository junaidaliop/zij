# AdamMini

Implements Adam-mini, a memory-efficient Adam variant that assigns a
single second-moment value, and hence a single learning rate, to each
parameter block.


$$
\begin{aligned}
     m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t                       \\
     v_{b,t} &= \beta_2 v_{b,t-1} + (1 - \beta_2)
         \mathrm{mean}\!\left(g_{b,t} \odot g_{b,t}\right)      \\
     \hat{m}_t &= m_t / (1 - \beta_1^t), \qquad
         \hat{v}_{b,t} = v_{b,t} / (1 - \beta_2^t)                    \\
     \theta_{b,t} &= \theta_{b,t-1} - \eta\,
         \frac{\hat{m}_{b,t}}{\sqrt{\hat{v}_{b,t}} + \epsilon}
\end{aligned}
$$

where the blocks $b$ follow the model architecture: embedding and
output layers keep Adam's coordinate-wise second moment, query and key
projections use one block per attention head, fused QKV weights use one
block per head and query group, and every remaining parameter tensor
forms a single block. Weight decay is decoupled as in AdamW and disabled
for normalization layers.


**Note:** The constructor takes the model itself rather than a parameter iterable, since the block partition is derived from parameter names. A plain iterable of tensors is also accepted; its entries are treated as unnamed, one block per tensor.

Reference: Yushun Zhang, Congliang Chen, Ziniu Li, Tian Ding, Chenwei Wu,
Diederik P. Kingma, Yinyu Ye, Zhi-Quan Luo, Ruoyu Sun,
"Adam-mini: Use Fewer Learning Rates To Gain More", ICLR 2025.
https://arxiv.org/abs/2406.16793

---
[Back to the Canon](../index.md)
