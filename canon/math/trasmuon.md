# TrasMuon

Implements TrasMuon, a trust-region adaptive scaling extension of Muon for orthogonalized momentum.

TrasMuon orthogonalizes the momentum via Newton-Schulz iteration as in Muon, then applies two corrections on top of the orthogonal direction. A row-wise second-moment scale adapts the magnitude per output feature, and an energy-based feature-wise damping vector $c_t$ shrinks columns whose input-feature energy spikes above a running reference, acting as a soft trust region. The step size is RMS-calibrated so the effective update norm stays stable across layers, and weight decay is decoupled.

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
\tilde m_t &= m_t \big/ \Big( \|m_t\|_F / \sqrt{d_{\mathrm{out}} d_{\mathrm{in}}} + \epsilon \Big) \\
O_t &= \mathrm{NS}(\tilde m_t; T) \\
v_t^{\mathrm{row}} &= \beta_2 v_{t-1}^{\mathrm{row}} + (1-\beta_2)\, \mathrm{mean}_j\big( O_{t,\cdot j}^2 \big) \\
O_t^{\mathrm{base}} &= \mathrm{diag}\big( (v_t^{\mathrm{row}} + \epsilon)^{-1/2} \big)\, O_t \\
E_{t,j} &= \sum_{i=1}^{d_{\mathrm{out}}} m_{t,ij}^2, \qquad r_{t,j} = E_{t,j} \big/ (E_t^{\mathrm{ref}} + \epsilon) \\
c_{t,j} &= \mathrm{clip}\!\left( \frac{1}{1 + \alpha \log(1 + r_{t,j})},\; c_{\min},\; 1 \right) \\
\hat\eta_t &= \eta\, \sqrt{d_{\mathrm{out}} d_{\mathrm{in}}} \big/ \big( \|O_t^{\mathrm{base}}\|_F + \epsilon \big) \\
\theta_t &= (1 - \eta\lambda)\,\theta_{t-1} - \hat\eta_t\, \big( O_t^{\mathrm{base}} \odot c_t \big)
\end{aligned}
$$

where $g_t$ is the gradient, $m_t$ the momentum, $\beta_1,\beta_2$ decay rates, $\mathrm{NS}(\cdot;T)$ the $T$-step Newton-Schulz orthogonalization, $v_t^{\mathrm{row}}$ the row-wise second moment, $E_{t,j}$ the energy of column $j$, $E_t^{\mathrm{ref}}$ an EMA of the median column energy, $r_{t,j}$ the relative energy ratio, $c_t \in [c_{\min},1]^{d_{\mathrm{in}}}$ the feature-wise damping vector ($\alpha$ a damping strength, broadcast over columns by $\odot$), $\hat\eta_t$ the RMS-calibrated step size, $\eta$ the base learning rate, $\lambda$ the decoupled weight decay, and $\epsilon$ a stability constant.

Reference: Peng Cheng, Jiucheng Zang, Qingnan Li, Liheng Ma, Jimmy Jian, Boxing Chen, Yingxue Zhang, Yufei Cui, Wen Tong, "TrasMuon: Trust-Region Adaptive Scaling for Orthogonalized Momentum Optimizers", arXiv 2025. https://arxiv.org/abs/2602.13498

---
[Back to the Canon](../README.md)
