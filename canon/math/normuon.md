# SingleDeviceNorMuon

Implements NorMuon, a Muon variant that follows Newton-Schulz orthogonalization
with a per-neuron second-moment normalization.

NorMuon keeps Muon's orthogonalized momentum but observes that the
orthogonalized update still leaves the per-neuron (per-row) magnitudes
unbalanced. It maintains a second-moment estimate $v_t$ for each row of the
orthogonalized update $O_t$, formed from the row-wise mean of squared entries,
and rescales each row by its root second moment. This couples Muon's
conditioning across directions with an Adam-style adaptive learning rate per
neuron. A Frobenius-norm rescaling restores the update's RMS magnitude so the
learning rate carries over from Muon, and decoupled weight decay is applied to
the parameters.

$$
\begin{aligned}
M_t &= \beta_1\, M_{t-1} + (1 - \beta_1)\, g_t \\
O_t &= \mathrm{NewtonSchulz}(M_t) \\
v_t &= \beta_2\, v_{t-1} + (1 - \beta_2)\, \mathrm{mean}_{\text{cols}}(O_t \odot O_t) \\
\hat{O}_t &= O_t \oslash \left( \sqrt{V_t} + \epsilon \right) \\
\hat{\eta} &= \frac{0.2\, \eta\, \sqrt{m\,n}}{\lVert \hat{O}_t \rVert_F} \\
\theta_t &= \theta_{t-1} - \eta\,\lambda\,\theta_{t-1} - \hat{\eta}\, \hat{O}_t
\end{aligned}
$$

where $\theta \in \mathbb{R}^{m \times n}$ are the 2D parameters, $\eta$ is the
learning rate, $g_t$ is the gradient, $M_t$ is the first-moment (momentum)
buffer, $\mathrm{NewtonSchulz}$ is Muon's fixed-coefficient Newton-Schulz
orthogonalization, $\odot$ and $\oslash$ are elementwise product and division,
$\mathrm{mean}_{\text{cols}}$ takes the mean across columns to give a per-row
(per-neuron) scalar, $v_t \in \mathbb{R}^{m}$ is the per-neuron second-moment
estimate, $V_t$ broadcasts $v_t$ across the columns of each row, $\hat{O}_t$ is
the neuron-normalized update, $\lVert \cdot \rVert_F$ is the Frobenius norm,
$\hat{\eta}$ is the RMS-matching learning rate, $\beta_1$ and $\beta_2$ are the
moment decay rates, $\epsilon$ guards the division, and $\lambda$ is the
decoupled weight-decay coefficient.

Reference: Zichong Li, Liming Liu, Chen Liang, Weizhu Chen, Tuo Zhao,
"NorMuon: Making Muon more efficient and scalable", ICLR 2026.
https://arxiv.org/abs/2510.05491

---
[Back to the Canon](../README.md)
