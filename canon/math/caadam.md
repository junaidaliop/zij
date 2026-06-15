# CaAdam

Implements CaAdam, a connection-aware variant of Adam that scales the learning rate per layer using architectural information.

CaAdam keeps the standard Adam moment estimates and bias correction, but multiplies the step by a per-layer scaling factor $S$ derived from the network's structure rather than from the gradient statistics. The intuition is that layers differ in their number of connections (or their depth), so a single global learning rate is suboptimal; the scaling acts as a structural prior on the effective step size. Three scaling schemes are proposed: an additive and a multiplicative scheme centered on the median connection count, and a depth-based scheme.

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t} \\
\theta_{t+1} &= \theta_t - \eta\, S \,\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} \\
S_{\mathrm{add}} &=
\begin{cases}
1 + \gamma\,\dfrac{\tilde{c}-c}{\tilde{c}-c_{\min}}, & c \le \tilde{c} \\
1 - \gamma\,\dfrac{c-\tilde{c}}{c_{\max}-\tilde{c}}, & c > \tilde{c}
\end{cases} \\
S_{\mathrm{mul}} &= \exp(\sigma \log \gamma), \qquad
\sigma =
\begin{cases}
\dfrac{\tilde{c}-c}{\tilde{c}-c_{\min}}, & c \le \tilde{c} \\
\dfrac{c-\tilde{c}}{c_{\max}-\tilde{c}}, & c > \tilde{c}
\end{cases} \\
S_{\mathrm{depth}} &= (1+\gamma)^{\frac{d_m-(1+d)}{d_m}}
\end{aligned}
$$

where $c$ is the number of connections of the layer a parameter belongs to, $\tilde{c}$, $c_{\min}$, $c_{\max}$ are the median, minimum, and maximum connection counts across layers, $d$ is the depth of the current layer and $d_m$ the total network depth, $\gamma$ is the scaling strength (default $0.95$), and $S$ is whichever of $S_{\mathrm{add}}$, $S_{\mathrm{mul}}$, $S_{\mathrm{depth}}$ is selected.

Reference: Rémi Genet, Hugo Inzirillo, "CaAdam: Improving Adam optimizer using connection aware methods", arXiv 2024. https://arxiv.org/abs/2410.24216

---
[Back to the Canon](../README.md)
