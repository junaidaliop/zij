# HEW-Local SGD

Implements HEW-Local SGD (Heterogeneous-Horizon Exact-Weight Local SGD), a SCAFFOLD-style local-SGD method that lets each node run a different number of local steps and aggregates with exactly optimized weights.

Each active node $i$ initializes its local model from the current server model $x_t$, runs $H_i$ local SGD steps with a SCAFFOLD-style control-variate correction, and reports its endpoint displacement. The server combines these displacements using weights $w_{i,t}$ and per-node amplitudes $\theta_{i,t}$ obtained by minimizing a one-step progress bound, which removes the bias caused by nodes having heterogeneous local horizons $H_i$.

$$
\begin{aligned}
\eta_{i,t} &= \frac{\theta_{i,t}}{L H_i}, \qquad y_{i,t}^{(0)} = x_t, \\
y_{i,t}^{(\ell+1)} &= y_{i,t}^{(\ell)} - \eta_{i,t}\bigl(g_{i,t,\ell} - c_{i,t} + c_t\bigr), \qquad \ell = 0,\dots,H_i-1, \\
\Delta_{i,t} &= y_{i,t}^{(H_i)} - x_t, \\
c_{i,t+1} &= c_{i,t} - c_t + \frac{1}{H_i\,\eta_{i,t}}\bigl(x_t - y_{i,t}^{(H_i)}\bigr), \\
x_{t+1} &= x_t + \sum_{i\in\mathcal{S}_t} w_{i,t}\,\Delta_{i,t}, \\
c_{t+1} &= c_t + \frac{1}{n}\sum_{i\in\mathcal{S}_t}\bigl(c_{i,t+1} - c_{i,t}\bigr),
\end{aligned}
$$

where $x_t$ is the global model, $y_{i,t}^{(\ell)}$ node $i$'s local model at inner step $\ell$, $g_{i,t,\ell}$ a minibatch gradient, $c_{i,t}$ the local and $c_t$ the global control variate, $H_i$ the number of local steps at node $i$, $\eta_{i,t}$ the local step size, $L$ the smoothness constant, $\mathcal{S}_t$ the active set, and $w_{i,t}$, $\theta_{i,t}$ the aggregation weights and amplitudes chosen by minimizing the one-step upper bound of Theorem 3.2.

Reference: Dmitry Pasechnyuk-Vilensky and Martin Takáč, "Heterogeneous-Horizon Exact-Weight Local SGD", arXiv 2026. https://arxiv.org/abs/2604.24463

---
[Back to the Canon](../index.md)
