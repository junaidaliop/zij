# ADEF

Implements ADEF (Accelerated Distributed Error Feedback), a Nesterov-accelerated distributed method with contractive compression, error feedback, and gradient-difference compression.

ADEF runs an accelerated (estimate-sequence) scheme on a server coordinating $n$ clients that may only send compressed messages. To make compression safe under data heterogeneity it combines two mechanisms. A control variate $\tilde g_t^i$ is maintained by compressing the change in each client's stochastic gradient (gradient-difference compression), so the residual that still needs to be sent is small. Error feedback then compresses that residual $\delta_t^i$ while carrying the leftover compression error forward in a per-client memory $e_t^i$, scaled by the acceleration weight $a_{t+1}$. The server averages both compressed messages to form an inexact gradient $\hat g_t$ and plugs it into the accelerated coupling of the iterate $\theta_t$ and the auxiliary mirror point $\nu_t$ through the extrapolation point $y_t$.

For each client $i$, with $A_{t+1}=A_t+a_{t+1}$, $y_t=\tfrac{A_t}{A_{t+1}}\theta_t+\tfrac{a_{t+1}}{A_{t+1}}\nu_t$, and stochastic gradient $g_t^i=g_i(y_t,\xi_t^i)$, the update is:

$$
\begin{aligned}
\tilde\delta_t^i &= g_t^i-\tilde g_{t-1}^i, \qquad \tilde\Delta_t^i = \mathcal{C}(\tilde\delta_t^i), \qquad \tilde g_t^i = \tilde g_{t-1}^i+\tilde\Delta_t^i, \\
\delta_t^i &= g_t^i-\tilde g_t^i-\tfrac{1}{a_{t+1}}\,e_t^i, \qquad \Delta_t^i = \mathcal{C}(\delta_t^i), \qquad e_{t+1}^i = a_{t+1}\big(\Delta_t^i-\delta_t^i\big), \\
\tilde g_t &= \tilde g_{t-1}+\tfrac{1}{n}\textstyle\sum_{i=1}^{n}\tilde\Delta_t^i, \qquad \hat g_t = \tilde g_t+\tfrac{1}{n}\textstyle\sum_{i=1}^{n}\Delta_t^i, \\
\nu_{t+1} &= \nu_t-a_{t+1}\,\hat g_t, \qquad \theta_{t+1} = \tfrac{A_t}{A_{t+1}}\theta_t+\tfrac{a_{t+1}}{A_{t+1}}\nu_{t+1}.
\end{aligned}
$$

where $\theta_t$ are the model parameters, $\nu_t$ the auxiliary mirror sequence, $y_t$ the Nesterov extrapolation point, $(a_t)$ the accelerated step weights with cumulative sum $A_t$, $g_t^i$ the local stochastic gradient at client $i$, $\tilde g_t^i$ its control variate and $\tilde g_t$ the server-side control variate, $\mathcal{C}$ a $\delta$-contractive compressor satisfying $\mathbb{E}\,\|\mathcal{C}(x)-x\|^2\le(1-\delta)\|x\|^2$, $\tilde\Delta_t^i,\Delta_t^i$ the two compressed messages, $e_t^i$ the per-client error-feedback memory, $\hat g_t$ the aggregated inexact gradient, and $n$ the number of clients.

Reference: Yuan Gao, Anton Rodomanov, Jeremy Rack, Sebastian U. Stich, "Accelerated Distributed Optimization with Compression and Error Feedback", arXiv 2025. https://arxiv.org/abs/2503.08427

---
[Back to the Canon](../README.md)
