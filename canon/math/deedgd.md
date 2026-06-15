# DEED-GD

Implements DEED-GD, the gradient-descent instance of a general quantization scheme that drives communication cost toward zero bits per iteration.

DEED targets distributed training where $N$ workers send gradients to a central server. Instead of quantizing the gradient itself, each worker quantizes the difference between its current gradient and a locally tracked error-feedback state, and the server quantizes the difference between the aggregated state and the previously broadcast state. Because the algorithm converges linearly, these differences shrink geometrically, so the quantization precision can be tightened on a fixed schedule $s\,c'^{\,k+1}/2$, letting the per-round bit budget decay to zero while the model still converges.

Per iteration $k$, each worker $i$ computes its local gradient $g_k^i=\nabla f_i(\theta_k)$ and runs:

$$
\begin{aligned}
d_k^i &= Q\!\left(g_k^i - s_{k-1}^i,\; \tfrac{s\,c'^{\,k+1}}{2}\right), &
s_k^i &= d_k^i + s_{k-1}^i, \\
s_k &= \frac{1}{N}\sum_{i=1}^{N} d_k^i + s_{k-1}, &
u_k &= Q\!\left(s_k - v_{k-1},\; \tfrac{s\,c'^{\,k+1}}{2}\right), \\
v_k &= u_k + v_{k-1}, &
\theta_{k+1} &= \theta_k - \eta\, v_k.
\end{aligned}
$$

where $Q(\cdot,\varepsilon)$ is an encode-then-decode quantizer whose output has absolute error at most $\varepsilon$; $d_k^i$ is the bits worker $i$ transmits and $u_k$ the bits the server broadcasts; $s_k^i$ and $v_k$ are worker- and server-side error-feedback accumulators; $s_k$ is the averaged aggregate; $\eta\in(0,\,2/(L+\mu)]$ is the step size; $c=1-\eta\mu$ and $c<c'<1$ set the precision-decay rate; and $s$ is the quantization-level scalar controlling the precision-versus-communication tradeoff.

Reference: Tian Ye, Peijun Xiao, Ruoyu Sun, "DEED: A General Quantization Scheme for Communication Efficiency in Bits", arXiv 2020. https://arxiv.org/abs/2006.11401

---
[Back to the Canon](../README.md)
