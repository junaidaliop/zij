# Ringmaster LMO

Implements Ringmaster LMO, an asynchronous momentum method built on a linear minimization oracle (LMO) over the unit ball.

The server maintains a single momentum buffer and parameter iterate, updating them each time a gradient arrives from any worker. Incoming gradients may be stale: a gradient produced at iterate $x_{k-\delta_k}$ carries a delay $\delta_k$, and the server accepts it only when $\delta_k < R_k$, discarding excessively old gradients to balance freshness against throughput. The accepted gradient feeds an exponential moving average, and the parameter step follows the LMO direction $\mathrm{lmo}(m)\in\arg\min_{\|u\|\le 1}\langle m,u\rangle$ rather than the raw momentum, which generalizes asynchronous SGD to LMO-based (Muon-style) updates.

$$
\begin{aligned}
m_{k+1} &= (1-\alpha_k)\,m_k + \alpha_k\,g_k, \\
x_{k+1} &= x_k + \eta_k\,\mathrm{lmo}(m_{k+1}), \\
\mathrm{lmo}(y) &\in \arg\min_{\|u\|\le 1}\,\langle y, u\rangle.
\end{aligned}
$$

where $g_k=\nabla f(x_{k-\delta_k};\xi^{i_k}_{k-\delta_k})$ is the stochastic gradient returned by worker $i_k$ with delay $\delta_k$ (accepted only if $\delta_k < R_k$), $\alpha_k\in(0,1]$ is the momentum parameter, $\eta_k>0$ is the stepsize, and $\mathrm{lmo}$ is the linear minimization oracle over the unit ball.

Reference: Abdurakhmon Sadiev, Artavazd Maranjyan, Ivan Ilin, Peter Richtárik, "Ringmaster LMO: Asynchronous Linear Minimization Oracle Momentum Method", arXiv 2025. https://arxiv.org/abs/2605.18174

---
[Back to the Canon](../README.md)
