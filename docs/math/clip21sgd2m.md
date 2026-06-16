# Clip21-SGD2M

Implements Clip21-SGD2M, distributed clipped SGD with double momentum and error feedback for differential privacy.

Naive per-step gradient clipping introduces a bias that prevents convergence to the exact solution. Clip21 removes this bias by clipping the *difference* between a momentum-smoothed gradient estimate and a running buffer, then feeding the clipped increment back into that buffer (error feedback). Clip21-SGD2M layers two momentum mechanisms on top: a client-side momentum $\beta_1$ that smooths the stochastic gradient into $v_t$, and a server-side momentum $\beta_2$ that controls how fast the error-feedback buffer $g_t$ tracks $v_t$. Optional Gaussian noise $\omega_t$ added to each clipped increment yields the differentially private variant.

For each worker $i$ (gradients drawn from local data $f_i$), the per-iteration update is

$$
\begin{aligned}
\theta_{t+1} &= \theta_t - \gamma\, g_t, \\
v_{i,t+1} &= (1-\beta_1)\, v_{i,t} + \beta_1\, \nabla f_i(\theta_{t+1}; \xi_{i,t+1}), \\
c_{i,t+1} &= \mathrm{clip}_\tau\!\left(v_{i,t+1} - g_{i,t}\right) + \omega_{i,t+1}, \\
g_{i,t+1} &= g_{i,t} + \beta_2\, \mathrm{clip}_\tau\!\left(v_{i,t+1} - g_{i,t}\right), \\
g_{t+1} &= g_t + \frac{\beta_2}{n} \sum_{i=1}^{n} c_{i,t+1},
\end{aligned}
$$

where $\theta_t$ are the model parameters, $\gamma > 0$ the stepsize, $\beta_1,\beta_2 \in (0,1]$ the client- and server-side momentum coefficients, $g_t$ the aggregated error-feedback gradient estimate (with per-worker buffers $g_{i,t}$, initialized $g_0 = v_0 = 0$), $v_{i,t}$ the momentum-smoothed local gradient, $\mathrm{clip}_\tau(x) = \min\{1, \tau/\lVert x\rVert\}\, x$ the clipping operator with threshold $\tau > 0$, $\omega_{i,t} \sim \mathcal{N}(0, \sigma_\omega^2 I)$ the optional DP noise ($\sigma_\omega = 0$ in the non-private case), and $n$ the number of workers.

Reference: Rustem Islamov, Samuel Horváth, Aurelien Lucchi, Peter Richtárik, Eduard Gorbunov, "Double Momentum and Error Feedback for Clipping with Fast Rates and Differential Privacy", arXiv 2025. https://arxiv.org/abs/2502.11682

---
[Back to the Canon](../index.md)
