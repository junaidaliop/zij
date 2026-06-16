# QM-quantization optimizer (Schrodinger gradient-flow)

Implements quantization-based optimization, a stochastic global optimizer whose quantized objective induces a gradient-flow diffusion that maps onto a Schrödinger equation.

The method quantizes the range of the objective with a monotonically increasing resolution, so the quantization error acts as an annealed noise source. Under the White Noise Hypothesis this turns plain gradient descent into a Langevin-type diffusion whose noise scale shrinks as $Q_p(t)\uparrow\infty$. Recasting the associated Fokker–Planck equation through the substitution $\rho = |\psi|^2$ yields a Schrödinger equation, and the resulting tunneling effect is what lets iterates climb out of local minima toward the global optimum.

In continuous time the dynamics are the stochastic differential equation $dX_t = -\nabla_x f(X_t)\,dt + \sqrt{C_q\,Q_p^{-1}(t)}\,dW_t$. Its Euler–Maruyama discretization gives the per-step parameter update:

$$
\begin{aligned}
f_Q &= Q_p^{-1}\left\lfloor Q_p\left(f + \tfrac{1}{2}Q_p^{-1}\right)\right\rfloor, \qquad Q_p(t) = \eta\, b^{\,\bar h(t)} \\
\theta_{t+1} &= \theta_t - \eta\, \nabla_\theta f(\theta_t) + \sqrt{2\,\eta\, Q_p^{-1}(t)}\; \xi_t
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $\nabla_\theta f$ the gradient, $f_Q$ the quantized objective, $Q_p(t)$ the monotonically increasing quantization resolution with base $b$ and power $\bar h(t)\uparrow\infty$, $C_q$ the diffusion constant, $\xi_t \sim \mathcal{N}(0, I)$ the injected noise, and $\lfloor\cdot\rfloor$ the floor operator.

Reference: Jinwuk Seok, Changsik Cho, "Quantum mechanical framework for quantization-based optimization: from Gradient flow to Schrödinger equation", ICLR 2026 (withdrawn) / arXiv 2026. https://arxiv.org/abs/2603.11536

---
[Back to the Canon](../index.md)
