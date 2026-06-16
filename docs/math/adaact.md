# AdaAct

Implements AdaAct, an adaptive method that preconditions updates by the variance of layer input activations rather than of gradients.

AdaAct departs from Adam-style optimizers by replacing the per-parameter second moment of the gradient with a per-neuron second moment of the layer's input activations. For each layer, the (bias-augmented) input activation vector $\tilde a$ is collected over the mini-batch and its diagonal second moment is tracked with an exponential moving average $V_t$. The gradient momentum $M_t$ is then scaled column-wise by the inverse square-root of this activation statistic, yielding neuron-level rather than coordinate-level adaptation, and the parameters are updated with decoupled weight decay.

$$
\begin{aligned}
\tilde A_t &= \frac{1}{|B_t|}\sum_{i\in B_t}\mathrm{diag}\!\left(\tilde a_i\,\tilde a_i^{\top}\right) \\
V_t &= \beta_2 V_{t-1} + (1-\beta_2)\,\tilde A_t, \qquad \hat V_t = \frac{V_t}{1-\beta_2^{\,t}} \\
M_t &= \beta_1 M_{t-1} + (1-\beta_1)\,G_t, \qquad \hat M_t = \frac{M_t}{1-\beta_1^{\,t}} \\
\hat G_t &= \hat M_t\,\bigl(\sqrt{\hat V_t} + \epsilon I\bigr)^{-1}, \qquad \hat g_t = \mathrm{vec}(\hat G_t) \\
\theta_t &= \theta_{t-1} - \eta_t\,\bigl(\hat g_t + \lambda\,\theta_{t-1}\bigr)
\end{aligned}
$$

where $\tilde a_i$ is the layer input activation for sample $i$ augmented with a trailing $1$ for the bias, $\tilde A_t$ and $V_t$ are the diagonal (per-neuron) activation second moments over batch $B_t$, $G_t$ is the mini-batch gradient of the layer weights, $M_t$ its momentum, $\hat M_t,\hat V_t$ the bias-corrected estimates, $\hat G_t$ the preconditioned weight gradient (a matrix-times-diagonal scaling of its columns), $\eta_t$ the learning rate, $\lambda$ the decoupled weight decay, $\beta_1,\beta_2$ the decay rates, and $\epsilon$ the stability constant.

Reference: Hyunseok Seung, Jaewoo Lee, Hyunsuk Ko, "An Adaptive Method Stabilizing Activations for Enhanced Generalization", 2024 IEEE International Conference on Data Mining Workshops (ICDMW) 2024. https://arxiv.org/abs/2506.08353

---
[Back to the Canon](../index.md)
