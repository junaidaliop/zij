# Double Preconditioning (DoPr)

Implements Double Preconditioning (DoPr), a layer-wise scheme that applies an activation-covariance preconditioner before any base optimizer.

DoPr decouples preconditioning into two stages. First, an activation preconditioner (AP) right-multiplies the layer gradient by the inverse uncentered covariance of that layer's input activations, reshaping the update along the local activation metric. The resulting AP-gradient is then fed into an ordinary gradient preconditioner (GP) such as Adam or Muon, which produces the descent direction. AP is a drop-in intervention aimed at test-time performance rather than validation loss.

For a feedforward layer with weights $\theta$, gradient $g_t$, and input activations $z_i$:

$$
\begin{aligned}
\hat{\Sigma}_z &= \frac{1}{n} \sum_{i=1}^{n} z_i z_i^\top \\
M_t &= g_t \, \bigl( \hat{\Sigma}_z + \gamma \, \mathrm{tr}(\hat{\Sigma}_z) \, I \bigr)^{-1} \\
D_t &= \mathrm{GP}(M_t) \\
\theta_t &= (1 - \eta \lambda) \, \theta_{t-1} - \eta \, D_t
\end{aligned}
$$

where $\hat{\Sigma}_z$ is the empirical second-moment matrix of the $n$ layer-input activations $z_i$, $\gamma$ damps the inverse via the trace of $\hat{\Sigma}_z$, $\mathrm{GP}(\cdot)$ is the chosen base gradient preconditioner, $\eta$ is the learning rate, and $\lambda$ is the decoupled weight decay.

Reference: Thomas T. Zhang, Alok Shah, Yifei Zhang, Vincent Zhang, Nikolai Matni, Max Simchowitz, "Double Preconditioning (DoPr): Optimization for Test-Time Performance, not Validation Loss", arXiv 2026. https://arxiv.org/abs/2606.06418

---
[Back to the Canon](../README.md)
