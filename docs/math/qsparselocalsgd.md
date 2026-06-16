# Qsparse-local-SGD

Implements Qsparse-local-SGD, distributed SGD combining gradient quantization, sparsification, local computation, and error feedback.

Each worker runs $H$ local SGD steps between synchronizations. Only at a synchronization step does a worker compress the accumulated drift between the last global model and its local model, using a composed operator $\mathrm{QComp}_k = Q_s \circ \mathrm{Comp}_k$ that first sparsifies (top-$k$ or rand-$k$) and then quantizes. The compression error is stored in a memory term $m_t$ and added back into the next message, so the bias introduced by lossy compression is corrected over time (error feedback). The master averages the compressed messages and updates the global model.

For each worker $r$, the local gradient step is taken every iteration, while compression, the memory update, and aggregation occur only when $t+1$ is a synchronization step:

$$
\begin{aligned}
\hat{\theta}^{(r)}_{t+1/2} &= \hat{\theta}^{(r)}_t - \eta_t\, g^{(r)}_t,\quad g^{(r)}_t = \nabla f_{i^{(r)}_t}\!\big(\hat{\theta}^{(r)}_t\big),\\
\Delta^{(r)}_t &= m^{(r)}_t + \theta_t - \hat{\theta}^{(r)}_{t+1/2},\\
c^{(r)}_t &= \mathrm{QComp}_k\!\big(\Delta^{(r)}_t\big),\\
m^{(r)}_{t+1} &= \Delta^{(r)}_t - c^{(r)}_t,\\
\theta_{t+1} &= \theta_t - \frac{1}{R}\sum_{r=1}^{R} c^{(r)}_t,
\end{aligned}
$$

where $\theta_t$ is the global model, $\hat{\theta}^{(r)}_t$ the local model on worker $r$, $\eta_t$ the learning rate, $g^{(r)}_t$ the stochastic gradient on mini-batch $i^{(r)}_t$, $m^{(r)}_t$ the error-feedback memory, $R$ the number of workers, and $\mathrm{QComp}_k = Q_s \circ \mathrm{Comp}_k$ the sparsify-then-quantize compression operator. On non-synchronization steps $\theta_{t+1}=\theta_t$, $m^{(r)}_{t+1}=m^{(r)}_t$, and $\hat{\theta}^{(r)}_{t+1}=\hat{\theta}^{(r)}_{t+1/2}$.

Reference: Debraj Basu, Deepesh Data, Can Karakus, Suhas Diggavi, "Qsparse-local-SGD: Distributed SGD with Quantization, Sparsification, and Local Computations", NeurIPS 2019. https://arxiv.org/abs/1906.02367

---
[Back to the Canon](../index.md)
