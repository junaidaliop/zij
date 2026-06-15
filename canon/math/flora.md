# Flora

Implements Flora, an Adafactor-style optimizer that compresses the momentum state with resampled random projections to recover high-rank updates at sublinear memory cost.

Flora observes that a LoRA update is approximately a fixed random down-/up-projection of the gradient, which confines the total weight change to a low-rank subspace. Instead of fixing the projection, Flora stores the first moment in a randomly down-projected space and resamples the projection matrix every $\kappa$ steps, so the accumulated update is no longer rank-limited while the optimizer state shrinks from $O(nm)$ to $O(nr)$. At each step the gradient $g_t$ (shape $n\times m$) is projected down to $r$ columns to update the compressed moment $m_t$, the moment is carried across a resampling by re-expressing it in the new basis, and it is projected back up to full shape to form the parameter update.

$$
\begin{aligned}
A_t &\sim \mathcal{N}\!\left(0,\tfrac{1}{r}\right), \quad A_t \in \mathbb{R}^{r\times m} \\
m_t' &= \begin{cases} m_{t-1}\, A_{t-1} A_t^{\top}, & t \equiv 0 \ (\mathrm{mod}\ \kappa)\\ m_{t-1}, & \text{otherwise} \end{cases} \\
m_t &= \beta\, m_t' + (1-\beta)\, g_t\, A_t^{\top} \\
\theta_t &= \theta_{t-1} - \gamma\, \frac{m_t\, A_t}{\mathrm{RMS}(m_t A_t)}
\end{aligned}
$$

where $\theta$ are the parameters, $\gamma$ the learning rate, $g_t$ the gradient, $m_t \in \mathbb{R}^{n\times r}$ the compressed first moment, $A_t$ the resampled Gaussian projection of rank $r$, $\beta$ the momentum decay, $\kappa$ the resampling interval, and $\mathrm{RMS}(\cdot)$ the Adafactor root-mean-square scaling of the reconstructed update $m_t A_t$.

Reference: Yongchang Hao, Yanshuai Cao, Lili Mou, "Flora: Low-Rank Adapters Are Secretly Gradient Compressors", ICML 2024. https://arxiv.org/abs/2402.03293

---
[Back to the Canon](../README.md)
