# SketchedAMSGrad

Implements SketchedAMSGrad, a communication-efficient distributed AMSGrad that compresses momentum with a count-sketch and error feedback.

In the gradient-averaging variant, each worker keeps a local momentum $m_t$ and sketches the error-corrected momentum before sending it to the master. The master averages the sketches and the variance statistics, applies the AMSGrad max operation to keep a non-decreasing second moment, then desketches with per-coordinate variance normalization and keeps only the top-$k$ coordinates of the resulting Adam-style update direction. The unrecovered mass is carried forward as compression error (error feedback), so no signal is permanently lost.

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) h_t^2 \\
\hat v_t &= \max(\hat v_{t-1}, v_t) \\
S_t &= \mathcal{S}\!\left(m_t + \tfrac{\alpha_{t-1}}{\alpha_t}\, e_{t-1}\right) \\
\Delta_t &= \mathrm{Top\text{-}k}\!\left(\frac{\mathcal{U}(S_t)}{\sqrt{\hat v_t}}\right) \\
e_t &= m_t + \tfrac{\alpha_{t-1}}{\alpha_t}\, e_{t-1} - \Delta_t \\
\theta_{t+1} &= \theta_t - \alpha_t\, \Delta_t
\end{aligned}
$$

where $\theta$ are the parameters, $\alpha_t$ is the step size, $g_t$ is the stochastic gradient, $h_t$ are the aggregated gradient coordinates feeding the variance estimate, $\beta_1,\beta_2\in(0,1)$ are the moment decays, $m_t$ is the (worker-local) first moment, $v_t$ and $\hat v_t$ are the second moment and its running max, $\mathcal{S}$ and $\mathcal{U}$ are the count-sketch and unsketch operators, $\mathrm{Top\text{-}k}$ keeps the $k$ largest-magnitude coordinates, and $e_t$ is the accumulated compression error. Stability is provided by initializing $v_0=\hat v_0=\epsilon$, so $\hat v_t\ge\epsilon$.

Reference: Wenhan Xian, Feihu Huang, Heng Huang, "Communication-Efficient Adam-Type Algorithms for Distributed Data Mining", arXiv 2022. https://arxiv.org/abs/2210.07454

---
[Back to the Canon](../README.md)
