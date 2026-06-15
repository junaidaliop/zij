# GADAM

Implements GADAM, a genetic-evolutionary Adam that evolves a population of models trained with Adam.

GADAM maintains a population of $g$ unit models. Within each generation, every model is trained locally with the standard Adam update on a data batch. Models are then ranked by validation fitness, and new offspring are produced by a performance-weighted crossover of parent pairs followed by a fitness-correlated mutation; the best $g$ models from the union of parents and offspring survive to the next generation. The local learning step is plain Adam:

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \\
\theta_t &= \theta_{t-1} - \eta_t \frac{\sqrt{1-\beta_2^{\,t}}}{1-\beta_1^{\,t}} \cdot \frac{m_{t-1}}{\sqrt{v_{t-1}}+\epsilon}
\end{aligned}
$$

The genetic layer combines two parents $i,j$ (with validation losses $\hat{\mathcal{L}}_i,\hat{\mathcal{L}}_j$) into a child, then mutates it:

$$
\begin{aligned}
p_{i,j} &= \frac{e^{-\hat{\mathcal{L}}_i}}{e^{-\hat{\mathcal{L}}_i}+e^{-\hat{\mathcal{L}}_j}} \\
\tilde{\theta}[m] &= \mathbb{1}(r \le p_{i,j}) \, \theta_i[m] + \mathbb{1}(r > p_{i,j}) \, \theta_j[m] \\
\tilde{\theta}[m] &= \mathbb{1}(r' \le p_q) \, \mathrm{rand}(0,1) + \mathbb{1}(r' > p_q) \, \tilde{\theta}[m]
\end{aligned}
$$

where $\theta$ are parameters, $\eta_t$ the learning rate, $g_t$ the gradient, $m_t,v_t$ the first and second moments with decays $\beta_1,\beta_2$, $\epsilon$ a stability constant, $\mathbb{1}(\cdot)$ the indicator, $r,r'$ uniform random draws, $p_{i,j}$ the softmax inheritance probability favoring the lower-loss parent, and $p_q$ a mutation rate that decreases with parent fitness.

Reference: Jiawei Zhang, Fisher B. Gouza, "GADAM: Genetic-Evolutionary ADAM for Deep Neural Network Optimization", arXiv 2018. https://arxiv.org/abs/1805.07500

---
[Back to the Canon](../README.md)
