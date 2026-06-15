# BGADAM

Implements BGADAM, a genetic-evolutionary wrapper around Adam that breeds a population of models to escape local optima.

BGADAM maintains $g$ candidate models per generation, each trained for several steps with the standard Adam update. After training, models are scored by their validation loss and assigned selection probabilities; pairs of parents are then recombined by a weight-wise crossover and perturbed by mutation to form the next generation. A boosting weight is carried over the training instances so that examples misclassified by the current population are emphasized in subsequent generations. Only the genetic-evolutionary operators below are specific to BGADAM; the inner training of each model uses Adam unchanged.

$$
\begin{aligned}
p_i &= \frac{\exp(-\hat{l}_i)}{\sum_{j=1}^{g}\exp(-\hat{l}_j)}, \\
\tau &= \frac{p_{i_n}}{p_{i_n}+p_{j_n}}, \qquad p_m = p\,(1 - p_{i_n} - p_{j_n}), \\
\theta_n[h] &= \mathbb{1}(r \le \tau)\,\theta_{i_n}[h] + \mathbb{1}(r > \tau)\,\theta_{j_n}[h], \\
\theta_n[h] &\leftarrow \mathbb{1}(r' \le p_m)\,\mathcal{N}(0, 0.01) + \mathbb{1}(r' > p_m)\,\theta_n[h], \\
z_i &\leftarrow z_i \cdot \exp\!\big(\alpha\,(2\,\mathbb{1}[\text{misclassified}_i] - 1)\big), \qquad z_i \leftarrow \frac{z_i}{\sum_k z_k}.
\end{aligned}
$$

where $\theta_n[h]$ is the $h$-th weight of offspring model $n$ bred from parents $i_n, j_n$; $\hat{l}_i$ is the normalized validation loss of model $i$; $p_i$ is its selection probability; $\tau$ is the crossover threshold and $p_m$ the mutation rate (scaled by base rate $p$); $r, r' \sim \mathrm{U}(0,1)$; $\mathbb{1}(\cdot)$ is the indicator; $\mathcal{N}(0,0.01)$ is Gaussian mutation noise; $z_i$ is the boosting weight on training instance $i$ and $\alpha$ the boosting coefficient. Each model is trained with Adam between generations.

Reference: Jiyang Bai, Yuxiang Ren, Jiawei Zhang, "BGADAM: Boosting based Genetic-Evolutionary ADAM for Neural Network Optimization", arXiv 2019. https://arxiv.org/abs/1908.08015

---
[Back to the Canon](../README.md)
