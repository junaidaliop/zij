# MeZO

Implements MeZO, a memory-efficient zeroth-order optimizer that fine-tunes models using only forward passes.

MeZO adapts ZO-SGD with the classical SPSA (Simultaneous Perturbation Stochastic Approximation) gradient estimator: it perturbs all parameters by $\pm\epsilon z$ along a single shared random direction $z$, evaluates the loss at both perturbed points, and forms a finite-difference estimate of the directional derivative. Because the same $z$ multiplies the scalar finite difference, the estimated gradient is rank-one and can be applied in place, giving the same memory footprint as inference. The key trick is to store only the scalar projected gradient and the random seed used to regenerate $z$, so no full gradient vector is ever materialized.

$$
\begin{aligned}
\text{projected\_grad} &= \frac{L(\theta + \epsilon z;\, B_t) - L(\theta - \epsilon z;\, B_t)}{2\epsilon}, \\
\hat\nabla L(\theta;\, B_t) &= \text{projected\_grad}\cdot z, \\
\theta_{t+1} &= \theta_t - \eta_t \cdot \text{projected\_grad}\cdot z.
\end{aligned}
$$

where $\theta$ are the parameters, $\eta_t$ the learning rate, $\epsilon$ the perturbation scale, $L(\cdot;B_t)$ the loss on minibatch $B_t$, and $z\sim N(0, I_d)$ a single Gaussian perturbation drawn fresh each step (the default $n$-SPSA uses $n=1$; for $n>1$ the estimate averages over $n$ independent $z$).

Reference: Sadhika Malladi, Tianyu Gao, Eshaan Nichani, Alex Damian, Jason D. Lee, Danqi Chen, Sanjeev Arora, "Fine-Tuning Language Models with Just Forward Passes", NeurIPS 2023. https://arxiv.org/abs/2305.17333

---
[Back to the Canon](../README.md)
