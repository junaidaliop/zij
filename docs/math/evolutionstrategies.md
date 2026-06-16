# Evolution Strategies

Implements Evolution Strategies, a black-box gradient estimator that updates parameters from the returns of randomly perturbed copies.

ES treats the objective $F$ as a black box and optimizes the Gaussian-smoothed objective $\mathbb{E}_{\epsilon\sim N(0,I)}\,F(\theta+\sigma\epsilon)$. Its gradient with respect to $\theta$ is given by the score-function (REINFORCE-style) estimator $\frac{1}{\sigma}\,\mathbb{E}_{\epsilon\sim N(0,I)}\{F(\theta+\sigma\epsilon)\,\epsilon\}$. At each step the algorithm samples a population of $n$ perturbations, evaluates the return of each perturbed parameter vector, and takes a stochastic gradient ascent step that weights each perturbation by its return. Because only the scalar returns must be shared, the method parallelizes across many workers with minimal communication.

$$
\begin{aligned}
F_i &= F(\theta_t + \sigma\,\epsilon_i), \qquad \epsilon_1,\dots,\epsilon_n \sim N(0, I) \\
\theta_{t+1} &= \theta_t + \alpha\,\frac{1}{n\sigma}\sum_{i=1}^{n} F_i\,\epsilon_i
\end{aligned}
$$

where $\theta$ are the parameters, $\alpha$ is the learning rate, $\sigma$ is the noise standard deviation, $n$ is the population size, $\epsilon_i$ are i.i.d. standard-normal perturbations, and $F_i$ is the return (fitness) of the $i$-th perturbed parameter vector.

Reference: Tim Salimans, Jonathan Ho, Xi Chen, Szymon Sidor, Ilya Sutskever, "Evolution Strategies as a Scalable Alternative to Reinforcement Learning", arXiv 2017. https://arxiv.org/abs/1703.03864

---
[Back to the Canon](../index.md)
