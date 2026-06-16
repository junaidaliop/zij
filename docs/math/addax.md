# Addax

Implements Addax, a memory-efficient fine-tuning method that mixes a zeroth-order MeZO step with an in-place first-order SGD step.

Pure zeroth-order fine-tuning (MeZO) avoids storing activations and gradients but converges slowly, while SGD converges fast yet needs full activation memory. Addax splits each iteration's data by sequence length: long sequences (high activation cost) go through a memory-cheap zeroth-order SPSA estimate, short sequences go through an in-place SGD gradient. The two estimates are blended with a single coefficient $\alpha$, so the method spends activation memory only where it is affordable while keeping the fast first-order signal.

The zeroth-order term perturbs all parameters along one shared random direction $z$ and uses the symmetric two-point loss difference as a scalar projected gradient; the first-order term is an ordinary minibatch gradient. The combined step is:

$$
\begin{aligned}
g^0 &= \frac{1}{|\mathcal{B}^0|} \sum_{x \in \mathcal{B}^0} \frac{\ell(\theta + \epsilon z; x) - \ell(\theta - \epsilon z; x)}{2\epsilon} \\
g^1 &= \frac{1}{|\mathcal{B}^1|} \sum_{x \in \mathcal{B}^1} \nabla \ell(\theta; x) \\
\theta &\leftarrow \theta - \eta \left( \alpha\, z\, g^0 + (1 - \alpha)\, g^1 \right)
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $z \sim \mathcal{N}(0, I)$ a shared random perturbation direction, $\epsilon$ the perturbation scale, $g^0$ the scalar zeroth-order projected gradient over the long-sequence batch $\mathcal{B}^0$, $g^1$ the first-order gradient over the short-sequence batch $\mathcal{B}^1$, and $\alpha \in [0,1]$ the coefficient mixing the zeroth-order and first-order directions.

Reference: Zeman Li, Xinwei Zhang, Peilin Zhong, Yuan Deng, Meisam Razaviyayn, Vahab Mirrokni, "Addax: Utilizing Zeroth-Order Gradients to Improve Memory Efficiency and Performance of SGD for Fine-Tuning Language Models", ICLR 2025. https://arxiv.org/abs/2410.06441

---
[Back to the Canon](../index.md)
