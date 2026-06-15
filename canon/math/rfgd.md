# RFGD

Implements RFGD, a fractional gradient descent that is robust to the initial weights of a multilayer perceptron.

RFGD replaces the integer-order gradient with an enhanced Caputo fractional gradient of order $\alpha$, built so that the iteration converges to the true extreme point of a convex objective rather than to the spurious fixed point that a naive fractional gradient introduces. The lower terminal of the Caputo integral is taken at the previous iterate, so the fractional term depends on $|\theta_t - \theta_{t-1}|$; the only added hyperparameter over plain gradient descent is the order $\alpha$, and the method is reported to be far less sensitive to weight initialization than GD, Adam, Padam, AdaBelief, and AdaDiff.

Keeping the leading term of the Caputo series gives the per-coordinate update

$$
\begin{aligned}
\theta_{t+1} &= \theta_t - \eta\, g_t \cdot \frac{\left(|\theta_t - \theta_{t-1}| + \delta\right)^{1-\alpha}}{\Gamma(2-\alpha)} ,
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $g_t$ the gradient of the loss at $\theta_t$, $\alpha \in (0,1)$ the fractional order, $\theta_{t-1}$ the previous iterate serving as the Caputo lower terminal, $\delta > 0$ a small constant that prevents a singularity when consecutive iterates coincide, and $\Gamma$ the gamma function; as $\alpha \to 1$ the factor reduces to one and the step becomes ordinary gradient descent.

Reference: Xuetao Xie, Yi-Fei Pu, Jian Wang, "A fractional gradient descent algorithm robust to the initial weights of multilayer perceptron", Neural Networks 2023. https://doi.org/10.1016/j.neunet.2022.11.018

---
[Back to the Canon](../README.md)
