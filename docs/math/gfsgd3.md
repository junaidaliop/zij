# GF-SGD

Implements GF-SGD, a generalized fractional-order stochastic gradient descent that replaces the integer first derivative with a Caputo-style fractional derivative.

Standard SGD takes a step along the integer-order gradient. GF-SGD instead steps along a fractional-order gradient of order $\alpha$, which introduces a memory term built from the most recent parameter displacement, scaled by $1/\Gamma(2-\alpha)$. The "generalized" qualifier is that the fractional order is allowed to exceed one: where earlier fractional schemes restrict $\alpha\in(0,1)$, GF-SGD admits $\alpha>1$ and is reported to converge faster than standard SGD in that regime, while reducing to ordinary gradient descent at $\alpha=1$.

$$
\begin{aligned}
\theta_{t+1} &= \theta_t - \frac{\eta}{\Gamma(2-\alpha)\,\bigl(|\theta_t-\theta_{t-1}|+\epsilon\bigr)^{1-\alpha}}\, g_t
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ is the learning rate, $g_t=\nabla f(\theta_t)$ is the gradient, $\Gamma(\cdot)$ is the gamma function, $\alpha$ is the fractional order (with $\alpha=1$ recovering plain SGD and $\alpha>1$ admitted by the generalized form), and $\epsilon$ is a small constant guarding the displacement term.

Reference: Zeshan Aslam Khan, Muhammad Waqar, Muhammad Junaid Ali Asif Raja, Naveed Ishtiaq Chaudhary, Abeer Tahir Mehmood Anwar Khan, Muhammad Asif Zahoor Raja, "Generalized fractional optimization-based explainable lightweight CNN model for malaria disease classification", Computers in Biology and Medicine 2025. https://doi.org/10.1016/j.compbiomed.2024.109593

---
[Back to the Canon](../index.md)
