# Caputo CVNN FOGD

Implements Caputo-type fractional-order gradient descent for split-complex neural networks, training the network by descending along a Caputo fractional gradient of the error.

For a split-complex network the weights and activations are decomposed into real and imaginary parts and the real-valued quadratic error $E$ is treated as a function of the real-valued weights. Instead of the ordinary first-order gradient, the update follows the Caputo fractional derivative ${}^{C}D^{\alpha}$ of order $\alpha\in(0,1)$, which carries a memory term: the fractional derivative integrates the gradient history rather than using only the instantaneous slope, which is what the paper analyzes for monotonicity and weak convergence.

Writing $\theta$ for the real and imaginary weight components and $a$ for the lower terminal of the Caputo operator, the update is

$$
\begin{aligned}
\theta_{t+1} &= \theta_t - \eta\, {}^{C}_{a}D^{\alpha}_{\theta_t} E(\theta_t) \\
{}^{C}_{a}D^{\alpha}_{\theta}E(\theta) &= \frac{1}{\Gamma(1-\alpha)} \int_{a}^{\theta} (\theta - \tau)^{-\alpha}\, E'(\tau)\, d\tau \\
{}^{C}_{a}D^{\alpha}_{\theta_t} E(\theta_t) &\approx \frac{E'(\theta_t)}{\Gamma(2-\alpha)}\, \lvert (\theta_t - a) + \varepsilon \rvert^{\,1-\alpha}
\end{aligned}
$$

where $\theta$ are the real and imaginary parts of the weights, $\eta$ is the learning rate, $\alpha\in(0,1)$ is the fractional order, $E$ is the split-complex quadratic error, $E'$ its ordinary first-order gradient, $\Gamma(\cdot)$ the gamma function, $a$ the lower terminal, and $\varepsilon$ a small constant preventing the power term from vanishing when $\theta_t = a$. Recovering $\alpha\to 1$ reduces the rule to ordinary gradient descent; the gradient of the composite error is obtained through the Caputo chain rule ${}^{C}_{a}D^{\alpha}_{x} f(g(x)) = \frac{\partial f}{\partial g}\, {}^{C}_{a}D^{\alpha}_{x} g(x)$.

Reference: J. Wang, G. Yang, B. Zhang, Z. Sun, Y. Liu, J. Wang, "Convergence Analysis of Caputo-Type Fractional Order Complex-Valued Neural Networks", IEEE Access 5 (2017), 14560-14571. https://doi.org/10.1109/ACCESS.2017.2679185

---
[Back to the Canon](../README.md)
