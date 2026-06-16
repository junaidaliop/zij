# GFSGD

Implements GFSGD, a generalized fractional stochastic gradient descent that replaces the integer-order gradient step with a Caputo fractional-derivative step for training a CNN.

GFSGD applies the Caputo fractional derivative of order $\alpha$ to the cost function and discretizes it under the short-memory principle with fixed memory length $E = 1$, so each step uses only the immediate previous parameter change. A first-order truncation of the Caputo series leaves a single integer-order gradient scaled by $1/\Gamma(2-\alpha)$ and by a power-law factor of the last increment. The fractional order is taken in $0 < \alpha < 2$ (the optimal value reported is $\alpha = 1.5$); at $\alpha = 1$ the scaling collapses and the rule reduces to ordinary SGD.

The parameter $\theta$ is updated each step (transcribing the paper's Equation 9) by

$$
\begin{aligned}
\theta_{e+1} &= \theta_e + \frac{\eta}{\Gamma(2-\alpha)}\, g_e \,\big(|\theta_e - \theta_{e-1}| + \epsilon\big)^{\,1-\alpha}
\end{aligned}
$$

where $\theta_e$ is the parameter at step $e$, $g_e$ is the first-order gradient of the cost function with respect to $\theta$, $\eta$ is the learning rate, $\alpha$ is the fractional order, $\Gamma$ is the gamma function, $|\theta_e - \theta_{e-1}|$ is the immediate previous increment (the $E = 1$ memory term), and $\epsilon = 10^{-8}$ guards against division by zero.

Reference: Zeshan Aslam Khan, Muhammad Waqar, Naveed Ishtiaq Chaudhary, Muhammad Junaid Ali Asif Raja, Saadia Khan, Farrukh Aslam Khan, Iqra Ishtiaq Chaudhary, Muhammad Asif Zahoor Raja, "Fractional gradient optimized explainable convolutional neural network for Alzheimer's disease diagnosis", Heliyon 2024. https://doi.org/10.1016/j.heliyon.2024.e39037

---
[Back to the Canon](../index.md)
