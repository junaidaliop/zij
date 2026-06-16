# FTRL

Implements FTRL-Proximal, an online learning method that combines the accuracy of gradient descent with strong L1-induced sparsity.

FTRL-Proximal ("Follow The Proximally Regularized Leader") chooses each round's coefficients as the argmin of the accumulated linearized loss plus a proximal $L_2$ stabilizer and an $L_1$ penalty. The trick is a lazy representation: instead of storing the weights $\theta$, it stores a per-coordinate accumulator $z$, so the $L_1$ term yields exact zeros and the closed-form update is a soft-threshold. With $\lambda_1=0$ it reproduces ordinary online gradient descent. Each coordinate uses its own learning rate that decays with the accumulated squared gradient, giving high rates to rarely seen features and low rates to frequent ones.

Per coordinate $i$, writing $g_i$ for the gradient entry and $n_i=\sum_s g_{s,i}^2$, the per-step update is:

$$
\begin{aligned}
\eta_{t,i} &= \frac{\alpha}{\beta + \sqrt{n_i}} \\
\sigma_i &= \frac{1}{\alpha}\left(\sqrt{n_i + g_i^2} - \sqrt{n_i}\right) \\
z_i &\leftarrow z_i + g_i - \sigma_i\, \theta_{t,i} \\
n_i &\leftarrow n_i + g_i^2 \\
\theta_{t+1,i} &= \begin{cases} 0 & \text{if } |z_i| \le \lambda_1 \\ -\left(\dfrac{\beta + \sqrt{n_i}}{\alpha} + \lambda_2\right)^{-1}\!\left(z_i - \mathrm{sgn}(z_i)\,\lambda_1\right) & \text{otherwise} \end{cases}
\end{aligned}
$$

where $\theta$ are the parameters, $g_i$ the gradient (for logistic regression $g_i=(p_t-y_t)x_i$), $z_i$ the lazy accumulator, $n_i$ the running sum of squared gradients, $\eta_{t,i}$ the per-coordinate learning rate set by global constants $\alpha,\beta$, $\sigma_i$ the incremental inverse-rate increment satisfying $\sigma_i=\tfrac{1}{\eta_{t,i}}-\tfrac{1}{\eta_{t-1,i}}$, $\lambda_1$ the $L_1$ regularization strength, and $\lambda_2$ the $L_2$ regularization strength.

Reference: H. Brendan McMahan, Gary Holt, D. Sculley, Michael Young, Dietmar Ebner, Julian Grady, Lan Nie, Todd Phillips, Eugene Davydov, Daniel Golovin, Sharat Chikkerur, Dan Liu, Martin Wattenberg, Arnar Mar Hrafnkelsson, Tom Boulos, Jeremy Kubica, "Ad Click Prediction: a View from the Trenches", KDD 2013. https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41159.pdf

---
[Back to the Canon](../index.md)
