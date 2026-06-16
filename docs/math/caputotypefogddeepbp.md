# Caputo-Type FOGD (Deep BP)

Implements Caputo-Type FOGD (Deep BP), fractional-order gradient descent for deep backpropagation networks using the Caputo derivative.

The method replaces the integer-order gradient with a Caputo fractional derivative of order $v \in (0,1)$ of the loss with respect to each weight. Applying the Caputo operator to the power-law form $\theta^{1-v}$ that arises from the backpropagated error yields a closed-form fractional gradient scaled by the Gamma function, which is then used in the ordinary descent step. The fractional order interpolates between memory-laden updates and the classical first-order rule recovered as $v \to 1$.

$$
\begin{aligned}
D^{v}_{\theta} E &= \frac{g_t \, \theta^{1-v}}{\Gamma(2 - v)}, \\
\theta_{t+1} &= \theta_t - \eta \, D^{v}_{\theta} E.
\end{aligned}
$$

where $\theta$ is a weight, $g_t = \delta^{l+1} a^{l}$ is the standard integer-order error gradient backpropagated to that weight, $v \in (0,1)$ is the fractional order, $\Gamma(\cdot)$ is the Gamma function, and $\eta > 0$ is the learning rate. With $L_2$ regularization of strength $\lambda$, the fractional gradient gains the term $\lambda \theta \, 2^{1-v} / \Gamma(3 - v)$.

Reference: Y. Chen, G. Zhao, "A Caputo-type fractional-order gradient descent learning of deep BP neural networks", 2019 IEEE 3rd Advanced Information Management, Communicates, Electronic and Automation Control Conference (IMCEC), pp. 546-550, 2019. https://doi.org/10.1109/IMCEC46724.2019.8984089

---
[Back to the Canon](../index.md)
