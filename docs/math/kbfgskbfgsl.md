# K-BFGS / K-BFGS(L)

Implements K-BFGS / K-BFGS(L), a Kronecker-factored quasi-Newton method for training deep networks.

K-BFGS approximates each layer's inverse Hessian as a Kronecker product $H_a^l \otimes H_g^l$, where $H_a^l$ acts on the input activations and $H_g^l$ acts on the pre-activation gradients. The two factors are maintained by separate BFGS recursions: the $g$-factor uses curvature pairs built from the change in pre-activations and their gradients with Powell's double damping, while the $a$-factor uses a Hessian-action pair against the running activation covariance $A_l$. K-BFGS(L) replaces the explicit $H_g^l$ matrix with a limited-memory L-BFGS store of recent $(s,y)$ pairs.

For layer $l$ with weight matrix $W_l$, gradient $g_l$, input activations $a_{l-1}$, and pre-activation gradients $\mathbf{g}_l$:

$$
\begin{aligned}
g_l &\leftarrow \beta\, g_l + (1-\beta)\, \bar{g}_l \\
W_l &\leftarrow W_l - \alpha\, H_g^l\, g_l\, H_a^l \\
s_g^l &= \mathbb{E}_i[h_l^+(i)] - \mathbb{E}_i[h_l(i)], \qquad
y_g^l = \mathbb{E}_i[\mathbf{g}_l^+(i)] - \mathbb{E}_i[\mathbf{g}_l(i)] \\
s_a^l &= H_a^l\, \mathbb{E}_i[a_{l-1}(i)], \qquad
y_a^l = (A_l + \lambda I)\, s_a^l \\
H^+ &= (I - \rho\, s y^\top)\, H\, (I - \rho\, y s^\top) + \rho\, s s^\top, \qquad \rho = \tfrac{1}{s^\top y}
\end{aligned}
$$

where $\alpha$ is the learning rate, $\beta$ the gradient moving-average decay, $\bar{g}_l$ the minibatch-average gradient, $h_l$ the pre-activations, $A_l = \mathbb{E}_i[a_{l-1}(i) a_{l-1}(i)^\top]$ the running activation covariance, $\lambda$ a Levenberg-Marquardt damping term, and the last line is the BFGS inverse-Hessian update applied to each factor using its own (Powell-damped, for the $g$-factor) pair $(s,y)$.

Reference: Donald Goldfarb, Yi Ren, Achraf Bahamou, "Practical Quasi-Newton Methods for Training Deep Neural Networks", NeurIPS 2020. https://arxiv.org/abs/2006.08877

---
[Back to the Canon](../index.md)
