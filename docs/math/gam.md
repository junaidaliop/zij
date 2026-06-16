# GAM

Implements GAM (Gradient norm Aware Minimization), a sharpness-aware method that penalizes first-order flatness by controlling the maximum gradient norm in a neighborhood.

GAM measures sharpness through first-order flatness, defined as the radius $\rho$ scaled maximum gradient norm over a ball $B(\theta,\rho)$ around the current parameters, and adds it to the training loss as a regularizer with coefficient $\alpha$. To approximate the gradient of this penalty, GAM ascends toward the adversarial point $\theta^{\mathrm{adv}}$ that maximizes the gradient norm within the ball, then evaluates a Hessian-vector product there. The final step combines the ordinary loss gradient with this scaled gradient-norm term, both computed on the same mini-batch.

$$
\begin{aligned}
f_t &= \nabla^2 \mathcal{L}(\theta_t)\,\frac{\nabla \mathcal{L}(\theta_t)}{\lVert \nabla \mathcal{L}(\theta_t) \rVert + \xi} \\
\theta_t^{\mathrm{adv}} &= \theta_t + \rho_t\,\frac{f_t}{\lVert f_t \rVert + \xi} \\
h_t^{\mathrm{loss}} &= \nabla \mathcal{L}(\theta_t), \qquad
h_t^{\mathrm{norm}} = \rho_t\,\nabla^2 \mathcal{L}(\theta_t^{\mathrm{adv}})\,\frac{\nabla \mathcal{L}(\theta_t^{\mathrm{adv}})}{\lVert \nabla \mathcal{L}(\theta_t^{\mathrm{adv}}) \rVert + \xi} \\
\theta_{t+1} &\leftarrow \theta_t - \eta_t\left(h_t^{\mathrm{loss}} + \alpha\,h_t^{\mathrm{norm}}\right)
\end{aligned}
$$

where $\theta$ are the parameters, $\eta_t$ the learning rate, $\rho_t$ the neighborhood radius, $\alpha$ the flatness coefficient, $\xi$ a small numerical constant, $\mathcal{L}$ the mini-batch loss, $\nabla^2 \mathcal{L}$ its Hessian (applied as a Hessian-vector product), $f_t$ the gradient of the gradient norm, and $\theta_t^{\mathrm{adv}}$ the adversarial point maximizing the gradient norm within the ball of radius $\rho_t$.

Reference: Xingxuan Zhang, Renzhe Xu, Han Yu, Hao Zou, Peng Cui, "Gradient Norm Aware Minimization Seeks First-Order Flatness and Improves Generalization", CVPR 2023. https://arxiv.org/abs/2303.03108

---
[Back to the Canon](../index.md)
