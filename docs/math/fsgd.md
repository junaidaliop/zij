# FSGD

Implements FSGD (Fractional Stochastic Gradient Descent), a matrix-factorization learner whose latent-factor updates add a Caputo fractional-derivative term to ordinary SGD.

FSGD targets latent-factor recommender systems, where the predicted rating is $\hat{r}_{ui} = p_u^\top q_i$ for user factor $p_u$ and item factor $q_i$, and the error on an observed entry is $e_{ui} = r_{ui} - p_u^\top q_i$. The idea is to enrich each stochastic gradient step with the fractional gradient of the squared-error loss: alongside the usual integer-order term, a second term proportional to the Caputo fractional derivative of order $\alpha$ injects a memory of the current parameter magnitude through the factor $|p_u|^{1-\alpha}$ (resp. $|q_i|^{1-\alpha}$). Tuning $\alpha \in (0,1)$ interpolates between plain SGD ($\alpha \to 1$) and a more history-aware step, which the authors report improves convergence rate and prediction accuracy on rating data.

Per observed rating $(u,i)$ the latent factors are updated element-wise as

$$
\begin{aligned}
e_{ui} &= r_{ui} - p_u^\top q_i \\
p_u &\leftarrow p_u + \eta\, e_{ui}\, q_i + \frac{\eta_\alpha}{\Gamma(2-\alpha)}\, e_{ui}\, q_i \odot |p_u|^{\,1-\alpha} \\
q_i &\leftarrow q_i + \eta\, e_{ui}\, p_u + \frac{\eta_\alpha}{\Gamma(2-\alpha)}\, e_{ui}\, p_u \odot |q_i|^{\,1-\alpha}
\end{aligned}
$$

where $p_u, q_i$ are the user and item latent-factor vectors, $\eta$ is the integer-order learning rate, $\eta_\alpha$ the fractional-order learning rate, $\alpha \in (0,1)$ the fractional order, $\Gamma$ the gamma function, $\odot$ element-wise multiplication, and $|\cdot|$ the element-wise absolute value. The two updates share the same error $e_{ui}$ and differ only in the integer-order versus Caputo fractional-order gradient of the squared-error objective.

Reference: Zeshan Aslam Khan, Naveed Ishtiaq Chaudhary, Syed Zubair, "Fractional stochastic gradient descent for recommender systems", Electronic Markets 2019. https://doi.org/10.1007/s12525-018-0297-2

---
[Back to the Canon](../index.md)
