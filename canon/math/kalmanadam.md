# Kalman-Adam

Implements Kalman-Adam, an Adam variant that replaces the heuristic exponential moving average with optimal Bayesian moment estimation.

Kalman-Adam recasts gradient moment estimation as a state-space filtering problem. Instead of tracking the first and second moments with fixed-decay exponential moving averages, it runs two independent scalar Kalman filters that treat each incoming mini-batch gradient as a noisy measurement of an underlying true moment. Each filter maintains an error-covariance (uncertainty) estimate $P_m$ and $P_v$ for the first and second moment respectively; the Kalman gain is computed from these covariances at every step, so the effective smoothing adapts to the real-time uncertainty of the gradient signal rather than relying on static $\beta_1,\beta_2$ decay rates. The filtered moments then drive an Adam-style preconditioned step, which the authors report steers training toward flatter, better-generalizing minima while reducing peak memory.

Reference: Mohsin Ali, Ruby Bhatt, "Kalman-Adam: Optimal bayesian moment estimation for memory-Efficient and generalizable deep learning", Knowledge-Based Systems vol. 342, 2026. https://doi.org/10.1016/j.knosys.2026.115907

---
[Back to the Canon](../README.md)
