# First-Order Optimizers

First-order optimizers update parameters using only gradients and accumulated gradient statistics such as momentum and second-moment estimates. This page covers the stochastic gradient descent lineage, the Adam family, and more recent sign-based and variance-reduced methods. The `zij` column gives the class name for optimizers already implemented in the package.

| Optimizer | Venue | Paper | Code | `zij` |
|---|---|---|---|---|
| [ASGD](math/asgd.md) | SIAM Journal on Control and Optimization 1992 | [Acceleration of Stochastic Approximation by Averaging](https://doi.org/10.1137/0330046) | [community](https://github.com/pytorch/pytorch/blob/main/torch/optim/asgd.py) | `ASGD` |
| [Rprop](math/rprop.md) | ICNN 1993 | [A direct adaptive method for faster backpropagation learning: the RPROP algorithm](https://doi.org/10.1109/ICNN.1993.298623) | [community](https://github.com/pytorch/pytorch/blob/main/torch/optim/rprop.py) | `Rprop` |
| [Adagrad](math/adagrad.md) | JMLR 2011 | [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](https://jmlr.org/papers/v12/duchi11a.html) | [community](https://github.com/pytorch/pytorch/blob/main/torch/optim/adagrad.py) | `Adagrad` |
| [Adadelta](math/adadelta.md) | arXiv 2012 | [ADADELTA: An Adaptive Learning Rate Method](https://arxiv.org/abs/1212.5701) | [community](https://github.com/pytorch/pytorch/blob/main/torch/optim/adadelta.py) | `Adadelta` |
| [RMSprop](math/rmsprop.md) | Lecture notes 2012 | [Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf) | [community](https://github.com/pytorch/pytorch/blob/main/torch/optim/rmsprop.py) | `RMSprop` |
| [FTRL](math/ftrl.md) | KDD 2013 | [Ad Click Prediction: a View from the Trenches](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41159.pdf) | — | — |
| [SGD](math/sgd.md) | ICML 2013 | [On the importance of initialization and momentum in deep learning](https://proceedings.mlr.press/v28/sutskever13.html) | [community](https://github.com/pytorch/pytorch/blob/main/torch/optim/sgd.py) | `SGD` |
| [Adam](math/adam.md) | ICLR 2015 | [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) | [community](https://github.com/pytorch/pytorch/blob/main/torch/optim/adam.py) | `Adam` |
| [AdaMax](math/adamax.md) | ICLR 2015 | [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) | [community](https://github.com/pytorch/pytorch/blob/main/torch/optim/adamax.py) | `Adamax` |
| [Nadam](math/nadam.md) | ICLR Workshop 2016 | [Incorporating Nesterov Momentum into Adam](https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ) | [community](https://github.com/pytorch/pytorch/blob/main/torch/optim/nadam.py) | `NAdam` |
| [LARS](math/lars.md) | arXiv 2017 | [Large Batch Training of Convolutional Networks](https://arxiv.org/abs/1708.03888) | [community](https://github.com/huggingface/pytorch-image-models/blob/main/timm/optim/lars.py) | `LARS` |
| [SWATS](math/swats.md) | arXiv 2017 | [Improving Generalization Performance by Switching from Adam to SGD](https://arxiv.org/abs/1712.07628) | [community](https://github.com/jettify/pytorch-optimizer/blob/master/torch_optimizer/swats.py) | `SWATS` |
| [A2Grad](math/a2graduni.md) | arXiv 2018 | [Optimal Adaptive and Accelerated Stochastic Gradient Descent](https://arxiv.org/abs/1810.00553) | [community](https://github.com/severilov/A2Grad_optimizer) | `A2GradUni`, `A2GradInc`, `A2GradExp` |
| [AccSGD](math/accsgd.md) | ICLR 2018 | [On the insufficiency of existing momentum schemes for Stochastic Optimization](https://arxiv.org/abs/1803.05591) | [official](https://github.com/rahulkidambi/AccSGD) | `AccSGD` |
| [AMSGrad](math/amsgrad.md) | ICLR 2018 | [On the Convergence of Adam and Beyond](https://arxiv.org/abs/1904.09237) | [community](https://github.com/pytorch/pytorch/blob/main/torch/optim/adam.py) | — |
| [GADAM](math/gadam.md) | arXiv 2018 | [GADAM: Genetic-Evolutionary ADAM for Deep Neural Network Optimization](https://arxiv.org/abs/1805.07500) | — | — |
| [M-SVAG](math/msvag.md) | ICML 2018 | [Dissecting Adam: The Sign, Magnitude and Variance of Stochastic Gradients](https://arxiv.org/abs/1705.07774) | [official](https://github.com/lballes/msvag) | — |
| [PID](math/pid.md) | CVPR 2018 | [A PID Controller Approach for Stochastic Optimization of Deep Networks](https://openaccess.thecvf.com/content_cvpr_2018/html/An_A_PID_Controller_CVPR_2018_paper.html) | [official](https://github.com/tensorboy/PIDOptimizer) | `PID` |
| [VR-SGD](math/vrsgd.md) | IEEE TKDE 2018 | [VR-SGD: A Simple Stochastic Variance Reduction Method for Machine Learning](https://arxiv.org/abs/1802.09932) | — | — |
| [Yogi](math/yogi.md) | NeurIPS 2018 | [Adaptive Methods for Nonconvex Optimization](https://papers.nips.cc/paper_files/paper/2018/hash/90365351ccc7437a1309dc64e4db32a3-Abstract.html) | [community](https://github.com/jettify/pytorch-optimizer/blob/master/torch_optimizer/yogi.py) | `Yogi` |
| [AdaBound](math/adabound.md) | ICLR 2019 | [Adaptive Gradient Methods with Dynamic Bound of Learning Rate](https://arxiv.org/abs/1902.09843) | [official](https://github.com/Luolc/AdaBound) | `AdaBound`, `AdaBoundW` |
| [AdaMod](math/adamod.md) | arXiv 2019 | [An Adaptive and Momental Bound Method for Stochastic Learning](https://arxiv.org/abs/1910.12249) | [official](https://github.com/lancopku/AdaMod) | `AdaMod` |
| [AdamW](math/adamw.md) | ICLR 2019 | [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) | [official](https://github.com/loshchil/AdamW-and-SGDW) | `AdamW` |
| [AdaShift](math/adashift.md) | ICLR 2019 | [AdaShift: Decorrelation and Convergence of Adaptive Learning Rate Methods](https://arxiv.org/abs/1810.00143) | [community](https://github.com/mknbv/adashift) | `AdaShift` |
| [AggMo](math/aggmo.md) | ICLR 2019 | [Aggregated Momentum: Stability Through Passive Damping](https://arxiv.org/abs/1804.00325) | [official](https://github.com/AtheMathmo/AggMo) | `AggMo` |
| [AvaGrad](math/avagrad.md) | arXiv 2019 | [Domain-independent Dominance of Adaptive Methods](https://arxiv.org/abs/1912.01823) | [official](https://github.com/lolemacs/avagrad) | `AvaGrad` |
| [HAdam](math/hadam.md) | NeurIPS Workshop 2019 | [On Higher-order Moments in Adam](https://arxiv.org/abs/1910.06878) | — | — |
| [HyperAdam](math/hyperadam.md) | AAAI 2019 | [HyperAdam: A Learnable Task-Adaptive Adam for Network Training](https://arxiv.org/abs/1811.08996) | — | — |
| [Lookahead](math/lookahead.md) | NeurIPS 2019 | [Lookahead Optimizer: k steps forward, 1 step back](https://arxiv.org/abs/1907.08610) | [community](https://github.com/alphadl/lookahead.pytorch) | `Lookahead` |
| [NosAdam](math/nosadam.md) | IJCAI 2019 | [Nostalgic Adam: Weighting more of the past gradients when designing the adaptive learning rate](https://arxiv.org/abs/1805.07557) | — | — |
| [NovoGrad](math/novograd.md) | arXiv 2019 | [Stochastic Gradient Methods with Layer-wise Adaptive Moments for Training of Deep Networks](https://arxiv.org/abs/1905.11286) | [community](https://github.com/huggingface/pytorch-image-models/blob/main/timm/optim/nvnovograd.py) | `NovoGrad` |
| [QHAdam / QHM](math/qhadam.md) | ICLR 2019 | [Quasi-hyperbolic momentum and Adam for deep learning](https://arxiv.org/abs/1810.06801) | [official](https://github.com/facebookresearch/qhoptim) | `QHAdam`, `QHM` |
| [Ranger](math/ranger.md) | — | RAdam and Lookahead combination | [official](https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer) | `Ranger` |
| [Sadam](math/sadam.md) | arXiv 2019 | [Calibrating the Adaptive Learning Rate to Improve Convergence of ADAM](https://arxiv.org/abs/1908.00700) | — | — |
| [AdaBelief](math/adabelief.md) | NeurIPS 2020 | [AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed Gradients](https://arxiv.org/abs/2010.07468) | [official](https://github.com/juntang-zhuang/Adabelief-Optimizer) | `AdaBelief` |
| [Adam+](math/adam-v.md) | arXiv 2020 | [Adam+: A Stochastic Method with Adaptive Variance Reduction](https://arxiv.org/abs/2011.11985) | — | — |
| [AdamBS](math/adambs.md) | NeurIPS 2020 | [Adam with Bandit Sampling for Deep Learning](https://arxiv.org/abs/2010.12986) | — | — |
| [AdaSGD](math/adasgd.md) | arXiv 2020 | [AdaSGD: Bridging the gap between SGD and Adam](https://arxiv.org/abs/2006.16541) | — | — |
| [Cayley SGD](math/cayleysgd.md) | ICLR 2020 | [Efficient Riemannian Optimization on the Stiefel Manifold via the Cayley Transform](https://arxiv.org/abs/2002.01113) | [official](https://github.com/JunLi-Galios/Optimization-on-Stiefel-Manifold-via-Cayley-Transform) | — |
| [clipped-SGD](math/clippedsgd.md) | NeurIPS 2020 | [Stochastic Optimization with Heavy-Tailed Noise via Accelerated Gradient Clipping](https://arxiv.org/abs/2005.10785) | [official](https://github.com/eduardgorbunov/accelerated_clipping) | — |
| [DEAM](math/deam.md) | ASONAM 2020 | [DEAM: Adaptive Momentum with Discriminative Weight for Stochastic Optimization](https://arxiv.org/abs/1907.11307) | — | — |
| [diffGrad](math/diffgrad.md) | IEEE TNNLS 2020 | [diffGrad: An Optimization Method for Convolutional Neural Networks](https://arxiv.org/abs/1909.11015) | [official](https://github.com/shivram1987/diffGrad) | `DiffGrad` |
| [EAdam](math/eadam.md) | arXiv 2020 | [EAdam Optimizer: How ε Impact Adam](https://arxiv.org/abs/2011.02150) | [official](https://github.com/yuanwei2019/EAdam-optimizer) | — |
| [Fromage](math/fromage.md) | NeurIPS 2020 | [On the distance between two neural networks and the stability of learning](https://arxiv.org/abs/2002.03432) | [official](https://github.com/jxbz/fromage) | — |
| [Gradient Centralization (GC)](math/gradientcentralizationgc.md) | ECCV 2020 | [Gradient Centralization: A New Optimization Technique for Deep Neural Networks](https://arxiv.org/abs/2004.01461) | [official](https://github.com/Yonghongwei/Gradient-Centralization) | — |
| [LAMB](math/lamb.md) | ICLR 2020 | [Large Batch Optimization for Deep Learning: Training BERT in 76 minutes](https://arxiv.org/abs/1904.00962) | [community](https://github.com/huggingface/pytorch-image-models/blob/main/timm/optim/lamb.py) | `Lamb` |
| [LaProp](math/laprop.md) | arXiv 2020 | [LaProp: Separating Momentum and Adaptivity in Adam](https://arxiv.org/abs/2002.04839) | [official](https://github.com/Z-T-WANG/LaProp-Optimizer) | `LaProp` |
| [NIGT](math/nigt.md) | ICML 2020 | [Momentum Improves Normalized SGD](https://arxiv.org/abs/2002.03305) | [official](https://github.com/google-research/google-research/tree/master/nigt_optimizer) | — |
| [Padam](math/padam.md) | IJCAI 2020 | [Closing the Generalization Gap of Adaptive Gradient Methods in Training Deep Neural Networks](https://arxiv.org/abs/1806.06763) | [official](https://github.com/uclaml/Padam) | `PAdam` |
| [signSGD](math/signsgd.md) | ICML 2018 | [signSGD: Compressed Optimisation for Non-Convex Problems](https://arxiv.org/abs/1802.04434) | [community](https://github.com/kozistr/pytorch_optimizer) | `SignSGD` |
| [pbSGD](math/pbsgd.md) | IJCAI 2020 | [pbSGD: Powered Stochastic Gradient Descent Methods for Accelerated Non-Convex Optimization](https://www.ijcai.org/proceedings/2020/451) | [official](https://github.com/HAIRLAB/pbSGD) | — |
| [PCGrad](math/pcgrad.md) | NeurIPS 2020 | [Gradient Surgery for Multi-Task Learning](https://arxiv.org/abs/2001.06782) | [official](https://github.com/tianheyu927/PCGrad) | — |
| [RAdam](math/radam.md) | ICLR 2020 | [On the Variance of the Adaptive Learning Rate and Beyond](https://arxiv.org/abs/1908.03265) | [official](https://github.com/LiyuanLucasLiu/RAdam) | `RAdam` |
| [SGD-G2](math/sgdg2.md) | ICPR 2020 | [Stochastic Runge-Kutta methods and adaptive SGD-G2 stochastic gradient descent](https://arxiv.org/abs/2002.09304) | — | — |
| [ACMo](math/acmo.md) | AAAI 2021 | [ACMo: Angle-Calibrated Moment Methods for Stochastic Optimization](https://arxiv.org/abs/2006.07065) | — | — |
| [ACProp](math/acprop.md) | NeurIPS 2021 | [Momentum Centering and Asynchronous Update for Adaptive Gradient Methods](https://arxiv.org/abs/2110.05454) | [official](https://github.com/juntang-zhuang/ACProp-Optimizer) | — |
| [AdaL](math/adal.md) | arXiv 2021 | [AdaL: Adaptive Gradient Transformation Contributes to Convergences and Generalizations](https://arxiv.org/abs/2107.01525) | — | — |
| [AdamD](math/adamd.md) | arXiv 2021 | [AdamD: Improved bias-correction in Adam](https://arxiv.org/abs/2110.10828) | — | — |
| [AdamP](math/adamp.md) | ICLR 2021 | [AdamP: Slowing Down the Slowdown for Momentum Optimizers on Scale-invariant Weights](https://arxiv.org/abs/2006.08217) | [official](https://github.com/clovaai/AdamP) | `AdamP` |
| [Adaptive Gradient Clipping (AGC)](math/adaptivegradientclippingagc.md) | ICML 2021 | [High-Performance Large-Scale Image Recognition Without Normalization](https://arxiv.org/abs/2102.06171) | [official](https://github.com/google-deepmind/deepmind-research/tree/master/nfnets) | — |
| [AngularGrad](math/angulargrad.md) | arXiv 2021 | [AngularGrad: A New Optimization Technique for Angular Convergence of Convolutional Neural Networks](https://arxiv.org/abs/2105.10190) | [official](https://github.com/mhaut/AngularGrad) | — |
| [BGADAM](math/bgadam.md) | IJCNN 2021 | [BGADAM: Boosting based Genetic-Evolutionary ADAM for Neural Network Optimization](https://arxiv.org/abs/1908.08015) | — | — |
| [Gravity](math/gravity.md) | arXiv 2021 | [Gravity Optimizer: a Kinematic Approach on Optimization in Deep Learning](https://arxiv.org/abs/2101.09192) | [official](https://github.com/dariush-bahrami/gravity.optimizer) | `Gravity` |
| [MADGRAD](math/madgrad.md) | arXiv 2021 | [Adaptivity without Compromise: A Momentumized, Adaptive, Dual Averaged Gradient Method for Stochastic Optimization](https://arxiv.org/abs/2101.11075) | [official](https://github.com/facebookresearch/madgrad) | `MADGRAD`, `MirrorMADGRAD` |
| [MaxVA](math/maxva.md) | ECML PKDD 2021 | [MaxVA: Fast Adaptation of Step Sizes by Maximizing Observed Variance of Gradients](https://arxiv.org/abs/2006.11918) | [official](https://github.com/zhuchen03/MaxVA) | — |
| [Nero](math/nero.md) | ICML 2021 | [Learning by Turning: Neural Architecture Aware Optimisation](https://arxiv.org/abs/2102.07227) | [official](https://github.com/jxbz/nero) | — |
| [PNM](math/pnm.md) | ICML 2021 | [Positive-Negative Momentum: Manipulating Stochastic Gradient Noise to Improve Generalization](https://arxiv.org/abs/2103.17182) | [official](https://github.com/zeke-xie/Positive-Negative-Momentum) | — |
| [AdaPNM](math/adapnm.md) | ICML 2021 | [Positive-Negative Momentum: Manipulating Stochastic Gradient Noise to Improve Generalization](https://arxiv.org/abs/2103.17182) | [official](https://github.com/zeke-xie/Positive-Negative-Momentum) | `AdaPNM` |
| [Ranger21](math/ranger21.md) | arXiv 2021 | [Ranger21: a synergistic deep learning optimizer](https://arxiv.org/abs/2106.13731) | [official](https://github.com/lessw2020/Ranger21) | `Ranger21` |
| [SGDP](math/sgdp.md) | ICLR 2021 | [AdamP: Slowing Down the Slowdown for Momentum Optimizers on Scale-invariant Weights](https://arxiv.org/abs/2006.08217) | [official](https://github.com/clovaai/AdamP) | `SGDP` |
| [AdaFamily](math/adafamily.md) | arXiv 2022 | [AdaFamily: A family of Adam-like adaptive gradient methods](https://arxiv.org/abs/2203.01603) | — | — |
| [Adai](math/adai.md) | ICML 2022 | [Adaptive Inertia: Disentangling the Effects of Adaptive Learning Rate and Momentum](https://arxiv.org/abs/2006.15815) | [official](https://github.com/zeke-xie/adaptive-inertia-adai) | `Adai` |
| [AdamMC](math/adammc.md) | CVMI 2022 | [Moment Centralization based Gradient Descent Optimizers for Convolutional Neural Networks](https://arxiv.org/abs/2207.09066) | — | — |
| [Adan](math/adan.md) | arXiv 2022 | [Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models](https://arxiv.org/abs/2208.06677) | [official](https://github.com/sail-sg/Adan) | `Adan` |
| [AdaSmooth](math/adasmooth.md) | arXiv 2022 | [AdaSmooth: An Adaptive Learning Rate Method based on Effective Ratio](https://arxiv.org/abs/2204.00825) | — | `AdaSmooth` |
| [AEGDM](math/aegdm.md) | Annals of Applied Mathematics 2022 | [An Adaptive Gradient Method with Energy and Momentum](https://arxiv.org/abs/2203.12191) | [official](https://github.com/txping/AEGDM) | — |
| [Amos](math/amos.md) | arXiv 2022 | [Amos: An Adam-style Optimizer with Adaptive Weight Decay towards Model-Oriented Scale](https://arxiv.org/abs/2210.11693) | [official](https://github.com/google-research/jestimator) | `Amos` |
| [GDA-AM](math/gdaam.md) | ICLR 2022 | [GDA-AM: On the effectiveness of solving minimax optimization via Anderson Acceleration](https://arxiv.org/abs/2110.02457) | [official](https://github.com/hehuannb/GDA-AM) | — |
| [KOALA](math/koala.md) | AAAI 2022 | [KOALA: A Kalman Optimization Algorithm with Loss Adaptivity](https://arxiv.org/abs/2107.03331) | [official](https://github.com/Araachie/koala) | — |
| [RotoGrad](math/rotograd.md) | ICLR 2022 | [RotoGrad: Gradient Homogenization in Multitask Learning](https://arxiv.org/abs/2103.02631) | [official](https://github.com/adrianjav/rotograd) | — |
| [SRSGD](math/srsgd.md) | SIAM Journal on Imaging Sciences 2022 | [Scheduled Restart Momentum for Accelerated Stochastic Gradient Descent](https://arxiv.org/abs/2002.10583) | — | — |
| [Step-Tuned SGD](math/steptunedsgd.md) | Neural Processing Letters 2022 | [Second-order step-size tuning of SGD for non-convex optimization](https://arxiv.org/abs/2103.03570) | — | — |
| [AdaInject](math/adainject.md) | IEEE TAI 2023 | [AdaInject: Injection Based Adaptive Gradient Descent Optimizers for Convolutional Neural Networks](https://arxiv.org/abs/2109.12504) | [official](https://github.com/shivram1987/AdaInject) | — |
| [AdaNorm](math/adanorm.md) | WACV 2023 | [AdaNorm: Adaptive Gradient Norm Correction based Optimizer for CNNs](https://arxiv.org/abs/2210.06364) | [official](https://github.com/shivram1987/AdaNorm) | `AdaNorm` |
| [AGD](math/agd.md) | NeurIPS 2023 | [AGD: an Auto-switchable Optimizer using Stepwise Gradient Difference for Preconditioning Matrix](https://arxiv.org/abs/2312.01658) | — | — |
| [Aida](math/aida.md) | TMLR 2023 | [A DNN Optimizer that Improves over AdaBelief by Suppression of the Adaptive Stepsize Range](https://arxiv.org/abs/2203.13273) | [official](https://github.com/guoqiang-zhang-x/Aida-Optimizer) | — |
| [Lion](math/lion.md) | NeurIPS 2023 | [Symbolic Discovery of Optimization Algorithms](https://arxiv.org/abs/2302.06675) | [official](https://github.com/google/automl/tree/master/lion) | `Lion` |
| [Lookaround](math/lookaround.md) | NeurIPS 2023 | [Lookaround Optimizer: k steps around, 1 step average](https://arxiv.org/abs/2306.07684) | — | — |
| [MultiAdam](math/multiadam.md) | ICML 2023 | [MultiAdam: Parameter-wise Scale-invariant Optimizer for Multiscale Training of Physics-informed Neural Networks](https://arxiv.org/abs/2306.02816) | — | — |
| [RLEKF](math/rlekf.md) | AAAI 2023 | [RLEKF: An Optimizer for Deep Potential with Ab Initio Accuracy](https://arxiv.org/abs/2212.06989) | — | — |
| [Scheduled Weight Decay (SWD)](math/scheduledweightdecayswd.md) | NeurIPS 2023 | [On the Overlooked Pitfalls of Weight Decay and How to Mitigate Them: A Gradient-Norm Perspective](https://arxiv.org/abs/2011.11152) | [official](https://github.com/zeke-xie/stable-weight-decay-regularization) | — |
| [SGDF](math/sgdf.md) | arXiv 2023 | [Signal Processing Meets SGD: From Momentum to Filter](https://arxiv.org/abs/2311.02818) | — | — |
| [StableAdamW](math/stableadamw.md) | NeurIPS 2023 | [Stable and low-precision training for large-scale vision-language models](https://arxiv.org/abs/2304.13013) | [community](https://github.com/kozistr/pytorch_optimizer) | `StableAdamW` |
| [AdaAct](math/adaact.md) | ICDMW 2024 | [An Adaptive Method Stabilizing Activations for Enhanced Generalization](https://arxiv.org/abs/2506.08353) | — | — |
| [Adam-atan2](math/adamatan2.md) | ICML 2024 | [Scaling Exponents Across Parameterizations and Optimizers](https://arxiv.org/abs/2407.05872) | [community](https://github.com/lucidrains/adam-atan2-pytorch) | `AdamAtan2` |
| [Adam-Rel](math/adamrel.md) | NeurIPS 2024 | [Adam on Local Time: Addressing Nonstationarity in RL with Relative Adam Timesteps](https://arxiv.org/abs/2412.17113) | — | — |
| [AdEMAMix](math/ademamix.md) | arXiv 2024 | [The AdEMAMix Optimizer: Better, Faster, Older](https://arxiv.org/abs/2409.03137) | [official](https://github.com/apple/ml-ademamix) | `AdEMAMix` |
| [ADOPT](math/adopt.md) | NeurIPS 2024 | [ADOPT: Modified Adam Can Converge with Any β₂ with the Optimal Rate](https://arxiv.org/abs/2411.02853) | [official](https://github.com/iShohei220/adopt) | `ADOPT` |
| [AGS-GD](math/agsgd.md) | arXiv 2024 | [Anisotropic Gaussian Smoothing for Gradient-based Optimization](https://arxiv.org/abs/2411.11747) | — | — |
| [BADM](math/badm.md) | arXiv 2024 | [BADM: Batch ADMM for Deep Learning](https://arxiv.org/abs/2407.01640) | — | — |
| [CaAdam](math/caadam.md) | arXiv 2024 | [CaAdam: Improving Adam optimizer using connection aware methods](https://arxiv.org/abs/2410.24216) | [official](https://github.com/remigenet/CaAdam) | — |
| [CAdam](math/cadam.md) | arXiv 2024 | [CAdam: Confidence-Based Optimization for Online Learning](https://arxiv.org/abs/2411.19647) | — | — |
| [Cautious Optimizers](math/cautiousoptimizers.md) | arXiv 2024 | [Cautious Optimizers: Improving Training with One Line of Code](https://arxiv.org/abs/2411.16085) | [official](https://github.com/kyleliang919/C-Optim) | — |
| [EXAdam](math/exadam.md) | arXiv 2024 | [EXAdam: The Power of Adaptive Cross-Moments](https://arxiv.org/abs/2412.20302) | [official](https://github.com/AhmedMostafa16/EXAdam) | `EXAdam` |
| [FAdam](math/fadam.md) | arXiv 2024 | [FAdam: Adam is a natural gradient optimizer using diagonal empirical Fisher information](https://arxiv.org/abs/2405.12807) | [community](https://github.com/lessw2020/FAdam_PyTorch) | `FAdam` |
| [GrokAdamW](math/grokadamw.md) | — | AdamW variant with Grokfast-style gradient amplification | [official](https://github.com/QuixiAI/grokadamw) | `GrokAdamW` |
| [Grokfast](math/grokfast.md) | arXiv 2024 | [Grokfast: Accelerated Grokking by Amplifying Slow Gradients](https://arxiv.org/abs/2405.20233) | [official](https://github.com/ironjr/grokfast) | — |
| [INNAprop](math/innaprop.md) | arXiv 2024 | [A second-order-like optimizer with adaptive gradient scaling for deep learning](https://arxiv.org/abs/2410.05871) | [official](https://github.com/innaprop/innaprop) | — |
| [KATE](math/kate.md) | NeurIPS 2024 | [Remove that Square Root: A New Efficient Scale-Invariant Version of AdaGrad](https://arxiv.org/abs/2403.02648) | [official](https://github.com/nazya/KATE) | — |
| [MADA](math/mada.md) | ICML 2024 | [MADA: Meta-Adaptive Optimizers through hyper-gradient Descent](https://arxiv.org/abs/2401.08893) | — | — |
| [RSGDM](math/rsgdm.md) | CCSB 2024 | [Reducing Bias in Deep Learning Optimization: The RSGDM Approach](https://arxiv.org/abs/2409.15314) | — | — |
| [SET-Adam](math/setadam.md) | ECML PKDD 2024 | [On Suppressing Range of Adaptive Stepsizes of Adam to Improve Generalisation Performance](https://arxiv.org/abs/2302.01029) | — | — |
| [SNGM](math/sngm.md) | Science China Information Sciences 2024 | [Stochastic Normalized Gradient Descent with Momentum for Large-Batch Training](https://arxiv.org/abs/2007.13985) | — | — |
| [SRMM](math/srmm.md) | JMLR 2024 | [Stochastic regularized majorization-minimization with weakly convex and multi-convex surrogates](https://arxiv.org/abs/2201.01652) | [official](https://github.com/HanbaekLyu/SRMM) | — |
| [TAM](math/tam.md) | arXiv 2024 | [Torque-Aware Momentum](https://arxiv.org/abs/2412.18790) | — | — |
| [WarpAdam](math/warpadam.md) | arXiv 2024 | [WarpAdam: A new Adam optimizer based on Meta-Learning approach](https://arxiv.org/abs/2409.04244) | — | — |
| [AbsSADMM](math/abssadmm.md) | arXiv 2025 | [Stochastic ADMM with batch size adaptation for nonconvex nonsmooth optimization](https://arxiv.org/abs/2505.06921) | — | — |
| [AdamC](math/adamc.md) | arXiv 2025 | [Why Gradients Rapidly Increase Near the End of Training](https://arxiv.org/abs/2506.02285) | — | — |
| [AdamNX](math/adamnx.md) | arXiv 2025 | [AdamNX: An Adam improvement algorithm based on a novel exponential decay mechanism for the second-order moment estimate](https://arxiv.org/abs/2511.13465) | [official](https://github.com/mengzhu0308/AdamNX) | — |
| [AdamS](math/adams.md) | EMNLP 2025 | [AdamS: Momentum Itself Can Be A Normalizer for LLM Pretraining and Post-training](https://arxiv.org/abs/2505.16363) | — | — |
| [adaNAPG](math/adanapg.md) | arXiv 2025 | [Boosting Accelerated Proximal Gradient Method with Adaptive Sampling for Stochastic Composite Optimization](https://arxiv.org/abs/2507.18277) | — | — |
| [Ano](math/ano.md) | arXiv 2025 | [ANO : Faster is Better in Noisy Landscape](https://arxiv.org/abs/2508.18258) | [official](https://github.com/adrienkegreisz/ano-optimizer) | — |
| [BCOS](math/bcos.md) | arXiv 2025 | [Stochastic Approximation with Block Coordinate Optimal Stepsizes](https://arxiv.org/abs/2507.08963) | [official](https://github.com/facebookresearch/bcos) | — |
| [Cautious Weight Decay](math/cautiousweightdecay.md) | arXiv 2025 | [Cautious Weight Decay](https://arxiv.org/abs/2510.12402) | [community](https://github.com/kozistr/pytorch_optimizer) | — |
| [Conda](math/conda.md) | arXiv 2025 | [Conda: Column-Normalized Adam for Training Large Language Models Faster](https://arxiv.org/abs/2509.24218) | [official](https://github.com/jie040109/Conda) | — |
| [Coupled Adam](math/coupledadam.md) | ACL 2025 | [Better Embeddings with Coupled Adam](https://arxiv.org/abs/2502.08441) | — | — |
| [DecGD](math/decgd.md) | Machine Learning 2025 | [A New Adaptive Gradient Method with Gradient Decomposition](https://arxiv.org/abs/2107.08377) | — | — |
| [DEO](math/deo.md) | arXiv 2025 | [Dimer-Enhanced Optimization: A First-Order Approach to Escaping Saddle Points in Neural Network Training](https://arxiv.org/abs/2507.19968) | [official](https://github.com/YueHuLab/DimerTrainer) | — |
| [EmoNavi](math/emonavi.md) | — | An emotion-driven optimizer that feels loss and navigates accordingly | [official](https://github.com/muooon/EmoNavi) | — |
| [MARS](math/mars.md) | ICML 2025 | [MARS: Unleashing the Power of Variance Reduction for Training Large Models](https://arxiv.org/abs/2411.10438) | [official](https://github.com/AGI-Arena/MARS) | `MARS` |
| [FOCUS](math/focus.md) | arXiv 2025 | [FOCUS: First Order Concentrated Updating Scheme](https://arxiv.org/abs/2501.12243) | [official](https://github.com/liuyz0/FOCUS) | `FOCUS` |
| [FSGDM](math/fsgdm.md) | ICLR 2025 | [On the Performance Analysis of Momentum Method: A Frequency Domain Perspective](https://arxiv.org/abs/2411.19671) | — | — |
| [Grams](math/grams.md) | ICLR Workshop 2025 | [Grams: Gradient Descent with Adaptive Momentum Scaling](https://arxiv.org/abs/2412.17107) | [official](https://github.com/Gunale0926/Grams) | `Grams` |
| [HGM](math/hgm.md) | arXiv 2025 | [Hindsight-Guided Momentum (HGM) Optimizer: An Approach to Adaptive Learning Rate](https://arxiv.org/abs/2506.22479) | — | — |
| [HVAdam](math/hvadam.md) | AAAI 2025 | [HVAdam: A Full-Dimension Adaptive Optimizer](https://arxiv.org/abs/2511.20277) | — | — |
| [KO](math/ko.md) | arXiv 2025 | [KO: Kinetics-inspired Neural Optimizer with PDE Simulation Approaches](https://arxiv.org/abs/2505.14777) | — | — |
| [KOALA++](math/koala2.md) | NeurIPS 2025 | [KOALA++: Efficient Kalman-Based Optimization with Gradient-Covariance Products](https://arxiv.org/abs/2506.04432) | — | — |
| [Kourkoutas-Beta](math/kourkoutassoftmaxflex.md) | arXiv 2025 | [Kourkoutas-Beta: A Sunspike-Driven Adam Optimizer with Desert Flair](https://arxiv.org/abs/2508.12996) | [official](https://github.com/sck-at-ucy/kbeta) | `KourkoutasSoftmaxFlex` |
| [MIAdam](math/miadam.md) | AAAI 2025 | [A Method for Enhancing Generalization of Adam by Multiple Integrations](https://arxiv.org/abs/2412.12473) | [official](https://github.com/LongJin-lab/MIAdam) | — |
| [μ²-SGD](math/sgd-v.md) | ICLR 2025 | [Do Stochastic, Feel Noiseless: Stable Stochastic Optimization via a Double Momentum Mechanism](https://arxiv.org/abs/2304.04172) | — | — |
| [⊥Grad (OrthoGrad)](math/gradorthograd.md) | ICLR 2025 | [Grokking at the Edge of Numerical Stability](https://arxiv.org/abs/2501.04697) | [official](https://github.com/LucasPrietoAl/grokking-at-the-edge-of-numerical-stability) | — |
| [Overshoot](math/overshoot.md) | arXiv 2025 | [Overshoot: Taking advantage of future gradients in momentum-based stochastic optimization](https://arxiv.org/abs/2501.09556) | [official](https://github.com/kinit-sk/overshoot) | — |
| [PadamP](math/padamp.md) | arXiv 2025 | [Adaptive Moment Estimation Optimization Algorithm Using Projection Gradient for Deep Learning](https://arxiv.org/abs/2503.10005) | — | — |
| [Simplified-AdEMAMix](math/simplifiedademamix.md) | arXiv 2025 | [Connections between Schedule-Free Optimizers, AdEMAMix, and Accelerated SGD Variants](https://arxiv.org/abs/2502.02431) | [official](https://github.com/DepenM/Simplified-AdEMAMix) | — |
| [LyAm](math/lyam.md) | arXiv 2025 | [LyAm: Robust Non-Convex Optimization for Stable Learning in Noisy Environments](https://arxiv.org/abs/2507.11262) | — | — |
| [NIRMAL](math/nirmal.md) | arXiv 2025 | [Comparative Analysis of Novel NIRMAL Optimizer Against Adam and SGD with Momentum](https://arxiv.org/abs/2508.04293) | — | — |
| [SCSAdamW](math/scsadamw.md) | arXiv 2025 | [Beyond First-Order: Training LLMs with Stochastic Conjugate Subgradients and AdamW](https://arxiv.org/abs/2507.01241) | [official](https://github.com/yhz0/scs-experiments) | — |
| [SKA-SGD](math/skasgd.md) | arXiv 2025 | [Streaming Krylov-Accelerated Stochastic Gradient Descent](https://arxiv.org/abs/2505.07046) | — | — |
| [SoftSignSGD (S3)](math/softsignsgds3.md) | arXiv 2025 | [SoftSignSGD(S3): An Enhanced Optimizer for Practical DNN Training and Loss Spikes Minimization Beyond Adam](https://arxiv.org/abs/2507.06464) | — | — |
| [SPAM](math/spam-v.md) | arXiv 2025 | [SPAM: Spike-Aware Adam with Momentum Reset for Stable LLM Training](https://arxiv.org/abs/2501.06842) | [official](https://github.com/TianjinYellow/SPAM-Optimizer) | — |
| [VSGD](math/vsgd.md) | TMLR 2025 | [Variational Stochastic Gradient Descent for Deep Neural Networks](https://arxiv.org/abs/2404.06549) | [official](https://github.com/generativeai-tue/vsgd) | — |
| [ZetA](math/zeta.md) | arXiv 2025 | [ZetA: A Riemann Zeta-Scaled Extension of Adam for Deep Learning](https://arxiv.org/abs/2508.02719) | — | — |
| [AdaGC](math/adagc.md) | ICML 2026 | [AdaGC: Enhancing LLM Pretraining Stability via Adaptive Gradient Clipping](https://arxiv.org/abs/2502.11034) | — | `AdaGC` |
| [Anon](math/anon.md) | arXiv 2026 | [Anon: Extrapolating Adaptivity Beyond SGD and Adam](https://arxiv.org/abs/2605.02317) | — | — |
| [C-Adam](math/cadam2.md) | arXiv 2026 | [A Theoretical and Experimental Study of a Novel Adaptive Learning Algorithm](https://arxiv.org/abs/2605.29273) | — | — |
| [DualAdam](math/dualadam.md) | arXiv 2026 | [Combining Adam and its Inverse Counterpart to Enhance Generalization of Deep Learning Optimizers](https://arxiv.org/abs/2603.07122) | [official](https://github.com/LongJin-lab/DualAdam) | — |
| [FANoS](math/fanos.md) | arXiv 2026 | [FANoS-v2: Feedback-Controlled Momentum with Thermostat Damping for Lightweight Neural Optimization](https://arxiv.org/abs/2601.00889) | [official](https://github.com/nalin-dhiman/fanos) | — |
| [GradPower](math/gradpower.md) | ICML 2026 | [GradPower: Powering Gradients for Faster Language Model Pre-Training](https://arxiv.org/abs/2505.24275) | — | — |
| [HomeAdam](math/homeadam.md) | arXiv 2026 | [HomeAdam: Adam and AdamW Algorithms Sometimes Go Home to Obtain Better Provable Generalization](https://arxiv.org/abs/2603.02649) | — | — |
| [NOVAK](math/novak.md) | arXiv 2026 | [NOVAK: Unified adaptive optimizer for deep neural networks](https://arxiv.org/abs/2601.07876) | — | — |
| [PS-Clip-SGD](math/psclipsgd.md) | arXiv 2026 | [Robust and Fast Training via Per-Sample Clipping](https://arxiv.org/abs/2605.02701) | — | — |
| [SparseOpt](math/sparseopt.md) | ICML 2026 | [SparseOpt: Addressing Normalization-induced Gradient Skew in Sparse Training](https://arxiv.org/abs/2605.27541) | — | — |
| [Stable-SPAM / GradientStabilizer](math/stablespamgradientstabilizer.md) | ICML 2026 | [GradientStabilizer: Fix the Norm, Not the Gradient](https://arxiv.org/abs/2502.17055) | [official](https://github.com/TianjinYellow/StableSPAM) | — |
| [VRAdam](math/vradam.md) | ICLR 2026 | [A Physics-Inspired Optimizer: Velocity Regularized Adam](https://arxiv.org/abs/2505.13196) | [official](https://github.com/pranavjv/vradam) | — |
| [SparseAdam](math/sparseadam.md) | — | Adam variant for sparse gradients | [official](https://github.com/pytorch/pytorch/blob/main/torch/optim/sparse_adam.py) | `SparseAdam` |
| [OptMuon](math/optmuon.md) | arXiv 2026 | [OptMuon: Closed-Loop Orthogonalized Momentum Methods for Stochastic Optimization with Zero-Noise Optimality](https://arxiv.org/abs/2606.08783) | — | — |
| [FOGO](math/fogo.md) | arXiv 2026 | [FOGO: Forgetting-aware Orthogonalization Optimizer](https://arxiv.org/abs/2606.10406) | — | — |
| [AdamO](math/adamo.md) | ICML 2026 | [Preserving Plasticity in Continual Learning via Dynamical Isometry](https://arxiv.org/abs/2606.09762) | — | — |
| [MAdam](math/madam.md) | arXiv 2026 | [MAdam: Metric-Aware Multi-Objective Adam](https://arxiv.org/abs/2606.03904) | — | — |
| [MuCon](math/mucon.md) | arXiv 2026 | [MuCon: Clipped Muon Updates for LLM Training](https://arxiv.org/abs/2605.26459) | — | — |
| [NuMuon](math/numuon.md) | arXiv 2026 | [NuMuon: Nuclear-Norm-Constrained Muon for Compressible LLM Training](https://arxiv.org/abs/2603.03597) | — | — |
| [MiMuon](math/mimuon.md) | arXiv 2026 | [MiMuon: Mixed Muon Optimizer with Improved Generalization for Large Models](https://arxiv.org/abs/2605.19619) | — | — |
| [Pion](math/pion.md) | arXiv preprint (cs.LG, stat.ML) 2026 | [Pion: A Spectrum-Preserving Optimizer via Orthogonal Equivalence Transformation](https://arxiv.org/abs/2605.12492) | [official](https://github.com/Sphere-AI-Lab/pion) | — |
| [iMuon (Intrinsic Muon)](math/imuonintrinsicmuon.md) | arXiv 2026 | [Intrinsic Muon: Spectral Optimization on Riemannian Matrix Manifolds](https://arxiv.org/abs/2605.09238) | [official](https://github.com/1bang118/manifold-intrinsic-muon) | — |
| [Muon-OGD](math/muonogd.md) | arXiv 2026 | [Muon-OGD: Muon-based Spectral Orthogonal Gradient Projection for LLM Continual Learning](https://arxiv.org/abs/2605.08949) | — | — |
| [Newton-Muon](math/newtonmuon.md) | arXiv 2026 | [The Newton-Muon Optimizer](https://arxiv.org/abs/2604.01472) | [official](https://github.com/zhehangdu/Newton-Muon) | — |
| [MuonEq](math/muoneq.md) | arXiv 2026 | [MuonEq: Balancing Before Orthogonalization with Lightweight Equilibration](https://arxiv.org/abs/2603.28254) | [official](https://github.com/MaeChd/muon-eq) | — |
| [RMNP](math/rmnp.md) | arXiv 2026 | [RMNP: Row-Momentum Normalized Preconditioning for Scalable Matrix-Based Optimization](https://arxiv.org/abs/2603.20527) | [official](https://github.com/Dominator-Index/RMNP) | — |
| [MUD](math/mud.md) | arXiv preprint 2026 | [Beyond Muon: MUD (MomentUm Decorrelation) for Faster Transformer Training](https://arxiv.org/abs/2603.17970) | — | — |
| [NAMO](math/namo.md) | arXiv 2026 | [Adam Improves Muon: Adaptive Moment Estimation with Orthogonalized Momentum](https://arxiv.org/abs/2602.17080) | [official](https://github.com/minxin-zhg/namo) | — |
| [SpecMuon](math/specmuon.md) | arXiv 2026 | [Muon with Spectral Guidance: Efficient Optimization for Scientific Machine Learning](https://arxiv.org/abs/2602.16167) | — | — |
| [ARO](math/aro.md) | arXiv 2026 | [ARO: A New Lens On Matrix Optimization For Large Models](https://arxiv.org/abs/2602.09006) | — | — |
| [PRISM](math/prism.md) | arXiv 2026 | [PRISM: Structured Optimization via Anisotropic Spectral Shaping](https://arxiv.org/abs/2602.03096) | — | — |
| [MCSD / SPEL](math/mcsdspel.md) | arXiv 2026 | [Manifold constrained steepest descent](https://arxiv.org/abs/2601.21487) | — | — |
| [Variance-Adaptive Muon (Muon-NSR / Muon-VS)](math/varianceadaptivemuonmuonnsrmuonvs.md) | arXiv 2026 | [Variance-Adaptive Muon: Accelerating LLM Pretraining with NSR-Modulated and Variance-Scaled Momentum](https://arxiv.org/abs/2601.14603) | — | — |
| [MuonAll](math/muonall.md) | arXiv 2025 | [MuonAll: Muon Variant for Efficient Finetuning of Large Language Models](https://arxiv.org/abs/2511.06086) | [official](https://github.com/Saurabh750/optimizer) | — |
| [Gluon](math/gluon.md) | arXiv 2025 (also accepted at ICML 2025 HiLD workshop) | [Gluon: Making Muon & Scion Great Again! (Bridging Theory and Practice of LMO-based Optimizers for LLMs)](https://arxiv.org/abs/2505.13416) | — | — |
| [LPSGD / LPSGDM](math/lpsgdlpsgdm.md) | arXiv 2026 | [Beyond L2-norm and L-infinity-norm: A Curvature-Inspired ell_p-Norm Scheme for Deep Neural Networks](https://arxiv.org/abs/2606.02078) | — | — |
| [ABSignSGD](math/absignsgd.md) | ICLR 2026 | [Arbitrary-Order Block SignSGD for Memory-Efficient LLM Fine-Tuning](https://openreview.net/forum?id=NQsdnYkCar) | — | — |
| [StoSignSGD](math/stosignsgd.md) | arXiv 2026 | [StoSignSGD: Unbiased Structural Stochasticity Fixes SignSGD for Training Large Language Models](https://arxiv.org/abs/2604.15416) | — | — |
| [Hybrid SignSGD-SGD switching](math/hybridsignsgdsgdswitching.md) | arXiv 2026 | [Enhancing SignSGD: Small-Batch Convergence Analysis and a Hybrid Switching Strategy](https://arxiv.org/abs/2604.25550) | — | — |
| [SoftSignum / SoftMuon](math/softsignumsoftmuon.md) | ICML 2026 | [Softsign: Smooth Sign in Your Optimizer For Better Parameter Heterogeneity Handling](https://arxiv.org/abs/2605.31371) | [official](https://github.com/brain-lab-research/softsign) | — |
| [Accelerated SignGD](math/acceleratedsigngd.md) | arXiv 2025 | [Norm-Constrained Flows and Sign-Based Optimization: Theory and Algorithms](https://arxiv.org/abs/2508.18510) | — | — |
| [CLion](math/clion.md) | arXiv 2026 | [CLion: Efficient Cautious Lion Optimizer with Enhanced Generalization](https://arxiv.org/abs/2604.14587) | — | — |
| [OLion](math/olion.md) | arXiv 2026 | [OLion: Approaching the Hadamard Ideal by Intersecting Spectral and ell_{infty} Implicit Biases](https://arxiv.org/abs/2602.01105) | [official](https://github.com/kv-wang/OLion) | — |
| [MGUP](math/mgup.md) | NeurIPS 2025 | [MGUP: A Momentum-Gradient Alignment Update Policy for Stochastic Optimization](https://openreview.net/forum?id=TDFSKAspoQ) | [official](https://github.com/MaeChd/MGUP) | — |
| [Magma](math/magma.md) | arXiv 2026 | [On Surprising Effectiveness of Masking Updates in Adaptive Optimizers](https://arxiv.org/abs/2602.15322) | — | — |
| [AGGC](math/aggc.md) | ACL 2026 | [AGGC: Adaptive Group Gradient Clipping for Stabilizing Large Language Model Training](https://arxiv.org/abs/2601.11864) | [official](https://github.com/ZhiyuanLi218/AGGC) | — |
| [Clipped Scion](math/clippedscion.md) | NeurIPS 2025 | [Generalized Gradient Norm Clipping & Non-Euclidean (L_0,L_1)-Smoothness](https://arxiv.org/abs/2506.01913) | [official](https://github.com/LIONS-EPFL/ClippedScion) | — |
| [SPECTRA](math/spectra.md) | ICML 2026 | [Enhancing LLM Training via Spectral Clipping](https://arxiv.org/abs/2603.14315) | [official](https://github.com/mlolab/llm-spectral-clipping) | — |
| [Spectral Clipping (matrix-valued)](math/spectralclippingmatrixvalued.md) | arXiv 2026 | [Gradient Clipping Beyond Vector Norms: A Spectral Approach for Matrix-Valued Parameters](https://arxiv.org/abs/2605.11838) | — | — |
| [SPAMP](math/spamp.md) | ACM Multimedia Asia 2025 (7th ACM International Conference on Multimedia in Asia) | [Gradient Shaping Beyond Clipping: A Functional Perspective on Update Magnitude Control](https://arxiv.org/abs/2510.01578) | — | — |
| [NucGD](math/nucgd.md) | arXiv 2026 | [Towards The Implicit Bias on Multiclass Separable Data Under Norm Constraints](https://arxiv.org/abs/2603.22824) | [official](https://github.com/Tsokarsic/observing-the-implicit-bias-on-multiclass-seperable-data) | — |
| [Batched / Transported Scion](math/batchedtransportedscion.md) | arXiv 2026 | [Scale-Invariant Neural Network Optimization: Norm Geometry and Heavy-Tailed Noise](https://arxiv.org/abs/2605.18528) | — | — |
| [EMA bias-corrected iterate averaging](math/emabiascorrectediterateaveraging.md) | NeurIPS 2025 Workshop (OPT 2025) | [EMA Without the Lag: Bias-Corrected Iterate Averaging Schemes](https://arxiv.org/abs/2508.00180) | — | — |
| [RGrad-Avg](math/rgradavg.md) | OPT 2025 (17th Annual Workshop on Optimization for Machine Learning, co-located with NeurIPS 2025) | [On Riemannian Gradient Descent Algorithm using gradient averaging](https://opt-ml.org/papers/2025/paper7.pdf) | — | — |
| [SGD with adaptive preconditioning](math/sgdwithadaptivepreconditioning.md) | ICLR 2026 | [SGD with Adaptive Preconditioning: Unified Analysis and Momentum Acceleration](https://arxiv.org/abs/2506.23803) | — | — |
| [HTMuon](math/htmuon.md) | arXiv 2026 | [HTMuon: Improving Muon via Heavy-Tailed Spectral Correction](https://arxiv.org/abs/2603.10067) | [official](https://github.com/TDCSZ327/HTmuon) | — |
| [MARS-M](math/marsm.md) | arXiv 2025 | [MARS-M: When Variance Reduction Meets Matrices](https://arxiv.org/abs/2510.21800) | [official](https://github.com/AGI-Arena/MARS/tree/main/MARS_M) | — |
| [Drop-Muon](math/dropmuon.md) | arXiv 2025 | [Drop-Muon: Update Less, Converge Faster](https://arxiv.org/abs/2510.02239) | — | — |
| [Muon+](math/muon-v.md) | arXiv 2026 | [MUON+: Towards More Effective Muon via One Additional Normalization Step for LLM Pre-training](https://arxiv.org/abs/2602.21545) | [official](https://github.com/K1seki221/MuonPlus) | — |
| [TrasMuon](math/trasmuon.md) | ICLR 2026 Workshop Sci4DL | [TrasMuon: Trust-Region Adaptive Scaling for Orthogonalized Momentum Optimizers](https://arxiv.org/abs/2602.13498) | — | — |
| [Adam-SHANG](math/adamshang.md) | arXiv 2026 | [Adam-SHANG: A Convergent Adam-Type Method for Stochastic Smooth Convex Optimization](https://arxiv.org/abs/2605.12878) | — | — |
| [EMA-Nesterov](math/emanesterov.md) | arXiv 2026 | [EMA-Nesterov: Stabilizing Nesterov's Lookahead for Accelerated Deep Learning Optimization](https://arxiv.org/abs/2605.25395) | — | — |
| [S-Adam](math/sadam2.md) | arXiv 2026 | [Singularity-aware Optimization via Randomized Geometric Probing: Towards Stable Non-smooth Optimization](https://arxiv.org/abs/2605.29547) | — | — |
| [IAdaPID-ADG](math/iadapidadg.md) | arXiv 2026 | [An Improved Adaptive PID Optimizer with Enhanced Convergence and Stability for Deep Learning](https://arxiv.org/abs/2605.21968) | — | — |
| [CT-AGD](math/ctagd.md) | arXiv 2026 | [Accelerated Gradient Descent for Faster Convergence with Minimal Overhead](https://arxiv.org/abs/2605.16017) | — | — |
| [GPA (Generalized Primal Averaging)](math/gpageneralizedprimalaveraging.md) | arXiv 2025 | [Smoothing DiLoCo with Primal Averaging for Faster Training of LLMs](https://arxiv.org/abs/2512.17131) | [official](https://github.com/facebookresearch/optimizers) | — |
| [SNOO](math/snoo.md) | arXiv 2025 | [SNOO: Step-K Nesterov Outer Optimizer - The Surprising Effectiveness of Nesterov Momentum Applied to Pseudo-Gradients](https://arxiv.org/abs/2510.15830) | [official](https://github.com/vishal9-team/torchtitan-snoo) | — |
| [Riemannion](math/riemannion.md) | ICLR 2026 | [LoRA meets Riemannion: Muon Optimizer for Parametrization-independent Low-Rank Adapters](https://arxiv.org/abs/2507.12142) | — | — |
| [Optimal Projection-Free Adaptive SGD](math/optimalprojectionfreeadaptivesgd.md) | arXiv 2026 | [Optimal Projection-Free Adaptive SGD for Matrix Optimization](https://arxiv.org/abs/2604.02505) | — | — |
| [AdamCB](math/adamcb.md) | ICLR 2025 | [ADAM Optimization with Adaptive Batch Selection](https://arxiv.org/abs/2512.06795) | — | — |
| [Kalman-Adam](math/kalmanadam.md) | Knowledge-Based Systems 2026 | [Kalman-Adam: Optimal bayesian moment estimation for memory-Efficient and generalizable deep learning](https://doi.org/10.1016/j.knosys.2026.115907) | — | — |
| [AdamHD (AdamHuberDecay)](math/adamhdadamhuberdecay.md) | NeurIPS 2025 Workshop (ScaleOpt: GPU-Accelerated and Scalable Optimization) | [AdamHD: Decoupled Huber Decay Regularization for Language Model Pre-Training](https://arxiv.org/abs/2511.14721) | — | — |
| [MVN-Grad](math/mvngrad.md) | arXiv 2026 | [Adaptive Optimization via Momentum on Variance-Normalized Gradients](https://arxiv.org/abs/2602.10204) | — | — |
| [Compositional Muon (CM)](math/compositionalmuoncm.md) | Tilde Research blog 2026 | [Towards Compositional Steepest Descent](https://blog.tilderesearch.com/blog/compositional-muon) | [official](https://github.com/tilde-research/comp-muon-release) | — |
