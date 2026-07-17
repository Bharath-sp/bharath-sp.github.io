---
layout: post
title: Wasserstein Convergence of SGMs
categories: [Deep Generative Models]
toc: true
---

Can we establish upper convergence bounds for Score-based Generative Models under Semiconvexity and Discontinuous Gradient conditions? The paper titled ["Wasserstein Convergence of Score-based Generative Models under Semiconvexity and Discontinuous Gradients" by Bruno and Sabanis (2025)](https://arxiv.org/abs/2505.03432){:target="_blank"} addresses this question. This article provides the detailed derivation of one of the results from this paper.

* TOC
{:toc}

## Abstract
Most convergence proofs in Score-based Generative Models (SGMs) such as de-noising score matching and diffusion models assume ideal settings:

* The energy function $f$ of the target distribution is $\mu$- strongly convex.
* The score function (the gradient of the log density) is Lipschitz continuous.

The Langevin diffusion algorithm converges to the target $p^*$ for any distribution, but the rate of convergence happens at an acceptable rate when the energy function $f$ of the target distribution is $\mu$- strongly convex. Such conditions are rarely satisfied in practice, and doesn't truly reflect the complexity of the real data.

In this paper, the authors have relaxed these assumptions and have established convergence guarantees for SGMs targeting **semi-convex distributions with potentially discontinuous gradients** in terms of the Wasserstein-2 distance. They have established explicit, dimension and parameter dependent, non-asymptotic Wasserstein-2 convergence **upper bounds** under these relaxed assumptions.

## Background
The core idea behind diffusion models is:

**Forward process:**

We are given samples from some distribution $p^\*$. We add noise to these samples until its distribution becomes pure Gaussian noise.

Let $(X_t)$ be the forward process where $t\in [0,T]$ and $X_0 \sim p^\*$. The Langevin diffusion process with the source distribution $p^*$ and the target the pure Gaussian noise $\mathcal{N}(0, I)$ is characterized by:

$$
dX_t = -X_t \, dt + \sqrt{2}\, dW_t
$$

where $W_t$ is the Weiner process.

**Reverse Process:**

We start with noise and transform it back to the target distribution by estimating the score function of the forward process.

Let $(Y_t)$ be the reverse process where $t\in [0,T]$. And it should be

$$
(Y_t)_{t \in [0,T]} = (X_{T-t})_{t \in [0,T]}
$$

The reverse process can also be characterized by a SDE:

$$
dY_t = (Y_t + 2 \nabla \log p_{T-t}(Y_t)) \, dt + \sqrt{2} \, d \bar{W}_t
$$

where $p_t = p(X_t)$.

Ideally $Y_0 \sim p(X_T)$. But in practice, the initial distribution is taken to be the standard Gaussian distribution, $\mathcal{N}(0, I)$. As a result, the backward process becomes:

$$
d\tilde{Y}_t = (\tilde{Y}_t + 2 \nabla \log p_{T-t}(\tilde{Y}_t)) \, dt + \sqrt{2} \, d \bar{W}_t   \hspace{1cm} \text{where } \tilde{Y}_0 \sim \mathcal{N}(0,I)
$$

Since the target distribution $p^*$ is unknown, the score function $\nabla \log p_t$ cannot be computed exactly. We learn an estimator $s$ that approximates the score of the forward process over a fixed time window $[0,T]$. On incorporating this approximation, the process is:

$$
dY^{\text{aux}}_t = (Y^{\text{aux}}_t + 2 s_{\theta^*}(T-t, Y^{\text{aux}}_t)) \, dt + \sqrt{2} \, d \bar{W}_t  \hspace{1cm} \text{where } Y^{\text{aux}}_0 \sim \mathcal{N}(0,I)
$$

In practice, we discretize the process. Let $\gamma  \in (0,1)$ be the step size for time steps $j=1, \dots, J$. Then the EM approximation of the reverse process is:

$$
Y^{\text{EM}}_{j+1} = Y^{\text{EM}}_j  + \gamma (Y^{\text{EM}}_j + 2 s_{\theta^*}(T-t, Y^{\text{EM}}_j)) \, dt + \sqrt{2\gamma} \, N_j  \hspace{1cm} \text{where } Y^{\text{EM}}_0 \sim \mathcal{N}(0,I)
$$


<figure markdown="0" class="figure zoomable">
<img src="{{'/assets/images/deep_generative_models/diffusion_processes.png' | relative_url}}" alt="Forward and backward process of Langevin Diffusion. The deviation between the final distribution of the EM approximation we reach through the backward process and the target distribution is broken  down into four components." width="50%"><figcaption>
  <strong>Figure 1.</strong> Forward and backward process of Langevin Diffusion. The deviation between the final distribution of the EM approximation we reach through the backward process and the target distribution is broken  down into four components.
  </figcaption>
</figure>

This breaks down the deviation between the final distribution of the EM approximation and the target distribution into four components. From the bottom of the figure,

* The first component is the deviation between the EM approximation and the auxiliary process, which is due to the discretization error.
* The second component is the deviation between the auxiliary process and the backward process, which is due to the score estimation error.
* The third component is the deviation between the backward process that starts from the standard Gaussian and the backward process that starts from the final distribution of the forward process. This is due to the difference in their initial distributions.
* The fourth component is the deviation due to estimation error and finite number of steps in the backward process in practice.

## Wasserstein Convergence Analysis of SGMs

Since we are measuring the deviation in terms of a distance, then using the triangle inequality, we can bound the deviation as follows:

$$
\begin{align*}
W_2(p^\text{EM}_J, p^*) \leq & W_2\left(p_J^\text{EM}, p_T^\text{aux})\right) \\
& + W_2\left(p_T^\text{aux}, \tilde{p}_T\right) \\
& + W_2\left(\tilde{p}_T, q_T\right) \\
& + W_2\left(q_T, p^*\right) \\
\end{align*}
$$


### Upper bound on $W_2\left(q_T, \tilde{p}_T\right)$:

Here we will derive an upper bound on the Wasserstein-2 distance between two distributions at time $T$: the distribution of the backward process that starts from the final distribution of the forward process $q_T$ and the distribution of the backward process that starts from the standard Gaussian $\tilde{p}_T$.

We know that from the definition of the Wasserstein-2 distance:

$$
W_2\left(q_T, \tilde{p}_T\right) \leq \sqrt{\mathbb{E}\left[\|Y_T - \tilde{Y}_T\|^2\right] }
$$

There are two diffusion processes $\tilde{Y}_t$ and $Y_t$. These processes have the same Brownian motion but different starting points.

* $Y_t$ starts from the final distribution of the forward process $p_T$.
* $\tilde{Y}_t$ starts from the invariant distribution of the forward process (the standard Gaussian)

These processes can be characterized by SDE respectively as follows:

$$
\begin{align*}
dY_t & = (Y_t + 2 \nabla \log p_{T-t}(Y_t)) \, dt + \sqrt{2} \, d \bar{W}_t \\
d\tilde{Y}_t & = (\tilde{Y}_t + 2 \nabla \log p_{T-t}(\tilde{Y}_t)) \, dt + \sqrt{2} \, d \bar{W}_t
\end{align*}
$$

Define a difference process

<div class="scroll-equation" id="eq:eq1">

$$
\begin{equation}
Z_t = Y_t - \tilde{Y}_t \hspace{1cm} \forall t \in [0, T]
\end{equation}
$$

</div>

Then,

<div class="scroll-equation" id="eq:eq2">

$$
\begin{align}
dZ_t & = dY_t - d\tilde{Y}_t \nonumber \\
& = [(Y_t - \tilde{Y}_t) + 2(\nabla \log p_{T-t}(Y_t) - \nabla \log p_{T-t}(\tilde{Y}_t))] dt\\
\end{align}
$$

</div>

Since both processes share the same noise term $\sqrt{2} \, d\bar{W}_t$, the noise cancels out:

Let $f(Z_t) = \|Z_t\|^2 = \sum_i Z_{i,t}^2$ be a function of $Z_t$. Then, the change in $f$ with respect to $Z_t$ is given by

$$
\begin{align*}
\nabla_{Z_t} f(Z_t) & = 2Z_t \\
df(Z_t) & = 2 Z_t^\top dZ_t = 2 \langle Z_t, dZ_t \rangle
\end{align*}
$$


On substituting <a href="#eq:eq1">(1)</a> and <a href="#eq:eq2">(2)</a> and using the linearity of the inner product, i.e., $\langle a, c \cdot \mathbf{v} \rangle = c \langle a, \mathbf{v} \rangle$, 


$$
d\|Z_t\|^2 = 2 \langle Y_t - \tilde{Y}_t,\,\, (Y_t - \tilde{Y}_t) + 2(\nabla \log p_{T-t}(Y_t) - \nabla \log p_{T-t}(\tilde{Y}_t)) \rangle \, dt 
$$

for any $t \in [0, T]$. Again, using the linearity property:

$$
\langle \mathbf{w} , \mathbf{u} + \beta \cdot \mathbf{v} \rangle = \langle \mathbf{w}, \mathbf{u} \rangle + \beta \langle \mathbf{w}, \mathbf{v} \rangle
$$

We get

$$
\begin{align*}
d\|Z_t\|^2 & = 2 \left[  \langle Y_t - \tilde{Y}_t, Y_t - \tilde{Y}_t \rangle + 2\langle Y_t - \tilde{Y}_t, \,\, \nabla \log p_{T-t}(Y_t) - \nabla \log p_{T-t}(\tilde{Y}_t) \rangle \, \right] dt \\
& = 2 \|Y_t - \tilde{Y}_t \|^2\, dt + 4\langle Y_t - \tilde{Y}_t, \,\, \nabla \log p_{T-t}(Y_t) - \nabla \log p_{T-t}(\tilde{Y}_t) \rangle \, dt \\
\end{align*}
$$

On integrating over time $t$ from 0 to $T$,

$$
\|Z_T\|^2 = \|Z_0\|^2 + \int_0^T 2\|Y_t - \tilde{Y}_t \|^2\, dt + \int_0^T 4\langle Y_t - \tilde{Y}_t, \,\, \nabla \log p_{T-t}(Y_t) - \nabla \log p_{T-t}(\tilde{Y}_t) \rangle \, dt
$$

On taking expectation on both the sides and interchanging the expectation and integral operators:

<div class="scroll-equation" id="eq:eq3">

$$
\begin{align}
\mathbb{E}\left[\|Y_T - \tilde{Y}_T\|^2\right] &= \mathbb{E}\left[\|Y_0 - \tilde{Y}_0\|^2 \right] \nonumber \\
& + \int_0^T 2 \mathbb{E}\left[\|Y_t - \tilde{Y}_t \|^2\, \right] dt \nonumber \\
& + \int_0^T 4 \mathbb{E}\left[ \langle Y_t - \tilde{Y}_t, \,\, \nabla \log p_{T-t}(Y_t) - \nabla \log p_{T-t}(\tilde{Y}_t) \rangle \, \right] dt
\end{align}
$$

</div>

From Cauchy-Schwartz inequality, we know $\langle a, b \rangle \leq \|a\| \|b\|$. Let 

* $a=Y_t - \tilde{Y}_t$ and 
* $b = \nabla \log p_{T-t}(Y_t) - \nabla \log p_{T-t}(\tilde{Y}_t)$

Then,

<div class="scroll-equation" id="eq:eq4">

$$
\begin{align}
\langle Y_t - \tilde{Y}_t, \,\, & \nabla \log p_{T-t}(Y_t) - \nabla \log p_{T-t}(\tilde{Y}_t) \rangle \leq \nonumber \\
 & \|Y_t - \tilde{Y}_t\| \cdot \| \nabla \log p_{T-t}(Y_t) - \nabla \log p_{T-t}(\tilde{Y}_t) \|
\end{align}
$$

</div>

**Smooth Potential:**

If the energy function of the target distribution $\mathcal{E}$ was $L$-smooth, then its gradient is Lipschitz continuous with constant $L$:

$$
\| \nabla \mathcal{E}(x) - \nabla \mathcal{E}(y) \| \leq L \|x - y\|  \hspace{1cm} \forall x, y
$$

Using this, <a href="#eq:eq4">(4)</a> can be written as:

$$
\begin{align*}
\langle Y_t - \tilde{Y}_t, \,\, \nabla \log p_{T-t}(Y_t) - \nabla \log p_{T-t}(\tilde{Y}_t) \rangle & \leq \|Y_t - \tilde{Y}_t\| \cdot L \, \| Y_t - \tilde{Y}_t\| \\
& \leq L \, \| Y_t - \tilde{Y}_t\|^2
\end{align*}
$$

Then <a href="#eq:eq3">(3)</a> becomes

$$
\begin{align*}
\mathbb{E}\left[\|Y_T - \tilde{Y}_T\|^2\right] & \leq \mathbb{E}\left[\|Y_0 - \tilde{Y}_0\|^2 \right] + \int_0^T 2 \mathbb{E}\left[\|Y_t - \tilde{Y}_t \|^2\, \right] dt + \int_0^T 4L \mathbb{E}\left[\| Y_t - \tilde{Y}_t\|^2 \, \right] dt \\
& = \mathbb{E}\left[\|Y_0 - \tilde{Y}_0\|^2 \right] +  \int_0^T (2+4L) \mathbb{E}\left[\| Y_t - \tilde{Y}_t\|^2 \, \right] dt \\
\end{align*}
$$

On applying the Grönwall's Inequality, we get:

$$
\begin{align*}
\mathbb{E}\left[\|Y_T - \tilde{Y}_T\|^2\right] & \leq \mathbb{E}\left[\|Y_0 - \tilde{Y}_0\|^2 \right] \, e^{(2+4L)T}
\end{align*}
$$

**Non-smooth potential:**

<figure markdown="0" class="figure zoomable">
<img src="{{'/assets/images/deep_generative_models/semi_convex_corollary.png' | relative_url}}" alt="Corollary 14 from the paper" width="100%"><figcaption>
  <strong>Figure 2.</strong> Corollary 14 from the paper
  </figcaption>
</figure>

It essentially says that even if the original energy is rough, as we add noise in the forward process, the distribution $p_t$ becomes smooth over time.

* At the very start of the process ($t \approx 0$), the smoothness is governed by the underlying semi-convexity of the potential.
* As $t \to \infty $, the coefficient $\beta_t^{OS}$ would then converge to $\mu$, which represents the $\mu$-strong convexity of the distribution at infinity.

This indicates that over time, the "semi-convexity" of the original energy is completely washed away by the noise, leaving a smooth, strongly convex landscape. Using this, the third term in <a href="#eq:eq3">(3)</a> can be written as:


$$
\langle Y_t - \tilde{Y}_t, \,\, \nabla \log p_{T-t}(Y_t) - \nabla \log p_{T-t}(\tilde{Y}_t) \rangle \leq -\beta_{T-t}^{\text{OS}} \|Y_t - \tilde{Y}_t \|^2
$$

Then <a href="#eq:eq3">(3)</a> becomes:

$$
\begin{align*}
\mathbb{E}\left[\|Y_T - \tilde{Y}_T\|^2\right] & \leq \mathbb{E}\left[\|Y_0 - \tilde{Y}_0\|^2 \right] + \int_0^T 2 \mathbb{E}\left[\|Y_t - \tilde{Y}_t \|^2\, \right] dt - 4 \int_0^T \beta_{T-t}^{\text{OS}}  \mathbb{E}\left[ \|Y_t - \tilde{Y}_t \|^2\right] dt \\
& = \mathbb{E}\left[\|Y_0 - \tilde{Y}_0\|^2 \right] + \int_0^T (2-4\beta_{T-t}^{\text{OS}}) \, \mathbb{E}\left[\|Y_t - \tilde{Y}_t \|^2\, \right] dt 
\end{align*}
$$

On applying the Grönwall's Inequality, we get:

$$
\begin{align*}
\mathbb{E}\left[\|Y_T - \tilde{Y}_T\|^2\right] & \leq \mathbb{E}\left[\|Y_0 - \tilde{Y}_0\|^2 \right] \, e^{\int_0^T (2-4\beta_{T-t}^{\text{OS}}) dt } \\
& = \mathbb{E}\left[\|Y_0 - \tilde{Y}_0\|^2 \right] \, e^{2 [T - 2\int_0^T \beta_{T-t}^{\text{OS}} dt] } \\
\end{align*}
$$

We know the forward process SDE is:

$$
dX_t = -X_t dt + \sqrt{2}\, dW_t  \hspace{1cm} \text{where } X_0 \sim p^*
$$

On solving this SDE, we get:

$$
X_t = m_t X_0 + \sigma_t \, N_t  \hspace{1cm} \text{ for } t\in [0, T]
$$

where $m_t = e^{-t}$, $\sigma_t^2 = 1- e^{-2t}$, and $N_t \sim \mathcal{N}(0, I)$. At $t=T$,

$$
X_T = m_T X_0 + \sigma_T \, N_T
$$

We know that $X_T = Y_0$ and $N_T \stackrel{d}{=} \tilde{Y}_0$. On substituting, 

$$
\begin{align*}
\mathbb{E}\left[\|Y_T - \tilde{Y}_T\|^2\right] 
& \leq \mathbb{E}\left[\|m_T X_0 + \sigma_T \, N_T - N_T\|^2 \right] \, e^{2 [T - 2\int_0^T \beta_{T-t}^{\text{OS}} dt] } \\
& =\mathbb{E}\left[\|m_T X_0 + (\sigma_T \, -1) N_T\|^2 \right] \, e^{2 [T - 2\int_0^T \beta_{T-t}^{\text{OS}} dt] } \\
\end{align*}
$$

Using the inequality $(a + b)^2 \leq 2a^2 + 2b^2$,

$$
\begin{align*}
\mathbb{E}[\|m_T X_0 + (\sigma_T - 1)\tilde{Y}_0\|^2] & \leq 2 \mathbb{E}[\|m_T X_0\|^2] + 2 \mathbb{E}[\|(\sigma_T - 1)\tilde{Y}_0\|^2] \\
& \leq 2 \left( m_T^2 \, \mathbb{E}[\|X_0\|^2] + (\sigma_T - 1)^2 \, \mathbb{E}[\|\tilde{Y}_0\|^2) \right)
\end{align*}
$$

We know $m_T^2 = e^{-2T}$ and $(\sigma_T - 1)^2 \leq e^{-2T}$, then our final bound can be given as:

$$
\begin{align*}
\mathbb{E}\left[\|Y_T - \tilde{Y}_T\|^2\right] 
& \leq 2 \left(\mathbb{E}[\|X_0\|^2] +   d \right) \, e^{2 [T - 2\int_0^T \beta_{T-t}^{\text{OS}} dt] -2T} \\
\end{align*}
$$

where $\mathbb{E}[\|\tilde{Y}_0\|^2]$ is equal to the trace of the identity covariance matrix, which is exactly $d$.

$$
\begin{align*}
W_2\left(q_T, \tilde{p}_T\right) & \leq \sqrt{\mathbb{E}\left[\|Y_T - \tilde{Y}_T\|^2\right] } \\
& \leq \sqrt{2} \left( \sqrt{\mathbb{E}[\|X_0\|^2}] +   \sqrt{d} \right) \, e^{2 [T - 2\int_0^T \beta_{T-t}^{\text{OS}} dt - T]} \\
& \leq \sqrt{2} \left( \sqrt{\mathbb{E}[\|X_0\|^2}] +   \sqrt{d} \right) \, e^{- 2\int_0^T \beta_{T-t}^{\text{OS}} dt} 
\end{align*}
$$

Which is of the order $O(\sqrt{d})$. The distance grows sub-linearly with the number of features or dimensions ($d$).

The estimates are explicit and exhibit the best known optimal dependencies in terms of data dimension, i.e., $O(\sqrt{d})$ in Theorem 19, and rate of convergence, i.e., $O(\gamma)$ in Theorem 21.

## Final Result
For any $\delta > 0$, if we choose the parameters $\epsilon, T, \varepsilon_{SN}, \gamma$ within the range (where the values of $\epsilon_\delta, T_{\delta}, \varepsilon_{SN, \delta}, \gamma_\delta$ are given in the table 3 of the paper)

* $0 < \epsilon < \epsilon_\delta$
* $T > T_{\delta}$
* $ 0 < \varepsilon_{SN} <  \varepsilon_{SN, \delta}$
* $0 < \gamma < \gamma_\delta$

It is shown that 

$$
W_2(p^\text{EM}_J, p^*) < \delta
$$

## References

1. Bruno, S., & Sabanis, S. (2025, May 6). Wasserstein Convergence of Score-based Generative Models under Semiconvexity and Discontinuous Gradients. arXiv.org. [https://arxiv.org/abs/2505.03432](https://arxiv.org/abs/2505.03432){:target="_blank"}
