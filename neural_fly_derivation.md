# Neural-Fly Enables Rapid Learning
for Agile Flight in Strong Winds

Book/Paper: Control
Field: Control
Contains Formulas: No
Contains Images: No

# 1 Euler–Lagrange dynamics with aerodynamic force + error-based control equation

## 1.1 Full dynamics (including aerodynamic force)

We consider an Euler–Lagrange (EL) mechanical system with generalized coordinates and velocities:

$$
q(t)\in\mathbb{R}^n,\qquad \dot q(t)\in\mathbb{R}^n.
$$

The dynamics (with aerodynamic / disturbance force) are:

$$
M(q)\,\ddot q + C(q,\dot q)\,\dot q + g(q) = u + f(q,\dot q,w),
$$

where:

$M(q)\in\mathbb{R}^{n\times n}\ \text{is symmetric positive definite},\quad
C(q,\dot q)\in\mathbb{R}^{n\times n},\quad
g(q)\in\mathbb{R}^{n}.$

The unknown aerodynamic force is denoted:

$$
f(q,\dot q,w)\in\mathbb{R}^{n},
$$

Key EL structural identity

A classical property of EL systems is that: $\dot M(q) - 2C(q,\dot q)\ \text{is skew-symmetric}$

which implies for any vector $x\in\mathbb{R}^n$

$x^\top\left(\dot M - 2C\right)x = 0
\quad\Longleftrightarrow\quad
x^\top \dot M x = 2x^\top C x.$

We will use this to simplify the Lyapunov derivative later.

## 1.2 Tracking errors and the “composite” error variable $s$

Let $q_d(t)$ be a desired trajectory (at least twice differentiable). Define tracking errors:

$$
\tilde q := q - q_d,\qquad \dot{\tilde q} := \dot q - \dot q_d.
$$

Choose a constant gain matrix:

$$
\Lambda \in \mathbb{R}^{n\times n},\quad \Lambda \succ 0.
$$

Define the “composite” tracking error:

$$
s := \dot{\tilde q} + \Lambda \tilde q.
$$

Define the reference velocity/acceleration (standard in EL control):

$$
\dot q_r := \dot q_d - \Lambda \tilde q,\qquad\ddot q_r := \ddot q_d - \Lambda \dot{\tilde q}.
$$

Then you can verify:

$$
s = \dot q - \dot q_r,\qquad\dot s = \ddot q - \ddot q_r.
$$

## 1.3 Aerodynamic force model, estimation, and why parameter-error divergence is dangerous

We assume the unknown aerodynamic force admits a **low-dimensional linear-in-parameters representation** plus a residual:

$$
f(q,\dot q,w) = \phi(q,\dot q)\,a(t) + d(t).
$$

- $\phi(q,\dot q)\in\mathbb{R}^{n\times h}:$ known “basis/features” (e.g., a neural network output evaluated online)

       $a(t)\in\mathbb{R}^{h}$ ：unknown time-varying coefficient (captures wind-dependent effects)

       $d(t)\in\mathbb{R}^{n}$ :   unmodeled remainder (representation error, unmodeled physics, etc.)

We maintain an online estimate $\hat a(t)$, Define parameter estimation error

$$
\tilde a = \hat a - a.
$$

Then the estimated aerodynamic force is:

$$
\hat f = \phi(q,\dot q)\,\hat a.
$$

The **force estimation error** becomes:

$$
f - \hat f = \phi a + d - \phi \hat a = -\phi \tilde a + d.
$$

Why $\tilde a$ can destabilize the system

If $\tilde a(t)$  is not guaranteed bounded, then the term$-\phi\tilde a$  can grow without bound and acts like an **unbounded input**

injected into the closed-loop error dynamics. Even if we add stabilizing feedback $−Ks$, an unbounded forcing term can dominate and drive $s$ large, breaking tracking (and potentially saturating actuators in practice).

So:**stability requires controlling both $s$ and $\tilde a$**

## 1.4 Control law and the closed-loop error equation in $s$

Choose a feedback gain:

$$
K\in\mathbb{R}^{n\times n},\quad K\succ 0.
$$

Define the control input:

$$
u = M(q)\,\ddot q_r + C(q,\dot q)\,\dot q_r + g(q)\ -\ K s\ -\ \phi(q,\dot q)\,\hat a.
$$

Now derive the $s$ dynamics.

Start from the plant:

$$
M\ddot q + C\dot q + g = u + f.
$$

Substitute  $u$ and cancel terms:

$$
M\ddot q + C\dot q + g=\left(M\ddot q_r + C\dot q_r + g - Ks - \phi\hat a\right) + f.
$$

Bring $M\ddot q_r + C\dot q_r + g$ to the left:

$$
M(\ddot q - \ddot q_r) + C(\dot q - \dot q_r) = -Ks - \phi\hat a + f.
$$

Use:

$$
\dot s = \ddot q - \ddot q_r,\qquad s = \dot q - \dot q_r,
$$

to obtain:

$$
M\dot s + Cs = -Ks - \phi\hat a + f.
$$

Replace $f=\phi a + d$ , and group $\tilde a=\hat a-a$

$$
M\dot s + Cs = -Ks - \phi(\hat a - a) + d
$$

$$
M\dot s + (C+K)s = -\phi \tilde a + d
$$

This is the key “tracking-error dynamics with parameter error injection.

# 2.Why we introduce a joint Lyapunov function + exponential contraction to an error ball

## 2.1 If we only use tracking energy, we need $\tilde a$  bounded (not automatic)

A natural tracking-only Lyapunov candidate is:

$$
V_s := \frac12 s^\top M(q)\,s.
$$

Its derivative is:

$$
\dot V_s = s^\top M\dot s + \frac12 s^\top \dot M s.
$$

Using the closed-loop equation:

$$
M\dot s = -(C+K)s - \phi\tilde a + d,
$$

we get:

$$
\dot V_s=s^\top\left(-(C+K)s - \phi\tilde a + d\right) + \frac12 s^\top \dot M s=- s^\top K s\ +\ s^\top(d-\phi\tilde a)\ +\ \left(-s^\top C s+\frac12 s^\top \dot M s\right).
$$

Using EL identity:

$$
-s^\top C s+\frac12 s^\top \dot M s = 0,
$$

$$
\dot V_s = - s^\top K s + s^\top(d-\phi\tilde a).
$$

Bounding:

$$
s^\top K s \ge \lambda_{\min}(K)\|s\|^2,
$$

$$
\qquad
s^\top(d-\phi\tilde a)\le \|s\|\,\|d-\phi\tilde a\|.
$$

$$
\dot V_s \le -\lambda_{\min}(K)\|s\|^2 + \|s\|\,\|d-\phi\tilde a\|.
$$

This implies “ultimate boundedness” **only if** $\|d-\phi\tilde a\|$ is bounded—i.e., you need $\tilde a$ bounded. But **boundedness of $\tilde a$** is *not automatic* unless we design an update law for $\hat a$ that enforces it.

Hence we introduce a **joint** Lyapunov function including $\tilde a$

## 2.2 Joint Lyapunov function (tracking + parameter-error energy)

We choose:

$$
V := \frac12 s^\top M s\ +\ \frac12 \tilde a^\top P^{-1}\tilde a,
$$

where:

$$
P(t)\in\mathbb{R}^{h\times h},\quad P(t)\succ 0
$$

is a time-varying positive definite matrix (later we will define its dynamics).

This is “equivalent energy” because:

- $s^\top M s$ is kinetic-like energy in the tracking error
- $\tilde a^\top P^{-1}\tilde a$  measures parameter error magnitude under the metric $P^{-1}$

## 2.3 Derivative of the joint Lyapunov function and the key coupling term

Compute:

$$
\dot V = \dot V_s + \dot V_a,
$$

where $\dot V_s$ is already known:

$$
\dot V_s = -s^\top K s + s^\top(d-\phi\tilde a).
$$

Now handle:

$$
V_a := \frac12 \tilde a^\top P^{-1}\tilde a.
$$

Differentiate (product rule; both $\tilde a$ and $P^{-1}$ vary):

$$
\dot V_a=\tilde a^\top P^{-1}\dot{\tilde a}+\frac12 \tilde a^\top \dot{(P^{-1})}\tilde a.
$$

$$
\dot V=-s^\top K s+s^\top d-s^\top \phi\tilde a+\tilde a^\top P^{-1}\dot{\tilde a}+\frac12 \tilde a^\top \dot{(P^{-1})}\tilde a.
$$

The problematic coupling term is:

$$
- s^\top \phi\tilde a.
$$

To make $\dot V$negative (up to bounded disturbance terms), we want **an opposite term** $+\ s^\top\phi\tilde a$  appear from $\tilde a^\top P^{-1}\dot{\tilde a}$

This motivates including a **tracking-driven** component in $\dot{\hat a}$  such that:

$$
\dot{\tilde a} \supset P\phi^\top s.
$$

Because then:

$$
\tilde a^\top P^{-1}(P\phi^\top s) = \tilde a^\top \phi^\top s = s^\top\phi\tilde a,
$$

which cancels the dangerous coupling exactly.

This is the core “why tracking term is necessary” from the Lyapunov viewpoint.

## 2.4 Resulting contraction to an error ball (high-level inequality form)

After designing $\dot{\hat a}$ and $\dot P$ appropriately (next section),˙$\dot V$ can be arranged into the form:

$$
\dot V \le -\alpha V + \beta,
$$

with constants $\alpha>0$ and $\beta\ge 0$ depending on:

- $K$ (damping)
- boundedness of $d(t)$, measurement noise, and time-variation of $a(t)$

Solving the scalar differential inequality yields:

$$
V(t) \le e^{-\alpha t}V(0) + \frac{\beta}{\alpha}\left(1-e^{-\alpha t}\right),
$$

hence:

$$
\limsup_{t\to\infty} V(t) \le \frac{\beta}{\alpha},
$$

which implies$[s;\tilde a]$ converges exponentially into a ball whose radius is proportional to $\sqrt{\beta/\alpha}$,This is the precise meaning of “equivalent energy shrinks to an error ball.”

# 3. Optimal parameter estimation: obtain $\hat a$, $\dot P$ and the gain (Kalman–Bucy structure)

## 3.1 Measurement and process models for the unknown coefficient $a(t)$

We assume we can form a (noisy) measurement $y(t)\in\mathbb{R}^n$ of the aerodynamic force component represented by $\phi a$

$$
y(t) = \phi(q,\dot q)\,a(t) + \epsilon(t),
$$

where $\epsilon(t)$ is measurement noise with covariance:

$$
\mathbb{E}[\epsilon(t)\epsilon(\tau)^\top] = R\,\delta(t-\tau),\qquad R\succ 0.
$$

We also model the time-variation of $a(t)$ as a stable drift + process noise:

$$
\dot a(t) = -\lambda a(t) + \nu(t),
$$

with:

$$
\mathbb{E}[\nu(t)\nu(\tau)^\top] = Q\,\delta(t-\tau),\qquad Q\succeq 0.
$$

Here:

- $\lambda\ge 0$ enforces “forgetting / leakage”
- $Q$  encodes how fast we allow $a(t)$ to vary (wind variability)

## 3.2 Kalman–Bucy (continuous-time) estimator for $a(t)$

### i. Stochastic formulation of the parameter–observation system

We consider an unknown scalar parameter $a(t)$ evolving in time and observed indirectly through noisy measurements.The deterministic rate equation

$$
\dot a(t) = -\lambda a(t) + \nu(t)
$$

is interpreted as a stochastic differential equation by modeling the uncertainty term $\nu(t)$ as white noise.Specifically, we write the parameter dynamics in Itô form as

$$
\mathrm{d}a(t) = -\lambda a(t)\mathrm{d}t + \sqrt{Q}\mathrm{d}W_t
$$

where $W_t$ is a standard Brownian motion satisfying

$$
\mathbb{E}[\mathrm{d}W_t] = 0,\qquad \mathbb{E}[(\mathrm{d}W_t)^2] = \mathrm{d}t
$$

and $Q\ge 0$ represents the process noise intensity.

The observation model is given in algebraic form as

$$
y(t) = \phi(t)a(t) + \epsilon(t)
$$

where  $\epsilon(t)$ is zero-mean measurement noise with variance $R>0.$In continuous time, this is expressed in differential form as

$$
\mathrm{d}y(t) = \phi(t)a(t)\mathrm{d}t + \sqrt{R}\mathrm{d}V_t
$$

where $V_t$ is an independent standard Brownian motion with

$$
\mathbb{E}[\mathrm{d}V_t] = 0,\qquad \mathbb{E}[(\mathrm{d}V_t)^2] = \mathrm{d}t
$$

We assume $W_t$ and $V_t$  are independent.

### ii Gaussianity of the stochastic processes

We assume a Gaussian prior for the initial parameter value,

$$
a(0)\sim\mathcal{N}(\bar a_0,P_0).
$$

Since the SDE for $a(t)$ is linear and driven by Gaussian noise, its explicit solution can be written as

$$
a(t) = \mathrm{e}^{-\lambda t}a(0) + \int_0^t \mathrm{e}^{-\lambda (t-s)}\sqrt{Q}\mathrm{d}W_s
$$

Both terms on the right-hand side are linear functionals of Gaussian random variables; therefore $a(t)$  is Gaussian for all $t$

Similarly, the observation increment  $\mathrm{d}y(t)$ is a linear function of  $a(t)$ plus Gaussian noise.

Hence, for any finite collection of times, the random variables

$$
\big(a(t),,y(t_1),\ldots,y(t_n)\big)
$$

are jointly Gaussian.

Let $\mathcal{Y}_t := \sigma{y(s)\mid 0\le s\le t}$ denote the observation history.

Then $a(t)$  and $\mathcal{Y}_t$ are jointly Gaussian in the sense that all finite-dimensional distributions are jointly Gaussian.

### iii Estimation objective and MMSE criterion

At time $t$ , we seek an estimate of $a(t)$ using only the available observation history $\mathcal{Y}_t$.
Any admissible estimator must be of the form $\eta(\mathcal{Y}_t)$.

The estimation objective is to minimize the mean-square error

$$
\mathbb{E}\big[(a(t)-\eta(\mathcal{Y}_t))^2\big]
$$

This defines a projection problem in the Hilbert space $L^2(\Omega)$ of square-integrable random variables, equipped with the inner product

$$
\langle X,Z\rangle := \mathbb{E}[XZ].
$$

The set of all $\mathcal{Y}_t$-measurable random variables forms a closed linear subspace of $L^2(\Omega)$.

### iv Orthogonal decomposition and Pythagorean identity

For any admissible estimator $\eta(\mathcal{Y}_t)$, we decompose the estimation error as

$$
a(t)-\eta = \big(a(t)-\mathbb{E}[a(t)\mid\mathcal{Y}_t]\big) + \big(\mathbb{E}[a(t)\mid\mathcal{Y}_t]-\eta\big)
$$

Squaring and taking expectations yields:

$$
\mathbb{E}[(a-\eta)^2] = \mathbb{E}[(a-\mathbb{E}[a\mid\mathcal{Y}_t])^2] + \mathbb{E}[(\mathbb{E}[a\mid\mathcal{Y}_t]-\eta)^2]
$$

$$
\qquad\qquad\qquad\quad + 2\mathbb{E}\big[(a-\mathbb{E}[a\mid\mathcal{Y}_t])(\mathbb{E}[a\mid\mathcal{Y}_t]-\eta)\big]
$$

Using the tower property,

$$
\mathbb{E}[Z] = \mathbb{E}[\mathbb{E}[Z\mid\mathcal{Y}_t]],
$$

The cross term:

$$
\mathbb{E}\big[(a-\mathbb{E}[a\mid\mathcal{Y}_t])(\mathbb{E}[a\mid\mathcal{Y}_t]-\eta)\big] = \mathbb{E}\big[\mathbb{E}\big[(a-\mathbb{E}[a\mid\mathcal{Y}_t])(\mathbb{E}[a\mid\mathcal{Y}_t]-\eta)\mid\mathcal{Y}_t\big]\big]
$$

Given Y，$\mathbb{E}[a\mid\mathcal{Y}_t]-\eta$ can be confirmed，and 

$$
\mathbb{E}\big[a-\mathbb{E}[a\mid\mathcal{Y}_t]\mid\mathcal{Y}_t  \big] = 0
$$

we obtain

$$
\mathbb{E}[(a-\eta)^2] = \mathbb{E}[(a-\mathbb{E}[a\mid\mathcal{Y}_t])^2] + \mathbb{E}[(\mathbb{E}[a\mid\mathcal{Y}_t]-\eta)^2]
$$

This is the Pythagorean identity in $L^2(\Omega)$.

### v. Optimal estimator as conditional expectation

The first term in the above decomposition is independent of $\eta$, while the second term is nonnegative and vanishes if and only if

$$
\eta(\mathcal{Y}_t) = \mathbb{E}[a(t)\mid\mathcal{Y}_t].
$$

Therefore, the unique minimum–mean–square error (MMSE) estimator is

$$
\hat a(t) = \mathbb{E}[a(t)\mid\mathcal{Y}_t]
$$

This result does not rely on Gaussianity; it follows solely from the quadratic loss criterion and the geometry of $L^2(\Omega)$

To compute $\hat a(t)$  in continuous time, we exploit the fact that new information arrives through the observation increment  $\mathrm{d}y(t)$.

The observation history satisfies

$$
\mathcal{Y}_{t+\mathrm{d}t} = \sigma\big(\mathcal{Y}_t,\mathrm{d}y(t)\big)
$$

By definition,

$$
\hat a(t+\mathrm{d}t) = \mathbb{E}[a(t+\mathrm{d}t)\mid\mathcal{Y}_{t+\mathrm{d}t}] = \mathbb{E}[a(t+\mathrm{d}t)\mid\mathcal{Y}_t,\mathrm{d}y(t)]
$$

Writing $a(t+\mathrm{d}t)=a(t)+\mathrm{d}a(t)$ and subtracting  $\hat a(t)$, we obtain

$$
\mathrm{d}\hat a = \hat a(t+\mathrm{d}t) - \hat a(t) = \mathbb{E}[a(t+\mathrm{d}t)\mid\mathcal{Y}_t,\mathrm{d}y(t)] - \mathbb{E}[a(t)\mid\mathcal{Y}_t]
$$

$$
\boxed{\mathrm{d}\hat a = \mathbb{E}[\mathrm{d}a\mid\mathcal{Y}_t,\mathrm{d}y] + \big(\mathbb{E}[a(t)\mid\mathcal{Y}_t,\mathrm{d}y]-\mathbb{E}[a(t)\mid\mathcal{Y}_t]\big)}
$$

The first term represents the predicted drift of the estimate, while the second term represents the correction induced by the newly acquired observation increment.

Regardless of whether the condition is $\mathcal{Y}_t$  or $\mathrm{d}y$ ,the Brownian increment mean is 0.

$$
\boxed{\mathbb{E}[\mathrm{d}a\mid\mathcal{Y}_t,\mathrm{d}y] = \mathbb{E}[-\lambda a(t)\mathrm{d}t + \sqrt{Q}\mathrm{d}W_t\mid\mathcal{Y}_t,\mathrm{d}y] = -\lambda\mathbb{E}[a\mid\mathcal{Y}_t,\mathrm{d}y]\mathrm{d}t =  -\lambda\hat adt}
$$

By the joint Gaussian conditional mean formula，

$$
\mathbb{E}[\,x \mid y\,]=\mu_x+\Sigma_{xy}\,\Sigma_{yy}^{-1}\,(y-\mu_y)
$$

$$
\begin{pmatrix}
x \\ y
\end{pmatrix}
\sim
\mathcal{N}
\left(
\begin{pmatrix}
\mu_x \\ \mu_y
\end{pmatrix},
\begin{pmatrix}
\Sigma_{xx} & \Sigma_{xy} \\
\Sigma_{yx} & \Sigma_{yy}
\end{pmatrix}
\right)
$$

$$
\mathbb{E}[\,x \mid y\,]=\mu_x+\frac{\operatorname{Cov}(x,y)}{\operatorname{Var}(y)}\,\bigl(y-\mu_y\bigr)
$$

Conditioned on $\mathcal{Y}_t$, the pair (a(t) ,$\mathrm{d}y(t))$ is jointly Gaussian, since  $\mathrm{d}y(t)$ is a linear function of $a(t)$ plus Gaussian noise

$$
a(t)\mid\mathcal{Y}_t \sim \text{Gaussian}
$$

$$
dy\mid\mathcal{Y}_t = \phi(t)\,a(t)\,dt + R^{1/2}\,dV_t,dV_t \sim \mathcal{N}(0,\,dt)
$$

Hence,$\bigl(a(t),\,dy\bigr)\mid\mathcal{Y}_t$ is jointly Gaussian

$$
\mu_a = \mathbb{E}[a(t)\mid\mathcal{Y}_t] = \hat a(t)
$$

$$
\mu_y = \mathbb{E}[dy\mid\mathcal{Y}_t]
$$

$$
P = \operatorname{Var}(a(t)\mid\mathcal{Y}_t)
$$

$$
R = \operatorname{Var}(dy\mid\mathcal{Y}_t)
$$

$$
C = \operatorname{Cov}(a(t),dy\mid\mathcal{Y}_t)
$$

$$
\begin{pmatrix}a(t) \\ dy\end{pmatrix}\Bigg|\mathcal{Y}_t\sim\mathcal{N}\left(\begin{pmatrix}\mu_a \\ \mu_y\end{pmatrix},\begin{pmatrix}P & C \\C^\top & R\end{pmatrix}\right)
$$

$$
\mathbb{E}[a(t)\mid\mathcal{Y}_t,dy]=\hat a(t)+\operatorname{Cov}(a(t),dy\mid\mathcal{Y}_t)\operatorname{Var}(dy\mid\mathcal{Y}_t)^{-1}\left(dy-\mathbb{E}[dy\mid\mathcal{Y}_t]\right)
$$

so,

$$
\mathbb{E}[a(t)\mid\mathcal{Y}_t,\mathrm{d}y]-\mathbb{E}[a(t)\mid\mathcal{Y}_t] = \operatorname{Cov}(a(t),dy\mid\mathcal{Y}_t)\operatorname{Var}(dy\mid\mathcal{Y}_t)^{-1}\left(dy-\mathbb{E}[dy\mid\mathcal{Y}_t]\right) 
$$

Hence, the conditional expectation $\mathbb{E}[a(t)\mid\mathcal{Y}_t,\mathrm{d}y]$  is linear in $\mathrm{d}y$ and can be expressed as a local regression on the innovation

$$
\mathrm{d}\nu(t) = \mathrm{d}y(t) - \mathbb{E}[\mathrm{d}y(t)\mid\mathcal{Y}_t] = \mathrm{d}y(t) - \phi(t)\hat a(t)\mathrm{d}t
$$

This yields the continuous-time update structure for the optimal estimator.

Conditioned on $\mathcal{Y}_t$,Noise increment $\mathrm{d}V_t$ and $a(t)$ are independent, Calculate the two conditional covariances:

$$
\mathrm{d}y(t) = \phi(t)a(t)\mathrm{d}t + \sqrt{R}\mathrm{d}V_t
$$

$$
\boxed{\operatorname{Cov}(a(t),dy\mid\mathcal{Y}_t) = \operatorname{Cov}(a(t),\phi(t)a(t)\mathrm{d}t\mid\mathcal{Y}_t) =\operatorname{Cov}(a(t),a(t)\mathrm{d}t\mid\mathcal{Y}_t)\phi(t)^\top = P(t)\phi(t)^\top\mathrm{d}t}
$$

where $P(t)\mathrm dt = \operatorname{Cov}(a(t),a(t)\mathrm{d}t\mid\mathcal{Y}_t)$

$$
\operatorname{Cov}(a(t),a(t)\mathrm dt\mid\mathcal Y_t)=\mathrm dt\;\operatorname{Cov}(a(t),a(t)\mid\mathcal Y_t)=\mathrm dt\;\operatorname{Var}(a(t)\mid\mathcal Y_t).
$$

calculate $R = \operatorname{Var}(dy\mid\mathcal{Y}_t)$

$$
\boxed{R = \operatorname{Var}(dy\mid\mathcal{Y}_t) = \phi(t)a(t)\mathrm{d}t + \sqrt{R}\mathrm{d}V_t = Rdt + o(dt^2)}
$$

Finally, we obtain the differential of the $\mathrm{d}\hat a$

$$
\boxed{\mathrm{d}\hat a = -\lambda\hat adt +P(t)\phi(t)^\top\mathrm{d}t (Rdt)^{-1}(\mathrm{d}y(t) - \phi(t)\hat a(t)\mathrm{d}t)}
$$

### vi Define the mean-square estimation error and derive  $\dot P(t)$

Define the estimation error

$$
e(t) = a(t) - \hat a(t)
$$

We define the (scalar) error mean-square as

$$
P(t) = \mathbb E\!\left[e(t)^2\mid \mathcal Y_t\right]
$$

$$
P(t)=\mathbb E\!\left[e(t)^2\mid \mathcal Y_t\right]
      =\operatorname{Var}\!\left(a(t)\mid \mathcal Y_t\right).
$$

(Equivalently, one may use the conditional version $P(t):=\mathbb{E}[e(t)^2\mid \mathcal{Y}_t]$; in the linear–Gaussian setting this $P(t)$ turns out to be deterministic and satisfies the same Riccati ODE. 

$$
\mathrm{d}e(t) = \mathrm{d}a(t) - \mathrm{d}\hat a(t)
$$

$$
e_y(t) = y(t) - \phi(t)\hat a(t).
$$

$$
\mathrm{d}y(t) = \phi(t)a(t)\mathrm{d}t + \sqrt{R}\mathrm{d}V_t
$$

Substitute the expressions above:

$$
\mathrm{d}e(t) = \Big(-\lambda a(t)\mathrm{d}t + \sqrt{Q}\mathrm{d}W_t\Big) - \Big(-\lambda \hat a(t)\mathrm{d}t + K(t)\big(\mathrm{d}y(t) - \phi(t)\hat a(t)\mathrm{d}t)\Big) 
$$

Group terms. First the drift terms:

$$
-\lambda a(t)\mathrm{d}t + \lambda \hat a(t)\mathrm{d}t = -\lambda (a(t)-\hat a(t))\mathrm{d}t = -\lambda e(t)\mathrm{d}t
$$

Then the correction drift:

$$
-K(t)\big(\mathrm{d}y(t) - \phi(t)\hat a(t)\mathrm{d}t) = -K(t)\big(\phi(t)a(t)\mathrm{d}t + \sqrt{R}\mathrm{d}V_t -\phi(t)\hat a(t)\mathrm{d}t\big) = -K(t)\phi(t)e(t)\mathrm{d}t - K(t)\sqrt{R}\mathrm{d}V_t
$$

Therefore the error SDE is

$$
\mathrm{d}e(t) = -\big(\lambda + K(t)\phi(t)\big)e(t)\mathrm{d}t + \sqrt{Q}\mathrm{d}W_t - K(t)\sqrt{R}\mathrm{d}V_t
$$

$$
\mathrm{d}(e^2) = 2e\mathrm{d}e + (\mathrm{d}e)^2
$$

**The term  $2e\mathrm{d}e$.** Using the error SDE,

$$
2e\mathrm{d}e = 2e\Big( -(\lambda+K\phi)e\mathrm{d}t + \sqrt{Q}\mathrm{d}W_t - K\sqrt{R}\mathrm{d}V_t \Big)=-2(\lambda+K\phi)e^2\mathrm{d}t + 2e\sqrt{Q}\mathrm{d}W_t - 2eK\sqrt{R}\mathrm{d}V_t
$$

**The quadratic variation term $(\mathrm{d}e)^2$.** Since $\mathrm{d}t$ is higher order compared to Brownian increments, we keep only the $\mathrm{d}W_t$ and $\mathrm{d}V_t$ parts:

$$
\mathrm{d}e = \cdots + \sqrt{Q}\mathrm{d}W_t - K\sqrt{R}\mathrm{d}V_t
$$

Thus,

$$
(\mathrm{d}e)^2 = (\sqrt{Q}\mathrm{d}W_t - K\sqrt{R}\mathrm{d}V_t)^2== Q(\mathrm{d}W_t)^2 - 2K\sqrt{QR}\mathrm{d}W_t\mathrm{d}V_t + K^2 R(\mathrm{d}V_t)^2
$$

Using independence of $W_t$ and $V_t$,

$$
\mathrm{d}W_t\mathrm{d}V_t = 0
$$

and the Itô identities

$$
(\mathrm{d}W_t)^2=\mathrm{d}t,\qquad (\mathrm{d}V_t)^2=\mathrm{d}t
$$

we obtain

$$
(\mathrm{d}e)^2 = (Q + K^2 R)\mathrm{d}t
$$

$$
\boxed{\mathrm{d}(e^2) = -2(\lambda+K\phi)e^2\mathrm{d}t + (Q+K^2R)\mathrm{d}t + 2e\sqrt{Q}\mathrm{d}W_t - 2eK\sqrt{R}\mathrm{d}V_t}
$$

Recall $P(t)=\mathbb{E}[e(t)^2]$. Taking expectation on both sides:

$$
\mathrm{d}\mathbb{E}[e^2] = \mathbb{E}\big[\mathrm{d}(e^2)\big]
$$

Now use the fact that Itô integrals have zero mean (under standard integrability conditions):

$$
\mathbb{E}\big[e(t)\mathrm{d}W_t\big]=0,\qquad \mathbb{E}\big[e(t)\mathrm{d}V_t\big]=0.
$$

Hence the stochastic terms vanish in expectation, yielding

$$
\mathrm{d}P(t) = \Big(-2(\lambda+K(t)\phi(t))\mathbb{E}[e(t)^2] + Q + K(t)^2 R\Big)\mathrm{d}t
$$

$$
\dot P(t) = -2(\lambda+K(t)\phi(t))P(t) + Q + K(t)^2 R.
$$

For the linear–Gaussian MMSE filter, the gain is

$$
K(t) = \dfrac{P(t)\phi(t)}{R}.
$$

Substitute into the ODE:

$$
\dot P(t) = -2\lambda P(t) - 2\Big(\dfrac{P(t)\phi(t)}{R}\Big)\phi(t)P(t) + Q + \Big(\dfrac{P(t)\phi(t)}{R}\Big)^2 R.
$$

Compute each term carefully:

The middle drift correction term:

$$
- 2\Big(\dfrac{P\phi}{R}\Big)\phi P = -\dfrac{2\phi(t)^2}{R}P(t)^2
$$

The last term:

$$
\Big(\dfrac{P\phi}{R}\Big)^2 R = \dfrac{\phi(t)^2}{R}P(t)^2
$$

Combine them:

$$
-\dfrac{2\phi(t)^2}{R}P(t)^2 + \dfrac{\phi(t)^2}{R}P(t)^2 = -\dfrac{\phi(t)^2}{R}P(t)^2
$$

Therefore the final Riccati equation for the mean-square estimation error is

$$
\boxed{\dot P(t) = -2\lambda P(t) + Q - \dfrac{\phi(t)^2}{R}P(t)^2.}
$$

This equation shows explicitly how the uncertainty evolves:

- $-2\lambda P$ comes from the stable drift $-\lambda a$ (forgetting/decay of uncertainty),
- $+Q$  comes from process noise injection
- $-(\phi^2/R)P^2$ is the information gain from measurements (stronger sensing $\phi$ or smaller noise $R$ reduces $P$ faster).

### vii  Matrix generalization

Now let:

$$
a(t)\in\mathbb{R}^{h},\qquad\hat a(t)\in\mathbb{R}^{h},\qquad\tilde a=\hat a-a\in\mathbb{R}^{h},
$$

and measurement: $y(t)\in\mathbb{R}^{n},\qquad
y = \phi\,a + \epsilon,\qquad
\phi\in\mathbb{R}^{n\times h}.$

Noise covariances:

$$
\mathbb{E}[\epsilon\epsilon^\top]=R,\quad R\in\mathbb{R}^{n\times n},\ R\succ 0,\qquad\mathbb{E}[\nu\nu^\top]=Q,\quad Q\in\mathbb{R}^{h\times h},\ Q\succeq 0.
$$

The continuous-time Kalman–Bucy estimator has the form:

$$
\boxed{\dot{\hat a} = -\lambda \hat a + P\phi^\top R^{-1}\big(y-\phi\hat a\big)}
$$

and the covariance evolution becomes the matrix Riccati equation:

$$
\boxed{\dot P = -2\lambda P + Q - P\phi^\top R^{-1}\phi P.}
$$

The matrix Kalman gain is:

$$
\boxed{K_a(t) = P(t)\phi(t)^\top R^{-1}.}
$$

In our composite adaptation law, we will then add the tracking-driven term:

$$
\dot{\hat a} = -\lambda \hat a + P\phi^\top R^{-1}\big(y-\phi\hat a\big) + P\phi^\top s.
$$