# Neural-Fly Reproduction with Delay-Augmented Regressor

This repository reproduces the Neural-Fly framework  
(Michael O’Connell and Guanya Shi and Xichen Shi and Kamyar Azizzadenesheli and Anima Anandkumar and Yisong Yue and Soon-Jo Chung, "Neural-Fly enables rapid learning for agile flight in strong winds", 2022)  
and provides a minimal, physically motivated extension to its regressor structure.

In addition to a faithful baseline reproduction,  
we introduce a delay-augmented regressor to partially account for actuator-induced  
non-Markovian effects under closed-loop flight.

---

## 1. Core Idea of the Original Paper

The key modeling assumption in Neural-Fly is to decompose the aerodynamic disturbance  
into two components:

f_a(t) = φ(x(t))ᵀ a(t)

where:

- φ(x) is a wind-invariant basis function learned offline from flight data,  
- a(t) is a low-dimensional, wind-dependent coefficient vector estimated online,  
- x(t) = [v(t), q(t), pwm(t)] denotes the measured state and control input.

This formulation treats the aerodynamic residual force as a **linearly parameterized model**  
with respect to a(t), while allowing φ(·) to capture nonlinear state-dependent effects.

---

### Why This Works

This structure is effective for three key reasons:

1. **Wind-Invariant Representation**  
   The neural network φ(·) is trained across multiple wind conditions and trajectories.  
   It learns a representation that is invariant to specific gust realizations,  
   while preserving sensitivity to the quadrotor's state and actuation.

2. **Online Adaptation via Linear Parameterization**  
   All wind-specific variations are absorbed into the low-dimensional vector a(t).  
   Since the model is linear in a(t), classical adaptive filtering techniques  
   can be applied in real time.

3. **Kalman–Bucy Estimation**  
   The online estimation of a(t) is performed using a continuous-time Kalman–Bucy filter,  
   which provides closed-form update equations for both the coefficient estimates  
   and their covariance.  
   This enables fast adaptation to changing wind conditions  
   while maintaining closed-loop stability.

In essence, Neural-Fly converts a complex, nonlinear, and unknown aerodynamic disturbance  
into a linear-in-parameters surrogate model whose coefficients can be tracked online.

---

## 2. Closed-Loop Data and Modeling Limitations

All training data in Neural-Fly are collected under a strongly closed-loop flight controller  
with a stabilizing -KS term. As a result:

- The measured states v(t), q(t), and control inputs pwm(t) are actively regulated.  
- Wind-induced deviations are partially suppressed by feedback control.  
- The observed aerodynamic residual f_a(t) reflects a closed-loop response,  
  rather than a pure open-loop aerodynamic force.

Under this setting, the true non-Markovian nature of aerodynamic forces  
(e.g., due to vortex shedding and flow separation history)  
is not directly identifiable from the available measurements.

Consequently, the objective of Neural-Fly is not to recover the true aerodynamic dynamics,  
but to construct a **Markovian surrogate model**  
that best explains the closed-loop residual forces under limited observability.

---

## 3. Our Modification: Delay-Augmented Regressor

While Neural-Fly assumes a Markovian mapping  

f_a(t) ≈ φ(x(t))ᵀ a(t),

the closed-loop system introduces additional hidden states,  
most notably actuator dynamics:

pwm(t) → ω(t) → thrust(t).

This chain implies that thrust and aerodynamic forces do not respond instantaneously  
to pwm(t), but depend on its recent history.

To partially account for this effect with minimal architectural changes,  
we augment the regressor input with delayed PWM values:

x_aug(t) = [v(t), q(t), pwm(t), pwm(t−1), pwm(t−2), pwm(t−3)].

At a sampling rate of 50 Hz, this corresponds to a 60 ms memory window,  
which aligns with typical motor and ESC response time constants.

Importantly:

- The neural network φ(·) is retrained using the augmented input.  
- The online Kalman–Bucy estimator for a(t) remains unchanged.  
- The linear-in-parameters structure is fully preserved.

This modification does not attempt to model true aerodynamic memory.  
Instead, it compensates for actuator-induced non-Markovian effects  
and improves the statistical sufficiency of the regressor  
under closed-loop data collection.

---

## 4. Theory Derivation

We provide a detailed mathematical derivation of the Neural-Fly framework,  
including the linear-in-parameters formulation  
and the Kalman–Bucy estimator.

These notes include step-by-step derivations  
that are not explicitly provided in the original paper.
- 📄 [PDF version](docs/neural_fly_derivation.pdf)

---

## 5. Results Summary

We report the **mean absolute error (MAE)** on the test set under different wind conditions.  
Results are grouped by wind intensity and compared between the baseline Neural-Fly model  
and the delay-augmented regressor.

| Wind Condition | Baseline MAE | Delay-Augmented MAE |
|----------------|--------------|---------------------|
| 0 wind         | 0.36         | 0.33                |
| 35 wind        | 0.72         | 0.69                |
| 70 wind        | 1.53         | 1.43                |
| 100 wind       | 3.19         | 3.18                |

The delay-augmented regressor consistently reduces the average prediction error  
across moderate and strong wind conditions, with the most noticeable improvement  
observed under higher gust intensities.

## 6. Discussion

Although real aerodynamic forces are inherently history-dependent  
due to vortex dynamics, their non-Markovian nature is not cleanly identifiable  
from closed-loop flight data with limited state observability.

Our delay-augmented regressor does not attempt to model true aerodynamic memory.  
Instead, it compensates for actuator-induced non-Markovian effects  
and improves the sufficiency of the state representation.

This aligns with the original Neural-Fly philosophy:  
constructing a tractable surrogate model that is compatible  
with online adaptive estimation and closed-loop control,  
rather than pursuing a fully physical aerodynamic model.

---

## 7. Acknowledgement

This repository is based on the original Neural-Fly implementation by Zhou et al.  
All credit for the core methodology belongs to the original authors.

The original implementation is available at: https://github.com/aerorobotics/neural-fly

Our modifications are intended solely for research and educational purposes.

