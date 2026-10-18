# FlyDrive — Physics-First Research & Learning Plan

> **Working question**
>
> **Can biological connectome topology serve as a useful structural prior for reinforcement-learning-based autonomous control?**
>
> 이 프로젝트의 1차 목적은 “초파리 뇌로 기존 자율주행 모델을 대체한다”가 아니다.  
> **강화학습과 연속 제어를 직접 구현·재현하면서 이해하고**, 그 위에서 **Drosophila connectome topology가 policy architecture에 주는 효과를 controlled experiment로 측정**하는 것이 목적이다.

---

# 0. 가장 먼저 볼 수식들

## 0.1 Controlled dynamical system

프로젝트 전체를 다음의 확률적 동역학계로 본다.

\[
s_{t+1}\sim P(s_{t+1}\mid s_t,a_t),
\qquad
a_t\sim \pi_\theta(a_t\mid s_t).
\]

자율주행 toy model에서는

\[
s_t =
(x_t,y_t,\psi_t,v_t,\ldots),
\qquad
a_t=(\delta_t, a_t^{\rm long})
\]

처럼 상태와 제어입력을 정의한다.

우리가 바꾸려는 핵심은 **환경의 물리법칙이 아니라 policy \(\pi_\theta\)의 구조**다.

---

## 0.2 Objective

강화학습의 기본 목적함수는

\[
J(\theta)
=
\mathbb E_{\tau\sim\pi_\theta}
\left[
\sum_{t=0}^{T}\gamma^t r(s_t,a_t)
\right].
\]

trajectory는

\[
\tau=(s_0,a_0,s_1,a_1,\ldots)
\]

이고, trajectory probability는

\[
p_\theta(\tau)
=
p(s_0)
\prod_t
\pi_\theta(a_t\mid s_t)
P(s_{t+1}\mid s_t,a_t)
\]

이다.

이 식에서 중요한 점은 **물리 dynamics \(P\)** 와 **policy \(\pi_\theta\)** 를 분리해서 생각할 수 있다는 것이다.

---

## 0.3 Bellman equations

\[
V^\pi(s)
=
\mathbb E_\pi
\left[
r_t+\gamma V^\pi(s_{t+1})
\mid s_t=s
\right]
\]

\[
Q^\pi(s,a)
=
\mathbb E_\pi
\left[
r_t+\gamma Q^\pi(s_{t+1},a_{t+1})
\right].
\]

Advantage:

\[
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s).
\]

이 세 양의 의미를 수치 실험으로 직접 확인하는 것이 첫 학습 목표다.

---

## 0.4 Policy gradient

stochastic policy에 대해

\[
\nabla_\theta J(\theta)
=
\mathbb E_{\pi_\theta}
\left[
\nabla_\theta\log\pi_\theta(a_t\mid s_t)
A^{\pi_\theta}(s_t,a_t)
\right].
\]

이 식은 이후 PPO의 출발점이 된다.

---

## 0.5 PPO

old policy와 new policy의 probability ratio를

\[
r_t(\theta)
=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{\rm old}}(a_t\mid s_t)}
\]

라 하면 PPO의 clipped objective는

\[
L^{\rm CLIP}(\theta)
=
\mathbb E_t
\left[
\min
\left(
r_t(\theta)\hat A_t,
\operatorname{clip}
(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t
\right)
\right].
\]

GAE를 사용할 경우

\[
\delta_t
=
r_t+\gamma V(s_{t+1})-V(s_t),
\]

\[
\hat A_t^{\rm GAE}
=
\sum_{l=0}^{\infty}
(\gamma\lambda)^l\delta_{t+l}.
\]

---

## 0.6 Deterministic policy gradient

continuous action에서 deterministic policy

\[
a=\mu_\theta(s)
\]

를 사용하면

\[
\nabla_\theta J
\approx
\mathbb E_s
\left[
\nabla_a Q_\phi(s,a)
\big|_{a=\mu_\theta(s)}
\nabla_\theta\mu_\theta(s)
\right].
\]

DDPG와 TD3를 이해하기 위한 핵심 식이다.

---

## 0.7 Soft Actor-Critic

SAC는 reward뿐 아니라 policy entropy도 최대화한다.

\[
J(\pi)
=
\mathbb E
\left[
\sum_t
\gamma^t
\left(
r(s_t,a_t)
+
\alpha\mathcal H(\pi(\cdot\mid s_t))
\right)
\right].
\]

soft Bellman target은 개념적으로

\[
Q(s_t,a_t)
\leftarrow
r_t+
\gamma
\mathbb E_{a_{t+1}\sim\pi}
\left[
Q(s_{t+1},a_{t+1})
-
\alpha\log\pi(a_{t+1}\mid s_{t+1})
\right].
\]

PPO와 SAC의 차이를 직접 비교하는 것이 중요한 학습 목표다.

---

# 1. Vehicle dynamics

처음부터 CARLA를 사용하지 않는다.

먼저 직접 이해할 수 있는 **kinematic bicycle model**을 Gymnasium environment로 구현한다.

\[
\dot x=v\cos\psi
\]

\[
\dot y=v\sin\psi
\]

\[
\dot\psi
=
\frac{v}{L}\tan\delta
\]

\[
\dot v=a.
\]

Euler discretization:

\[
x_{t+1}=x_t+v_t\cos\psi_t\,\Delta t
\]

\[
y_{t+1}=y_t+v_t\sin\psi_t\,\Delta t
\]

\[
\psi_{t+1}
=
\psi_t+
\frac{v_t}{L}\tan\delta_t\,\Delta t
\]

\[
v_{t+1}=v_t+a_t\Delta t.
\]

### 첫 환경

초기에는 \(v=v_0\)로 고정하고 steering만 제어한다.

\[
a_t=\delta_t\in[-\delta_{\max},\delta_{\max}].
\]

state는 Frenet-like coordinate로 단순화한다.

\[
s_t=
(e_y,e_\psi,v,\kappa)
\]

- \(e_y\): lane center에서의 lateral error
- \(e_\psi\): road tangent와 차량 heading 차이
- \(v\): speed
- \(\kappa\): local road curvature

### Reward

예를 들어

\[
r_t
=
w_p\Delta s_t
-
w_y e_{y,t}^2
-
w_\psi e_{\psi,t}^2
-
w_\delta\delta_t^2
-
w_{\Delta\delta}
(\delta_t-\delta_{t-1})^2.
\]

각 항은 물리적으로 해석 가능하게 유지한다.

처음부터 reward를 복잡하게 만들지 않는다.

---

# 2. 이 프로젝트의 두 축

## Track A — 공부 / 재현

목표:

> 기존 RL algorithm이 왜 작동하는지를 논문과 수식에서 시작해 직접 구현하면서 이해한다.

처음에는 Stable-Baselines3를 답안으로 사용하지 않는다.

가능하면 작은 implementation을 직접 작성하고, 이후 검증용으로 library 결과와 비교한다.

## Track B — 연구

목표:

> 동일한 environment와 RL algorithm에서 policy architecture만 바꿨을 때 Drosophila connectome topology가 어떤 차이를 만드는지 측정한다.

즉 최종 비교는 대략

\[
\text{MLP}
\leftrightarrow
\text{RNN}
\leftrightarrow
\text{Random sparse graph}
\leftrightarrow
\text{Fly connectome}
\]

이다.

---

# 3. Baseline learning ladder

## Stage 0 — Dynamic programming / tabular RL

가능하면 작은 toy MDP에서 먼저

\[
V,\quad Q,\quad A
\]

를 직접 계산한다.

구현:

- Value iteration
- Policy iteration
- Tabular Q-learning

목표는 neural network가 들어오기 전에 Bellman equation을 수치적으로 이해하는 것이다.

---

## Stage 1 — DQN

### 읽을 논문

**Mnih et al., “Human-level control through deep reinforcement learning,” Nature (2015).**

DQN은 최종 continuous-control benchmark가 아니라 **deep RL의 기본 장치들을 공부하기 위한 구현 과제**로 둔다.

핵심:

\[
y_t
=
r_t+
\gamma
\max_{a'}
Q_{\theta^-}(s_{t+1},a')
\]

\[
L(\theta)
=
\mathbb E
\left[
(y_t-Q_\theta(s_t,a_t))^2
\right].
\]

배울 것:

- replay buffer
- target network
- bootstrapping
- off-policy learning
- instability of function approximation

추천 환경:

- CartPole
- MountainCar

**Exit condition:** 작은 PyTorch implementation으로 학습 곡선을 재현하고 target network/replay buffer 제거 ablation을 해본다.

---

## Stage 2 — Deterministic Policy Gradient → DDPG

### 읽을 논문

1. **Silver et al., “Deterministic Policy Gradient Algorithms,” ICML (2014).**
2. **Lillicrap et al., “Continuous Control with Deep Reinforcement Learning” (2015/2016).**

핵심:

\[
a=\mu_\theta(s)
\]

\[
\nabla_\theta J
=
\mathbb E
[
\nabla_aQ(s,a)
\nabla_\theta\mu_\theta(s)
].
\]

추천 환경:

- Pendulum-v1
- MountainCarContinuous-v0

이 단계부터 프로젝트의 실제 문제인 **continuous action**으로 들어간다.

---

## Stage 3 — PPO

### 읽을 논문

**Schulman et al., “Proximal Policy Optimization Algorithms” (2017).**

PPO는 FlyDrive의 첫 주력 baseline으로 사용한다.

이유:

- 비교적 구현이 단순함
- on-policy라 데이터 흐름을 이해하기 쉬움
- recurrent/connectome policy를 붙이기 상대적으로 자연스러움

실험:

1. Pendulum
2. MountainCarContinuous
3. 직접 만든 LaneKeeping-v0

먼저 MLP policy로 benchmark한다.

---

## Stage 4 — TD3

### 읽을 논문

**Fujimoto, van Hoof & Meger, “Addressing Function Approximation Error in Actor-Critic Methods,” ICML (2018).**

핵심 질문:

> 왜 actor-critic의 \(Q\) estimate가 과대평가될 수 있는가?

TD3에서는 두 critic을 두고

\[
y
=
r+
\gamma
\min_{i=1,2}
Q_{\phi_i^-}(s',\tilde a')
\]

를 사용한다.

이 단계는 반드시 최종 benchmark에 포함해야 하는 것은 아니지만, **function approximation error를 공부하기에 매우 좋은 실험**이다.

---

## Stage 5 — SAC

### 읽을 논문

**Haarnoja et al., “Soft Actor-Critic Algorithms and Applications” (2018).**

SAC는 PPO와 함께 최종 continuous-control baseline 후보로 둔다.

비교:

\[
\text{PPO: on-policy}
\]

vs.

\[
\text{SAC: off-policy + maximum entropy}.
\]

특히 sample efficiency 비교에 중요하다.

---

# 4. Benchmark suite

처음부터 자동차 환경 하나만 보면 구현 오류와 environment 특성을 분리하기 어렵다.

따라서 다음 ladder를 사용한다.

| Level | Environment | 목적 |
|---|---|---|
| 0 | tiny custom MDP | Bellman equation 검증 |
| 1 | CartPole | DQN / discrete control |
| 2 | Pendulum | 가장 단순한 continuous control |
| 3 | MountainCarContinuous | exploration + continuous control |
| 4 | LaneKeeping-v0 | 직접 구현한 vehicle dynamics |
| 5 | CurvedRoad-v0 | generalization |
| 6 | 향후 CARLA | 실제에 가까운 driving |

Gymnasium의 Pendulum은 torque continuous control이고, Continuous Mountain Car는 1차원 continuous force를 사용하므로 초기 algorithm 검증에 적합하다.

---

# 5. M1 Max 중심 Phase I

## Hardware

Primary:

- Apple M1 Max
- 32 GB unified memory
- PyTorch MPS / Metal

이 단계의 목표는 **최대 성능이 아니라 빠른 학습·검증 loop**다.

### Phase I에서 하지 않을 것

- whole-brain connectome
- RGB camera
- CARLA
- 거대한 hyperparameter sweep
- multi-GPU optimization

---

# 6. M1 Max 실험 순서

## M0 — Environment / MPS sanity check

측정:

\[
t_{\rm env},\quad
t_{\rm forward},\quad
t_{\rm backward},\quad
N_{\rm steps/s}
\]

CPU와 MPS를 모두 측정한다.

작은 network에서는 MPS overhead 때문에 CPU가 더 빠를 가능성도 있으므로 **측정해서 결정한다.**

---

## M1 — Known environments

직접 구현한 algorithm을 기존 Gymnasium environment에 적용한다.

순서:

1. DQN → CartPole
2. DDPG → Pendulum
3. PPO → Pendulum
4. TD3 → Pendulum
5. SAC → Pendulum
6. PPO/SAC → MountainCarContinuous

목적은 최고 점수가 아니라 **알고리즘별 learning dynamics를 이해하는 것**이다.

각 실험에서 최소한

\[
R(N_{\rm env})
\]

즉 environment interaction 수에 따른 return을 기록한다.

---

## M2 — Custom LaneKeeping-v0

Gymnasium API를 따르는 환경을 직접 구현한다.

초기 상태:

\[
s=(e_y,e_\psi)
\]

또는

\[
s=(e_y,e_\psi,v,\kappa).
\]

action:

\[
a=\delta.
\]

먼저 deterministic dynamics에서 시작한다.

이후

\[
s_{\rm obs}=s+\epsilon,
\qquad
\epsilon\sim\mathcal N(0,\Sigma)
\]

를 추가하여 sensor noise를 도입한다.

---

## M3 — Classical controller benchmark

RL끼리만 비교하지 않는다.

물리 기반 기준점을 하나 둔다.

예:

- proportional / PD steering controller
- Stanley controller
- Pure Pursuit

예를 들어 간단한 feedback law

\[
\delta
=
-k_y e_y-k_\psi e_\psi
\]

를 먼저 구현한다.

이 controller가 toy environment에서 어느 정도의 성능을 내는지 측정한다.

이 단계는 매우 중요하다.

RL이 단순한 feedback control보다 나은지 여부보다도, **environment와 reward가 물리적으로 정상인지 확인하는 sanity check**가 된다.

---

# 7. Benchmark protocol

모든 architecture에 가능한 한 같은 protocol을 적용한다.

## Primary axis

\[
\boxed{\text{Environment interactions}}
\]

wall-clock time만으로 비교하지 않는다.

### 기록

\[
R(N)
\]

\[
e_y^{\rm RMS}
=
\sqrt{
\frac1T
\sum_t e_{y,t}^2
}
\]

\[
e_\psi^{\rm RMS}
=
\sqrt{
\frac1T
\sum_t e_{\psi,t}^2
}
\]

control effort:

\[
E_u
=
\frac1T
\sum_t \delta_t^2
\]

smoothness:

\[
E_{\Delta u}
=
\frac1{T-1}
\sum_t
(\delta_t-\delta_{t-1})^2.
\]

추가로:

- failure rate
- completion rate
- training wall time
- inference latency
- peak memory
- steps/s

를 기록한다.

---

# 8. Statistics

RL은 seed variance가 크므로 single run으로 결론내리지 않는다.

초기 debugging에서는

\[
N_{\rm seed}=1
\]

로 충분하지만 비교 단계에서는 최소 여러 seed를 사용한다.

보고:

\[
\bar R(N)
\pm
{\rm uncertainty}.
\]

최종 비교에서는 mean/median뿐 아니라 seed별 curve를 보존한다.

---

# 9. Fly 단계로 넘어가는 조건

다음 조건을 만족한 후에 connectome을 넣는다.

- [ ] Bellman/value/Q 관계를 직접 구현으로 확인
- [ ] DQN 학습 성공
- [ ] deterministic policy gradient 이해 및 DDPG 구현
- [ ] PPO 구현 및 benchmark 성공
- [ ] SAC 또는 TD3 구현 성공
- [ ] LaneKeeping-v0 완성
- [ ] classical controller baseline 완성
- [ ] MLP PPO/SAC benchmark curve 확보
- [ ] seed/evaluation protocol 고정

즉 **Fly는 시작점이 아니라 마지막 architecture experiment**다.

---

# 10. Drosophila connectome 단계

FlyWire adult Drosophila whole-brain dataset은 약 \(1.4\times10^5\) neurons 규모다.

처음부터 전체 graph를 사용하지 않는다.

\[
N=10^3
\rightarrow
5\times10^3
\rightarrow
10^4
\rightarrow
\cdots
\]

순서로 확장한다.

graph를

\[
G=(V,E)
\]

라 하고 neuron state를

\[
h_i^{(t)}
\]

라고 하자.

가장 단순한 discrete dynamics는 예를 들어

\[
h_i^{(t+1)}
=
(1-\lambda_i)h_i^{(t)}
+
\sigma
\left(
\sum_{j:(j,i)\in E}
w_{ji}h_j^{(t)}
+
I_i^{(t)}
\right)
\]

처럼 둘 수 있다.

vector notation:

\[
\mathbf h_{t+1}
=
(1-\boldsymbol\lambda)\odot\mathbf h_t
+
\sigma
\left(
W_G\mathbf h_t+B\mathbf s_t
\right).
\]

output:

\[
a_t
=
C\mathbf h_t.
\]

핵심은 \(W_G\)의 sparsity pattern을 실제 connectome에서 가져오는 것이다.

---

# 11. 진짜 연구 질문

단순히

\[
\text{Fly vs MLP}
\]

만 비교해서는 biological topology의 효과라고 말하기 어렵다.

최소한 다음을 비교한다.

\[
G_{\rm fly}
\]

\[
G_{\rm rewired}
\]

\[
G_{\rm random}
\]

\[
G_{\rm generic\ RNN}.
\]

가장 중요한 null hypothesis는

\[
H_0:
\quad
\text{performance}(G_{\rm fly})
=
\text{performance}(G_{\rm matched\ random})
\]

이다.

관심 있는 양은 단순 final reward보다

\[
\frac{dR}{dN_{\rm env}}
\]

로 표현되는 learning efficiency,

noise sensitivity

\[
\frac{\partial R}{\partial \sigma_{\rm noise}},
\]

lesion sensitivity

\[
\frac{\partial R}{\partial f_{\rm removed}},
\]

그리고 out-of-distribution road geometry에서의 generalization이다.

---

# 12. Connectome 관련 읽을 자료

## FlyWire whole-brain connectome

**Schlegel et al. / FlyWire consortium, whole-brain annotation and connectome papers, Nature (2024).**

현재 whole adult fly brain의 구조와 cell-type annotation을 이해하기 위한 출발점.

특히 먼저 알아야 할 것:

- neuron / edge 정의
- synapse count와 edge weight의 관계
- thresholding
- cell types
- sensory / motor pathways
- directionality

## Network statistics of the whole-brain connectome

whole-brain graph를 단순 “거대한 adjacency matrix”가 아니라 network object로 이해하기 위해 읽는다.

## NeuroMechFly v2

**Wang-Chen et al., Nature Methods (2024).**

초파리의 sensorimotor control을 embodied simulation과 연결하는 사례다.

본 프로젝트와 동일한 연구는 아니지만,

\[
\text{neural controller}
\rightarrow
\text{body}
\rightarrow
\text{environment}
\rightarrow
\text{sensory feedback}
\]

이라는 closed loop를 어떻게 구성하는지 참고하기 좋다.

---

# 13. 추천 논문 — 우선순위

## Tier A — 직접 구현

1. **Mnih et al. (2015), DQN**
   - Bellman + neural approximation
   - replay buffer
   - target network

2. **Silver et al. (2014), Deterministic Policy Gradient**
   - continuous action의 수학적 기반

3. **Lillicrap et al. (2015), DDPG**
   - continuous-control actor-critic 구현

4. **Schulman et al. (2017), PPO**
   - 첫 주력 on-policy baseline

5. **Fujimoto et al. (2018), TD3**
   - \(Q\)-function approximation error와 bias 이해

6. **Haarnoja et al. (2018), SAC**
   - maximum-entropy RL
   - 주력 off-policy baseline

## Tier B — Fly 단계 직전

7. **FlyWire whole-brain connectome / annotation papers (Nature, 2024)**

8. **Network statistics of the whole-brain connectome of Drosophila (Nature, 2024)**

9. **NeuroMechFly v2 (Nature Methods, 2024)**

---

# 14. Compute roadmap

## M1 Max 32 GB

주 개발 머신.

사용:

- 모든 기본 RL 공부
- Gymnasium
- LaneKeeping-v0
- MLP/RNN baseline
- 작은 graph
- MPS profiling

## RTX 2070

사용:

- CUDA compatibility
- MPS vs CUDA sanity check
- 작은 graph experiment

## RTX 3090 24 GB

사용:

- larger sparse graph
- CUDA-optimized connectome dynamics
- larger batch / rollout
- 중간 규모 Fly experiments

## NERSC / NVIDIA A100

초기에는 사용하지 않아도 된다.

후반:

- large connectome
- multi-seed
- ablation
- parameter sweep
- scaling study

에 사용한다.

---

# 15. 예상 연구 진행 순서

```text
수식 / MDP
   ↓
Tabular RL
   ↓
DQN
   ↓
DPG / DDPG
   ↓
PPO
   ↓
TD3 / SAC
   ↓
Gym benchmark 고정
   ↓
Kinematic bicycle Gym
   ↓
Classical control baseline
   ↓
MLP / RNN RL baseline
   ↓
Random sparse recurrent network
   ↓
Drosophila connectome subgraph
   ↓
Topology ablation
   ↓
Robustness / generalization
   ↓
larger connectome
   ↓
(optional) CARLA / vision
```

---

# 16. Phase I의 명확한 종료점

M1 Max 단계가 끝났다고 판단하는 기준:

1. PPO와 SAC를 최소 구현으로 이해하고 실행할 수 있다.
2. Pendulum / MountainCarContinuous에서 reference implementation과 질적으로 유사한 learning curve를 얻는다.
3. 직접 만든 bicycle-model Gym에서 classical controller와 RL controller가 모두 작동한다.
4. evaluation protocol이 고정되어 있다.
5. MLP와 RNN baseline의 seed별 curve가 확보되어 있다.
6. graph recurrent policy를 끼워 넣을 interface가 준비되어 있다.

여기까지가 **baseline physics/RL laboratory**다.

그 다음부터 Fly experiment를 시작한다.

---

# 17. 프로젝트의 해석 원칙

가장 중요한 원칙은 다음이다.

> **Fly topology가 더 높은 reward를 냈다는 사실만으로 biological connectome이 더 좋은 architecture라고 결론내리지 않는다.**

parameter count, trainable edge 수, recurrence depth, activation dynamics, optimizer, compute budget이 다르면 topology 효과와 혼동된다.

따라서 최종 실험은 가능한 한

\[
\text{topology}
\]

만 독립변수로 남기는 방향으로 설계한다.

이 프로젝트의 흥미로운 결과는 “Fly가 이겼다”뿐만 아니라 다음 모두가 될 수 있다.

- 차이가 없음
- sample efficiency만 다름
- robustness만 다름
- small graph에서는 효과가 있으나 scaling하면서 사라짐
- random rewiring으로 효과가 사라짐
- degree distribution만 보존해도 효과가 유지됨

이런 결과들이 오히려 **connectome의 어떤 구조적 성질이 중요한가**라는 다음 물리적 질문으로 이어진다.

---

# References / implementation starting points

- Mnih et al., *Human-level control through deep reinforcement learning*, Nature 518, 529–533 (2015). DOI: 10.1038/nature14236
- Silver et al., *Deterministic Policy Gradient Algorithms*, ICML / PMLR 32 (2014).
- Lillicrap et al., *Continuous Control with Deep Reinforcement Learning*, arXiv:1509.02971.
- Schulman et al., *Proximal Policy Optimization Algorithms*, arXiv:1707.06347.
- Fujimoto, van Hoof & Meger, *Addressing Function Approximation Error in Actor-Critic Methods*, ICML / PMLR 80 (2018).
- Haarnoja et al., *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*, arXiv:1801.01290.
- Haarnoja et al., *Soft Actor-Critic Algorithms and Applications*, arXiv:1812.05905.
- FlyWire Consortium / Drosophila whole-brain connectome papers, Nature (2024).
- Wang-Chen et al., *NeuroMechFly v2: simulating embodied sensorimotor control in adult Drosophila*, Nature Methods 21, 2353–2362 (2024).
- Gymnasium Classic Control documentation: Pendulum, CartPole, MountainCar, MountainCarContinuous.

---

## Immediate next task

**Do not implement Fly yet.**

첫 작업은 다음 세 개다.

1. M1 Max에서 PyTorch CPU/MPS benchmark notebook 작성
2. `Pendulum-v1`의 방정식을 직접 읽고 dynamics를 별도 Python 함수로 재현
3. 아주 작은 MDP에서 \(V,Q,A\)와 Bellman update를 직접 계산

그 뒤 DQN → DDPG → PPO 순서로 진행한다.
