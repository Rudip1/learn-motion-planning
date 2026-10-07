# Notation

One notation is used throughout the module. Each chapter defines what it adds; this table collects the
symbols that recur. Vectors are bold only where a scalar with the same letter also appears.

## Frames, poses and angles

| Symbol | Meaning | Unit |
|---|---|---|
| $\{W\}$ | world (inertial) frame, $x$ east, $y$ north, angles counter-clockwise | |
| $\{V\}$ | vehicle frame: origin at the reference point, $x$ forward, $y$ to the left | |
| $q = (x, y, \theta)$ | pose (configuration) of the vehicle reference point in $\{W\}$ | m, m, rad |
| $\dot q$ | time derivative of $q$ | |
| $\mathcal{C}$ | configuration space; $\mathcal{C}_\text{free}$ its collision-free part | |
| $\operatorname{wrap}(\phi)$ | the angle $\phi$ mapped into $[-\pi, \pi)$ | rad |

**Angles.** Headings are kept continuous while integrating (so plots do not jump by $2\pi$); every *difference*
of headings is wrapped, $\operatorname{wrap}(\theta_a - \theta_b)$. Forgetting to wrap a difference is the most
common bug in planar robotics code. In C++ this is `motion_planning::angle_difference(to, from)`.

## Inputs and models (chapter 1)

| Symbol | Meaning | Unit |
|---|---|---|
| $u$ | control input; its components depend on the model | |
| $v$ | forward speed of the reference point | m/s |
| $\omega$ | yaw rate, $\omega = \dot\theta$ | rad/s |
| $\omega_L, \omega_R$ | left and right wheel angular speeds | rad/s |
| $r$ | wheel radius | m |
| $b$ | track width (distance between the wheel contact points) | m |
| $L$ | wheelbase (rear axle to front axle) | m |
| $l_r$ | distance from the rear axle to the reference point | m |
| $\gamma$ | front-wheel steering angle | rad |
| $\beta$ | slip angle of the reference point (chapter 1); bearing term of the pose law (eq. 1.19) | rad |
| $R$ | turning radius | m |
| $\Delta t$ | integration time step | s |
| $f(q, u)$ | model vector field, $\dot q = f(q, u)$ | |

The letter $\beta$ is used twice in chapter 1, following the two textbooks the material comes from; the
section where it appears says which one is meant.

## Planning (chapters 2–8)

| Symbol | Meaning |
|---|---|
| $q_\text{start}, q_\text{goal}$ | start and goal configurations |
| $\mathcal{O}$ | obstacle region in the workspace |
| $c(\cdot)$ | cost of an edge or a path |
| $g(n), h(n), f(n)$ | cost-to-come, heuristic cost-to-go and their sum in graph search |
| $U(q)$ | potential function |

## Deciding (chapters 9–11)

| Symbol | Meaning |
|---|---|
| $s, a, r$ | state, action, reward |
| $\gamma$ | discount factor (chapter 11 only; elsewhere $\gamma$ is the steering angle) |
| $V(s), Q(s, a)$ | state value and action value |
| $\pi$ | policy (the constant $\pi$ is never ambiguous in context) |

## Code

C++ lives in namespace `motion_planning`; Python imports it as `import motion_planning as mp`. A pose is an
`Eigen::Vector3d` in C++ and a length-3 NumPy array in Python. Trajectories are arrays with one pose per row.
Doxygen comments cite equations as `eq. (C.N)`, meaning equation $N$ of chapter $C$.
