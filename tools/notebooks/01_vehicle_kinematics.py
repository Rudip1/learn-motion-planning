# %% [markdown]
# # 1 · Vehicle kinematic models
#
# Run the unicycle, differential-drive and bicycle models from the C++ library, integrate them, see
# non-holonomy at work, and drive a model to a pose.

# %%
# Installs the module when it is missing (e.g. on Colab); a local `pip install -e .` is used as is.
import importlib.util

if importlib.util.find_spec("motion_planning") is None:
    %pip install -q git+https://github.com/Rudip1/learn-motion-planning

# %% [markdown]
# ## Recap
#
# Theory: [`1_theory/01_vehicle_kinematics.md`](../../1_theory/01_vehicle_kinematics.md).
#
# - A pose is $q = (x, y, \theta)$ of a reference point; a kinematic model is $\dot q = f(q, u)$.
# - Unicycle (1.2): $\dot x = v\cos\theta$, $\dot y = v\sin\theta$, $\dot\theta = \omega$.
# - Differential drive (1.3): $v = r(\omega_R + \omega_L)/2$, $\omega = r(\omega_R - \omega_L)/b$.
# - Rear-axle bicycle (1.7): a unicycle with $\omega = v\tan\gamma / L$; turning radius $R = L/\tan\gamma$ (1.5).
# - None of them can move sideways (1.10), but a loop of inputs moves them along the Lie bracket (1.11)–(1.12).
# - Euler, midpoint and RK4 have global error $O(\Delta t)$, $O(\Delta t^2)$, $O(\Delta t^4)$; for constant
#   inputs the unicycle has the exact arc solution (1.16).
# - The polar law (1.21) drives a unicycle to a pose iff $k_\rho > 0$, $k_\beta < 0$, $k_\alpha > k_\rho$ (1.23).

# %%
import matplotlib.pyplot as plt
import numpy as np

import motion_planning as mp
from motion_planning.plotting import SERIES, draw_vehicle, mark_pose, plot_path, use_style

use_style()
rng = np.random.default_rng(1)

# %% [markdown]
# ## Three models, one input profile
#
# Drive each model for 8 s. The unicycle gets $(v, \omega)$ directly, the differential drive gets the wheel
# speeds that produce the same $(v, \omega)$ through eq. (1.4), and the bicycle gets the steering angle that
# produces the same yaw rate through eq. (1.24). All three trace the same path — the inputs are just written
# in different coordinates.

# %%
dt = 0.05
t = np.arange(0, 8, dt)
v = 0.6 + 0.2 * np.sin(0.5 * t)
w = 0.8 * np.sin(0.7 * t)
body = np.c_[v, w]

uni = mp.Unicycle()
dd = mp.DifferentialDrive(wheel_radius=0.033, track_width=0.16)
bike = mp.KinematicBicycle(wheelbase=0.3, max_steering=0.9)

wheels = np.array([dd.body_to_wheel(u) for u in body])
steer = np.array([[vi, bike.steering_for(vi, wi)] for vi, wi in body])

q0 = np.zeros(3)
paths = {
    "unicycle (v, ω)": uni.simulate(q0, body, dt),
    "differential drive (ω_L, ω_R)": dd.simulate(q0, wheels, dt),
    "bicycle (v, γ)": bike.simulate(q0, steer, dt),
}
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4))
for (name, traj), lw in zip(paths.items(), [5, 3, 1.5]):
    ax.plot(traj[:, 0], traj[:, 1], lw=lw, label=name)
ax.set_aspect("equal")
ax.set_title("Same motion, three input spaces")
ax.legend()
ax2.plot(t, wheels[:, 0], label="ω_L [rad/s]")
ax2.plot(t, wheels[:, 1], label="ω_R [rad/s]")
ax2.plot(t, np.degrees(steer[:, 1]), label="γ [deg]")
ax2.set_xlabel("t [s]")
ax2.set_title("The inputs each model needed")
ax2.legend()
plt.show()

print("max difference between paths:",
      max(np.abs(p - paths["unicycle (v, ω)"]).max() for p in paths.values()))

# %% [markdown]
# The worked examples of the theory file, recomputed:

# %%
car = mp.KinematicBicycle(wheelbase=2.5, max_steering=0.7)
gamma = np.radians(15)
print(f"R = {car.turning_radius(gamma):.3f} m,  yaw rate at 10 m/s = {car.yaw_rate(10.0, gamma):.3f} rad/s")
print("differential drive (ω_L, ω_R) = (5, 10) rad/s  ->  (v, ω) =", dd.wheel_to_body([5.0, 10.0]))

# %% [markdown]
# ### ✏️ Exercise 1 — the bicycle from its geometry
#
# Using only the geometry of the steady turn (section 1.4), write the turning radius of the rear axle and the
# yaw rate of the rear-axle bicycle as functions of $L$, $\gamma$ and $v$. Handle $\gamma = 0$.

# %% [solution]
def turning_radius(L, gamma):
    return np.inf if np.tan(gamma) == 0 else L / np.tan(gamma)


def yaw_rate(L, v, gamma):
    return v * np.tan(gamma) / L

# %% [exercise]
def turning_radius(L, gamma):
    ...  # ✏️ eq. (1.5)


def yaw_rate(L, v, gamma):
    ...  # ✏️ eq. (1.6)

# %%
for L, v, g in [(2.5, 10.0, np.radians(15)), (0.3, 0.5, -0.4), (1.0, 2.0, 0.0)]:
    b = mp.KinematicBicycle(L, 0.7)
    assert np.isclose(turning_radius(L, g), b.turning_radius(g)) or np.isinf(b.turning_radius(g))
    assert np.isclose(yaw_rate(L, v, g), b.yaw_rate(v, g))
print("✓ bicycle geometry")

# %% [markdown]
# ## Integration
#
# ### ✏️ Exercise 2 — the exact unicycle step
#
# Implement eq. (1.16): the pose after `dt` seconds of constant $(v, \omega)$. Treat
# $|\omega\,\Delta t| < 10^{-9}$ as a straight line.

# %% [solution]
def exact_step(q, u, dt):
    x, y, th = q
    v, w = u
    if abs(w * dt) < 1e-9:
        return np.array([x + v * dt * np.cos(th), y + v * dt * np.sin(th), th + w * dt])
    th1 = th + w * dt
    return np.array([x + v / w * (np.sin(th1) - np.sin(th)), y - v / w * (np.cos(th1) - np.cos(th)), th1])

# %% [exercise]
def exact_step(q, u, dt):
    ...  # ✏️ eq. (1.16)

# %%
for _ in range(200):
    q = rng.uniform(-3, 3, 3)
    u = rng.uniform(-2, 2, 2) * [1, rng.choice([0.0, 1.0])]
    h = rng.uniform(0.01, 3)
    assert np.allclose(exact_step(q, u, h), mp.unicycle_exact_step(q, u, h), atol=1e-9)
print("✓ exact step")

# %% [markdown]
# ### ✏️ Exercise 3 — measure the order of each integrator
#
# For $u = (1, 1)$ over $T = 4.5$ s, compute the final-pose error of each integrator against the exact step for
# $n = 16, 32, \dots, 512$ steps, and estimate the order $p$ as the slope of $\log(\text{error})$ against
# $\log \Delta t$. Store the three slopes in a dict `order` keyed by `mp.Integrator`.

# %% [solution]
T, u = 4.5, np.array([1.0, 1.0])
exact = mp.unicycle_exact_step(np.zeros(3), u, T)
steps = np.array([16, 32, 64, 128, 256, 512])
order = {}
for method in [mp.Integrator.Euler, mp.Integrator.Midpoint, mp.Integrator.RK4]:
    err = [np.linalg.norm(uni.simulate(np.zeros(3), np.tile(u, (n, 1)), T / n, method)[-1] - exact)
           for n in steps]
    order[method] = np.polyfit(np.log(T / steps), np.log(err), 1)[0]

# %% [exercise]
T, u = 4.5, np.array([1.0, 1.0])
exact = mp.unicycle_exact_step(np.zeros(3), u, T)
steps = np.array([16, 32, 64, 128, 256, 512])
order = {}
# ✏️ fill `order`

# %%
assert abs(order[mp.Integrator.Euler] - 1) < 0.1
assert abs(order[mp.Integrator.Midpoint] - 2) < 0.1
assert abs(order[mp.Integrator.RK4] - 4) < 0.2
print("✓ orders", {k.name: round(float(v), 2) for k, v in order.items()})

# %% [markdown]
# ### 🔨 Break it — test the integrator on a full circle
#
# Repeat the measurement with $T = 2\pi$ (exactly one revolution). Every scheme now looks perfect: the errors
# are round-off. Each step of every scheme here is the same rigid motion, so after $n$ steps of $2\pi/n$ the
# polygon closes by symmetry. A test that only checks the final pose of a full turn cannot find the bug it is
# meant to find.

# %%
T_bad = 2 * np.pi
exact_bad = mp.unicycle_exact_step(np.zeros(3), u, T_bad)
for method in [mp.Integrator.Euler, mp.Integrator.RK4]:
    err = np.linalg.norm(uni.simulate(np.zeros(3), np.tile(u, (12, 1)), T_bad / 12, method)[-1] - exact_bad)
    print(f"{method.name:>5}: final error after one full circle in 12 steps = {err:.1e}")

# %% [markdown]
# ## Non-holonomy
#
# Along any trajectory of the unicycle the rolling constraint (1.10) holds exactly:

# %%
traj = paths["unicycle (v, ω)"]
qdot = np.array([uni.derivative(q, u) for q, u in zip(traj[:-1], body)])
res = [mp.nonholonomic_residual(q, qd) for q, qd in zip(traj[:-1], qdot)]
print("max |x' sinθ - y' cosθ| along the path:", np.max(np.abs(res)))

# %% [markdown]
# ### ✏️ Exercise 4 — parallel parking along the Lie bracket
#
# Starting from `q_start`, apply four exact flows of length `eps`: drive forwards, turn left, drive backwards,
# turn right (use `mp.unicycle_exact_step` with $u = (\pm1, 0)$ or $(0, \pm1)$). Return the final pose. Then
# check eq. (1.12): the displacement divided by $\varepsilon^2$ tends to $[g_1, g_2]$.

# %% [solution]
def parking_manoeuvre(q_start, eps):
    q = np.asarray(q_start, dtype=float)
    for u in ([1, 0], [0, 1], [-1, 0], [0, -1]):
        q = mp.unicycle_exact_step(q, np.array(u, dtype=float), eps)
    return q

# %% [exercise]
def parking_manoeuvre(q_start, eps):
    ...  # ✏️ four exact flows

# %%
q_start = np.array([1.0, 2.0, 0.7])
for eps in [0.3, 0.1, 0.03, 0.01]:
    d = (parking_manoeuvre(q_start, eps) - q_start) / eps**2
    print(f"eps = {eps:<5} displacement / eps² = {np.round(d, 4)}")
assert np.allclose((parking_manoeuvre(q_start, 1e-3) - q_start) / 1e-6, mp.unicycle_lie_bracket(q_start), atol=2e-3)
assert np.allclose(parking_manoeuvre([0, 0, 0], 0.1), [0.1 * (1 - np.cos(0.1)), -0.1 * np.sin(0.1), 0])
print("✓ bracket direction", mp.unicycle_lie_bracket(q_start))

# %% [markdown]
# Repeating the loop moves the vehicle sideways, one small step at a time — this is how a car parallel-parks.

# %%
fig, ax = plt.subplots(figsize=(4, 5))
q = np.zeros(3)
eps = 0.5
for k in range(4):
    for u in ([1, 0], [0, 1], [-1, 0], [0, -1]):
        seg = np.array([mp.unicycle_exact_step(q, np.array(u, float), s) for s in np.linspace(0, eps, 15)])
        ax.plot(seg[:, 0], seg[:, 1], color=SERIES[0] if u[0] > 0 else SERIES[1], lw=1.5)
        q = seg[-1]
    draw_vehicle(ax, q, 0.2, 0.1, color=SERIES[2], filled=True)
ax.plot([], [], color=SERIES[0], label="forwards")
ax.plot([], [], color=SERIES[1], label="backwards")
ax.set_aspect("equal")
ax.set_title(f"Four loops, ε = {eps}:\nnet motion is sideways (−y)")
ax.legend(loc="lower left")
plt.show()

# %% [markdown]
# ### 🔨 Break it — the wrong reference point
#
# Put the reference point half way between the axles ($l_r = L/2$). Its velocity now has the slip angle $\beta$
# of eq. (1.8), and the constraint (1.10) fails by exactly $-v\sin\beta$. A planner that assumes (1.10) for this
# point plans motions the car cannot do.

# %%
mid = mp.KinematicBicycle(wheelbase=2.0, max_steering=0.6, rear_to_reference=1.0)
q, u = np.array([0.0, 0.0, 0.4]), np.array([3.0, 0.5])
r = mp.nonholonomic_residual(q, mid.derivative(q, u))
print(f"residual = {r:.4f},   -v sin(beta) = {-u[0] * np.sin(mid.slip_angle(u[1])):.4f}")

# %% [markdown]
# ## Driving to a pose
#
# The polar law (1.21) on a car-like vehicle with a steering limit and acceleration limits:

# %%
car = mp.KinematicBicycle(wheelbase=1.2, max_steering=np.radians(35))
opts = mp.RegulationOptions(dt=0.05, position_tolerance=0.1, heading_tolerance=0.1,
                            speed=mp.RateLimits(-5.0, 5.0, -3.0, 1.5))
goal = np.array([15.0, 10.0, -np.pi / 2])
r = mp.regulate_pose(car, np.zeros(3), goal, mp.PoseGains(0.5, 1.5, -0.6), opts)
print("converged:", r.converged, "in", len(r.inputs) * opts.dt, "s;  final pose", np.round(r.states[-1], 3))

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
plot_path(ax, r.states, every=25, vehicle_length=1.2, label="rear axle")
mark_pose(ax, goal, size=1.5, label="goal")
ax.legend()
ax.set_title("Bicycle driven to a pose")
tt = np.arange(len(r.inputs)) * opts.dt
ax2.plot(tt, r.inputs[:, 0], label="v [m/s]")
ax2.plot(tt, np.degrees(r.inputs[:, 1]) / 10, label="γ [10 deg]")
ax2.set_xlabel("t [s]")
ax2.legend()
ax2.set_title("Applied inputs (rate-limited, saturated)")
plt.show()

# %% [markdown]
# ### ✏️ Exercise 5 — polar coordinates
#
# Implement eq. (1.19): return `(rho, alpha, beta)` for pose `q` and goal pose `goal`, with both angles wrapped
# to $[-\pi, \pi)$ (use `mp.wrap_angle`).

# %% [solution]
def polar(q, goal):
    dx, dy = goal[0] - q[0], goal[1] - q[1]
    rho = np.hypot(dx, dy)
    alpha = mp.wrap_angle(np.arctan2(dy, dx) - q[2])
    beta = mp.wrap_angle(goal[2] - q[2] - alpha)
    return rho, float(alpha), float(beta)

# %% [exercise]
def polar(q, goal):
    ...  # ✏️ eq. (1.19)

# %%
assert np.allclose(polar([0, 0, 0], [1, 1, np.pi / 2]), [np.sqrt(2), np.pi / 4, np.pi / 4])
for _ in range(500):
    q, g = rng.uniform(-5, 5, 3), rng.uniform(-5, 5, 3)
    e = mp.polar_error(q, g)
    assert np.allclose(polar(q, g), [e.rho, e.alpha, e.beta])
print("✓ polar coordinates")

# %% [markdown]
# ### ✏️ Exercise 6 — gains from the linearisation
#
# Write the $2\times2$ matrix of eq. (1.23) for the $(\alpha, \beta)$ subsystem and decide stability from its
# eigenvalues. The assert compares your answer with the closed-form conditions on 1000 random gain triples.

# %% [solution]
def linearised(k_rho, k_alpha, k_beta):
    return np.array([[k_rho - k_alpha, -k_beta], [-k_rho, 0.0]])


def stable(k_rho, k_alpha, k_beta):
    return k_rho > 0 and np.all(np.linalg.eigvals(linearised(k_rho, k_alpha, k_beta)).real < 0)

# %% [exercise]
def linearised(k_rho, k_alpha, k_beta):
    ...  # ✏️ eq. (1.23)


def stable(k_rho, k_alpha, k_beta):
    ...  # ✏️ use the eigenvalues

# %%
for _ in range(1000):
    k = rng.uniform(-2, 2, 3)
    assert stable(*k) == mp.PoseGains(*k).is_stable(), k
print("✓ stability region")

# %% [markdown]
# ### 🔨 Break it — gains, direction, steering limit
#
# 1. **$k_\beta > 0$.** The linearisation has a root in the right half-plane; the vehicle reaches the position
#    but not the heading.
# 2. **No reverse.** With the goal behind and `allow_reverse=False`, $|\alpha| > \pi/2$ and (1.20) no longer
#    describes the motion: the law first drives away from the goal.
# 3. **Euler with a coarse step.** Run the same regulation with a large `dt`; the law is fine, the simulation is
#    not.

# %%
uni_opts = mp.RegulationOptions(position_tolerance=0.02, heading_tolerance=0.05, max_steps=1500)
goal = np.array([0.0, 0.0, np.pi / 2])
cases = {
    "stable gains": (np.array([-3.0, -1.0, 0.0]), mp.PoseGains(0.5, 1.5, -0.6), True),
    "k_β > 0": (np.array([-3.0, -1.0, 0.0]), mp.PoseGains(0.5, 1.5, 0.6), True),
    "goal behind, reverse": (np.array([2.0, -1.0, 0.0]), mp.PoseGains(), True),
    "goal behind, no reverse": (np.array([2.0, -1.0, 0.0]), mp.PoseGains(), False),
}
fig, ax = plt.subplots(figsize=(7, 5))
for (name, (q0, gains, rev)), c in zip(cases.items(), SERIES):
    uni_opts.allow_reverse = rev
    res = mp.regulate_pose(mp.Unicycle(), q0, goal, gains, uni_opts)
    err = abs(mp.angle_difference(res.states[-1, 2], goal[2]))
    ax.plot(res.states[:, 0], res.states[:, 1], color=c,
            label=f"{name}: converged={res.converged}, heading error {np.degrees(err):.0f}°")
    draw_vehicle(ax, q0, 0.4, 0.2, color=c, filled=True)
mark_pose(ax, goal, size=0.6)
ax.set_aspect("equal")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1))
ax.set_title("What the stability conditions and the reverse rule buy")
plt.show()

# %% [markdown]
# ## What to remember
#
# - A kinematic model is $\dot q = f(q, u)$; the unicycle is the common core, other vehicles differ only in how
#   their inputs map to $(v, \omega)$.
# - The rear-axle midpoint cannot move sideways (1.10); sideways motion exists only through the Lie bracket,
#   at a cost of order $\varepsilon^2$ per manoeuvre — the reason non-holonomic planning (chapter 6) is hard.
# - Integrate with RK4 or exactly (1.16); test integrators against a closed form on a non-periodic horizon.
# - Wrap angle *differences*, never the integrated heading.
# - The polar pose law is stable iff $k_\rho > 0$, $k_\beta < 0$, $k_\alpha > k_\rho$, needs a reverse branch for
#   goals behind, and is discontinuous at the goal — as Brockett's theorem says it must be.
