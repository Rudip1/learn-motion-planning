# Chapter plan — learn-motion-planning   (`<pkg>` = `motion_planning`)

From robot models to paths, then from paths to decisions. Existing material to mine: vehicle-model notebook,
potential functions, graph search, RRT/RRT*, DWA and online RRT for a TurtleBot, Q-learning, PDDL domain.

**Part I — moving**
1. Vehicle kinematic models: unicycle, differential drive, kinematic bicycle; integration; non-holonomy.
2. Configuration space and occupancy grids: inflation, collision checking, distance transforms.
3. Potential fields: attractive/repulsive, brushfire and wavefront, local minima (break-it cell).
4. Graph search: BFS, Dijkstra, A*, admissible heuristics, weighted A*; test against hand-worked grids.
5. Sampling-based planning: PRM, RRT, RRT*; asymptotic optimality; compare against OMPL in tests.
6. Non-holonomic paths: Dubins curves (all six words), Dubins-RRT*.
7. Path smoothing and tracking: shortcutting, pure pursuit, Stanley.
8. Local planning: Dynamic Window Approach; why local planners need a global one.

**Part II — deciding**
9. Behaviour trees: sequence, fallback, decorators, blackboard; a small BT engine in C++.
10. Task planning: STRIPS/PDDL, forward search over a symbolic domain (planner in C++, domains in PDDL).
11. Reinforcement learning basics: MDPs, value iteration, tabular Q-learning on a grid world.

Used in: [lidar-exploration](https://github.com/Rudip1/lidar-exploration), [CA-MCW](https://github.com/Rudip1/CA-MCW).
Acknowledgement: chapter 11 grew out of joint work with Gebrecherkos Gebreslassie.
