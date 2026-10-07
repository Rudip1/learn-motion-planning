// Drive a kinematic bicycle to a goal pose and print the trajectory as CSV.
//
//   01_vehicle_kinematics [x_goal y_goal theta_goal_deg]   > trajectory.csv
#include <cstdlib>
#include <iostream>

#include "motion_planning/angles.hpp"
#include "motion_planning/pose_control.hpp"

int main(int argc, char** argv) {
    using namespace motion_planning;
    Pose goal(15.0, 10.0, -kPi / 2.0);
    if (argc == 4) goal = Pose(std::atof(argv[1]), std::atof(argv[2]), std::atof(argv[3]) * kPi / 180.0);

    const KinematicBicycle bike(/*L=*/1.2, /*max steering=*/35.0 * kPi / 180.0);
    RegulationOptions opt;
    opt.position_tolerance = 0.1;
    opt.heading_tolerance = 0.1;
    opt.speed.min_value = -5.0;
    opt.speed.max_value = 5.0;
    opt.speed.min_rate = -3.0;
    opt.speed.max_rate = 1.5;

    const RegulationResult r = regulate_pose(bike, Pose::Zero(), goal, PoseGains{}, opt);
    std::cout << "t,x,y,theta,v,gamma\n";
    for (Eigen::Index k = 0; k < r.inputs.rows(); ++k) {
        std::cout << k * opt.dt << ',' << r.states(k, 0) << ',' << r.states(k, 1) << ',' << r.states(k, 2)
                  << ',' << r.inputs(k, 0) << ',' << r.inputs(k, 1) << '\n';
    }
    std::cerr << (r.converged ? "reached" : "did not reach") << " the goal in " << r.inputs.rows()
              << " steps\n";
    return r.converged ? EXIT_SUCCESS : EXIT_FAILURE;
}
