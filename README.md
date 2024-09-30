# Inverse Kinematics on Guiding Vector Fields for Robot Path Following

## Research paper

Inverse kinematics is a fundamental technique for
motion and positioning control in robotics, typically applied
to end-effectors. In this paper, we extend the concept of
inverse kinematics to guiding vector fields for path following
in autonomous mobile robots. The desired path is defined by
its implicit equation, i.e., by a collection of points belonging to
one or more level sets. These level sets serve as a reference to
construct an error signal that drives the guiding vector field
toward the desired path, enabling the robot to converge and
travel along the path by following such vector field. We leverage
inverse kinematics to ensure that the constructed error signal
behaves like a linear system, facilitating control over the robot’s
transient motion toward the desired path and allowing for
the injection of feed-forward signals to induce precise motion-
behavior along the path. We start with the formal exposition
on how inverse kinematics can be applied to single-integrator
robots in an m-dimensional Euclidean space. We then propose
solutions to the theoretical and practical challenges of applying
this technique to unicycles with constant speeds to follow 2D
paths with precise transient control. We finish by validating the
formal results with experiments using fixed-wing drones.

    @misc{yuzhoujesusbautista2024ikgvf,
      title={Inverse Kinematics on Guiding Vector Fields for Robot Path Following}, 
      author={Yu Zhou, Jesus Bautista, Weijia Yao, Hector Garcia de Marina},
      year={2024},
      url={}, 
    }

This paper has been submitted to the IEEE International Conference on Robotics and Automation (ICRA) 2025.

## Features
This project includes:

* Visualization of Paparazzi UAV telemetry data from experiments
* Simulations of the Inverse Kinematics GVF algorithm
* Animations of both telemetry and simulation data

## Quick Install

To install, simply run:

```bash
python install.py
```

## Usage

For an overview of the project's structure and to see the code in action, we recommend running the Jupyter notebooks inside the `notebooks` directory.

## Credits

This repository is maintained by [Jesús Bautista Villar](https://sites.google.com/view/jbautista-research). For inquiries, suggestions, or further information, feel free to contact him at <jesbauti20@gmail.com>.
