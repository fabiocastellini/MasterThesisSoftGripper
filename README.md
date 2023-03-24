# Master's Degree in Computer Engineering for Robotics and Smart Industry
## Development and characterization of a 3D printed hemispherical soft tactile sensor for agricultural applications

> Graduand: _Fabio Castellini (VR464639)_ <br />
> Supervisor: _Prof. Riccardo Muradore_ <br />
> Co-supervisor: _Francesco Visentin, PhD_ <br />
> Academic Year: _2021-2022_


## _Abstract_
Soft robotics, and particularly soft gripping technology, faces many challenges. Due to several strict requirements such as small dimensions, low cost and efficient manufacturing process, accurate sensing capabilities, the development of soft grippers is still an open research problem. In this work a hemispherical deformable and cheap to manufacture tactile sensor is proposed and characterized. The device is 3D printed using a stereolithography (SLA) 3D printer and is made of a semi-transparent elastic polymer resin that is properly cured afterwards. The overall aim is to sense normal and tangential forces applied to the gripper. The gripper is designed and thought for agricultural applications such as grasping delicate fruits and vegetables.


## _Thesis Objectives_
In our work we focused on the development of a _cheap, 3D printed, marker-based hemispherical soft gripper_ capable of sensing forces when in contact with the external environment. 
The final goals of our work are to: 
- design and manufacture a soft tactile sensor with a suited pattern of fiducial markers to be tracked; 
- calibrate the sensing device in order to perform online estimation of both shear and normal forces; 
- design a structure that can hold the sensing device in place and mount it on the _Franka Emika Panda_ robot's end-effector 
- attempt a simple harvesting task exploiting the developed device within a force control loop on the gripper and an external depth camera mounted on the robotic arm.

---------

## _Guide for this repository_
This repository contains some of the developed scripts during my Thesis. Particularly, it contains the most important ROS scripts to perform the online strawberry picking exploiting the estimated force feedback.

- _panda_controller_ contains the PD controller for the 7-DoF Franka Emika Panda manipulator
- _easy_handeye_ is a ROS package used to perform eye-in-hand calibration (https://github.com/IFL-CAMP/easy_handeye)
- _darknet_ros_ is a ROS package used to exploit a pre-trained Yolo object detection Neural Network (https://github.com/leggedrobotics/darknet_ros)
- _tesi_gripper_ is a ROS package that contains the developed online pipeline to perform the picking of a ripe strawberry, exploiting RGBD camera's information and real-time force feedback

The "plant_detection_node.py" script can be summed up in the following Finite State Machine:
1) a 
2)

![Uploading pipeline.png…]()





