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

The "plant_detection_node.py" script can be summed up in the following Finite State Machine: <br>
- **[State 1] Find plant**: once the node is initialized, the RGB image given by the Realsense2 camera is exploited by a pre-trained CNN (YoloV3) to localize a potted plant, that is our target. Starting from the geometric center of the 2D detected bounding box we compute its 3D coordinate exploiting the depth capabilities of the Realsense2 camera.
- **[State 2] Plant approach**: using the estimated baricenter of the strawberry plant as a reference, plan and execute a linear trajectory to approach a centered target cartesian point, so that the camera is able to have a closer view of the plant.
- **[State 3] Find ripe strawberry**: using a CNN (YoloV2) fine-tuned on a strawberry dataset, detect a target strawberry localizing it with a 2D bounding box, computing the 3D associated baricenter of the fruit and classifying it as ripe or not.
- **[State 4] Strawberry approach**: once a ripe strawberry has been detected, plan and execute a linear trajectory to position the manipular's gripper in a suitable manner, so that picking can be performed. This is done starting from the estimated 3D baricenter of the strawberry and assuming a normal displacement of 2cm with respect to the elastic domes.
- **[State 5] Strawberry picking**: at this point, real-time force feedback is provided through the K-Nearest Regressor trained model and the gripper closes until a certain threshold (manual tunings suggest 1.75N) of normal force is measured by the developed sensor. Afterwards, the strawberry is harvested from the plant by moving the gripper back, while slightly rotating it towards the ground.


<p align="center">
  <img src="https://user-images.githubusercontent.com/76775232/227550597-063986f5-8253-4870-8272-8c18b44be32f.png" alt="pipeline" width="700"/>
</p>





