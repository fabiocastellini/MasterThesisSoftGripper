#include "ros/ros.h"
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/PoseArray.h"
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/WrenchStamped.h>
#include <tf/transform_listener.h>
#include <math.h>
#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Matrix3x3.h"
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <iostream>
#include <string>
#include <Eigen/Dense>
#include <tf_conversions/tf_eigen.h>
#include <std_msgs/Int32.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/String.h>
#include <std_srvs/Empty.h>

geometry_msgs::PoseStamped curr_pose;
geometry_msgs::PoseArray target_pose;
geometry_msgs::PoseStamped target_pose_stamped;
geometry_msgs::TwistStamped curr_twist;
geometry_msgs::WrenchStamped out_des_wrench;


std::string frame_id = "/panda_link0";
bool has_pose_data = false;
bool has_twist_data = false;
bool has_ref_data = false;
int counter_traj = 0;
bool go_controller = false;

bool togglePlay(std_srvs::Empty::Request  &req,
         std_srvs::Empty::Request &res)
{
    go_controller =! go_controller;
    ROS_INFO("I've received your input from service!");
    return true;
}
Eigen::Matrix<double, 6, 1> low_pass(Eigen::Matrix<double, 6, 1> new_value, Eigen::Matrix<double, 6, 1> prev_value, double dt, double cut_frequency)
{
    Eigen::Matrix<double, 6, 1> ret_value = prev_value + (new_value - prev_value) * (1.0 - exp(-dt * 2.0 * M_PI * cut_frequency));
    return ret_value;
}

geometry_msgs::PoseStamped pose2poseStamped(geometry_msgs::Pose pose, std_msgs::Header header)
{
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.frame_id = header.frame_id;
    pose_stamped.header.stamp = header.stamp;
    pose_stamped.pose.position.x = pose.position.x;
    pose_stamped.pose.position.y = pose.position.y;
    pose_stamped.pose.position.z = pose.position.z;
    pose_stamped.pose.orientation.x = pose.orientation.x;
    pose_stamped.pose.orientation.y = pose.orientation.y;
    pose_stamped.pose.orientation.z = pose.orientation.z;
    pose_stamped.pose.orientation.w = pose.orientation.w;

    return pose_stamped;
}
Eigen::Matrix<double, 4, 4> pose2eigen(geometry_msgs::PoseStamped pose)
{
    Eigen::Matrix<double, 4, 4> T = Eigen::Matrix<double, 4, 4>::Identity();

    T(0, 3) = pose.pose.position.x;
    T(1, 3) = pose.pose.position.y;
    T(2, 3) = pose.pose.position.z;

    Eigen::Quaterniond q(pose.pose.orientation.w, pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z);

    T.block<3, 3>(0, 0) = q.normalized().toRotationMatrix();

    return T;
}

double rotationBetween(double from, double to)
{
    return (from > to) ? -M_PI + std::fmod(from - to + M_PI, M_PI * 2)
                       :  M_PI - std::fmod(to - from + M_PI, M_PI * 2);
}

void wrapTo2Pi(double &a)
        {
            bool was_neg = a < 0;
            a = fmod(a, 2.0 * M_PI);
            if (was_neg)
            {
                a += 2.0 * M_PI;
            }
        }
Eigen::Matrix<double, 6, 1> compute_pose_error(Eigen::Matrix<double, 4, 4> T_des, Eigen::Matrix<double, 4, 4> T)
{
    Eigen::Matrix<double, 6, 1> err;

    err.block<3, 1>(0, 0) = T_des.block<3, 1>(0, 3) - T.block<3, 1>(0, 3);

    Eigen::Quaterniond orientation_quat_des = Eigen::Quaterniond(T_des.block<3, 3>(0, 0));
    Eigen::Quaterniond orientation_quat = Eigen::Quaterniond(T.block<3, 3>(0, 0));

    auto orientation_quat_des_euler = orientation_quat_des.toRotationMatrix().eulerAngles(2,1,2);
    auto orientation_quat_euler = orientation_quat.toRotationMatrix().eulerAngles(2,1,2);

    // wrapTo2Pi(orientation_quat_des_euler[0]);
    // wrapTo2Pi(orientation_quat_des_euler[1]);
    // wrapTo2Pi(orientation_quat_des_euler[2]);
    // wrapTo2Pi(orientation_quat_euler[0]);
    // wrapTo2Pi(orientation_quat_euler[1]);
    // wrapTo2Pi(orientation_quat_euler[2]);

    err[3] = rotationBetween(orientation_quat_euler[0], orientation_quat_des_euler[0]);
    err[4] = rotationBetween(orientation_quat_euler[1], orientation_quat_des_euler[1]);
    err[5] = rotationBetween(orientation_quat_euler[2], orientation_quat_des_euler[2]);

    std::cout << "orientation_quat_des_euler " << orientation_quat_des_euler << std::endl;


    //if (orientation_quat_des.coeffs().dot(orientation_quat.coeffs()) < 0.0)
    //{
    //    orientation_quat.coeffs() << -orientation_quat.coeffs();
    //}

    //Eigen::Quaterniond orientation_quat_error(orientation_quat.inverse() * orientation_quat_des);

    // err.block<3, 1>(3, 0) << orientation_quat_des_euler - orientation_quat_euler;
    //err.block<3, 1>(3, 0) << -T.block<3, 3>(0, 0) * err.block<3, 1>(3, 0);

    return err;
}

void currentPoseCb(const geometry_msgs::PoseStampedConstPtr &msg)
{
    curr_pose = *msg;
    has_pose_data = true;
}

void currentTwistCb(const geometry_msgs::TwistStampedConstPtr &msg)
{
    curr_twist = *msg;
    has_twist_data = true;
}

void smoothTrajectory(const geometry_msgs::PoseArrayConstPtr &msg)
{
    target_pose = *msg;
    has_ref_data = true;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "PD_controller");

    ros::NodeHandle public_node_handler;
    

    ros::Publisher out_des_wrench_pub = public_node_handler.advertise<geometry_msgs::WrenchStamped>("/panda_force_ctrl/desired_wrench", 1);
    ros::Publisher out_twist_pub = public_node_handler.advertise<geometry_msgs::TwistStamped>("/panda_force_ctrl/tracked_twist", 1);
    ros::Publisher out_variable_k_pub = public_node_handler.advertise<std_msgs::Float32MultiArray>("params", 1);
    ros::Publisher desired_reference_pub = public_node_handler.advertise<geometry_msgs::PoseStamped>("/panda_force_ctrl/reference", 1);
    ros::Subscriber curr_panda_pos_sub = public_node_handler.subscribe<geometry_msgs::PoseStamped>("/panda_force_ctrl/current_pose", 1, &currentPoseCb);
    ros::Subscriber curr_panda_twist_sub = public_node_handler.subscribe<geometry_msgs::TwistStamped>("/panda_force_ctrl/current_twist", 1, &currentTwistCb);
    ros::Subscriber trajectory = public_node_handler.subscribe<geometry_msgs::PoseArray>("/smooth_trajectory", 1, &smoothTrajectory);
    ros::ServiceServer server = public_node_handler.advertiseService("toggle_play",&togglePlay);

    float frequency = 1000;
    //public_node_handler.param<float>("frequency", frequency, 0);

    ros::Rate loop_rate(frequency);
    
    tf::TransformBroadcaster br;
    Eigen::Matrix4d targetEETf, currentEETf;
    Eigen::Matrix<double, 6, 1> twistEETf, twistEETf_filt;
    twistEETf_filt.setZero();

    Eigen::Matrix<double, 6, 1> kp;
    Eigen::Matrix<double, 6, 1> kd;

    std::vector<double> k_gains_;
    std::vector<double> d_gains_;
    if (!public_node_handler.getParam("k_gains", k_gains_) || k_gains_.size() != 6) {
    ROS_ERROR(
        "JointImpedanceExampleController:  Invalid or no k_gain parameters provided, aborting "
        "controller init!");
    return false;
    }

    if (!public_node_handler.getParam("d_gains", d_gains_) || d_gains_.size() != 6) {
    ROS_ERROR(
        "JointImpedanceExampleController:  Invalid or no k_gain parameters provided, aborting "
        "controller init!");
    return false;
    }

    kp(0,0)= k_gains_[0];
    kp(1,0)= k_gains_[1];
    kp(2,0)= k_gains_[2];
    kp(3,0)= k_gains_[3];
    kp(4,0)= k_gains_[4];
    kp(5,0)= k_gains_[5];

    kd(0,0)= d_gains_[0];
    kd(1,0)= d_gains_[1];
    kd(2,0)= d_gains_[2];
    kd(3,0)= d_gains_[3];
    kd(4,0)= d_gains_[4];
    kd(5,0)= d_gains_[5];

    while (ros::ok())
    {
        if (has_pose_data && has_twist_data && has_ref_data && go_controller)
        {   
            int len = target_pose.poses.size();
            
            if(counter_traj < (target_pose.poses.size())){
                ROS_INFO("I'm sending way point %i of %i",counter_traj, len);
                currentEETf = pose2eigen(curr_pose);
                target_pose_stamped = pose2poseStamped(target_pose.poses[counter_traj],target_pose.header);
                targetEETf = pose2eigen(target_pose_stamped);           

                tf::Transform des;
                des.setOrigin(tf::Vector3(target_pose.poses[counter_traj].position.x, target_pose.poses[counter_traj].position.y, target_pose.poses[counter_traj].position.z));
                des.setRotation(tf::Quaternion(target_pose.poses[counter_traj].orientation.x, target_pose.poses[counter_traj].orientation.y, target_pose.poses[counter_traj].orientation.z, target_pose.poses[counter_traj].orientation.w));
                br.sendTransform(tf::StampedTransform(des, ros::Time::now(), "/panda_link0", "/desired"));

                twistEETf[0] = curr_twist.twist.linear.x;
                twistEETf[1] = curr_twist.twist.linear.y;
                twistEETf[2] = curr_twist.twist.linear.z;
                twistEETf[3] = curr_twist.twist.angular.x;
                twistEETf[4] = curr_twist.twist.angular.y;
                twistEETf[5] = curr_twist.twist.angular.z;     

                twistEETf_filt = low_pass(twistEETf, twistEETf_filt, 1.0 / frequency, 50.0f);           
                
                //Eigen::Matrix<double, 6, 1> kp;
                //Eigen::Matrix<double, 6, 1> kd;
                //kp << 1000, 1000, 1000, 350, 350, 350;
                // kp << 1000, 900, 900, 350, 350, 350; // initial
                
                //kd << 10, 10, 10, 1, 1, 1;          
                //kd << 30, 30, 30, 3, 3, 3;          // initial
                Eigen::Matrix<double, 6, 1> tracking_err = compute_pose_error(targetEETf, currentEETf);
                Eigen::Matrix<double, 6, 6> Kp = kp.asDiagonal();
                Eigen::Matrix<double, 6, 6> Kd = kd.asDiagonal();           
                Eigen::Matrix<double, 6, 1> force_cmd = (Kp * tracking_err) - Kd * twistEETf_filt;

                out_des_wrench.header.stamp = ros::Time::now();
                out_des_wrench.header.frame_id = frame_id;
                out_des_wrench.wrench.force.x = force_cmd[0];
                out_des_wrench.wrench.force.y = force_cmd[1];
                out_des_wrench.wrench.force.z = force_cmd[2];
                out_des_wrench.wrench.torque.x = force_cmd[3];
                out_des_wrench.wrench.torque.y = force_cmd[4];
                out_des_wrench.wrench.torque.z = force_cmd[5];
                out_des_wrench_pub.publish(out_des_wrench);

                curr_twist.header.stamp = ros::Time::now();
                curr_twist.twist.linear.x = twistEETf_filt[0];
                curr_twist.twist.linear.y = twistEETf_filt[1];
                curr_twist.twist.linear.z = twistEETf_filt[2];
                curr_twist.twist.angular.x = twistEETf_filt[3];
                curr_twist.twist.angular.y = twistEETf_filt[4];
                curr_twist.twist.angular.z = twistEETf_filt[5];
                out_twist_pub.publish(curr_twist);

                desired_reference_pub.publish(target_pose_stamped);
                std_msgs::Float32MultiArray param_msg;
                param_msg.data.resize(12, 0.0);         
                param_msg.data[0] = kp[0];
                param_msg.data[1] = kp[1];
                param_msg.data[2] = kp[2];
                param_msg.data[3] = kp[3];
                param_msg.data[4] = kp[4];
                param_msg.data[5] = kp[5];          
                param_msg.data[6] = kd[0];
                param_msg.data[7] = kd[1];
                param_msg.data[8] = kd[2];
                param_msg.data[9] = kd[3];
                param_msg.data[10] = kd[4];
                param_msg.data[11] = kd[5];         
                out_variable_k_pub.publish(param_msg);

                counter_traj++;

            }else{
                out_des_wrench.header.frame_id = frame_id;
                out_des_wrench.header.stamp = ros::Time::now();
                out_des_wrench.wrench.force.x = 0.0;
                out_des_wrench.wrench.force.y = 0.0;
                out_des_wrench.wrench.force.z = 0.0;
                out_des_wrench.wrench.torque.x = 0.0;
                out_des_wrench.wrench.torque.y = 0.0;
                out_des_wrench.wrench.torque.z = 0.0;
                out_des_wrench_pub.publish(out_des_wrench); 
                counter_traj = 0;
                has_ref_data = false;
                go_controller = false;
            }
        
        }

        ros::spinOnce();
        if (!loop_rate.sleep())
        {
            ROS_WARN("[Autonomous Panda] Unable to keep the update frequency");
        }
    }

    return 0;
}

