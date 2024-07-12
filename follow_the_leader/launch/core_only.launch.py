#!/usr/bin/env python3
import launch
from launch import LaunchDescription, LaunchContext
from launch.actions import (
    IncludeLaunchDescription,
    DeclareLaunchArgument,
    SetLaunchConfiguration,
    EmitEvent,
    OpaqueFunction,
)
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.conditions import IfCondition, UnlessCondition, LaunchConfigurationEquals
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node, LifecycleNode
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():    
    ur_type = LaunchConfiguration("ur_type")
    camera_type = LaunchConfiguration("camera_type", default="d435")

    package_dir = get_package_share_directory("follow_the_leader")
    params_path = os.path.join(package_dir, "config")

    # ==============
    # Non-simulation
    # ==============

    # Load the YAML config files
    core_yaml_path = PythonExpression(["'{}/ftl_{}.yaml'.format(r'", params_path, "', '", ur_type, "')"])
    camera_yaml_path = PythonExpression(["'{}/camera_{}.yaml'.format(r'", params_path, "', '", camera_type, "')"])
    core_params_file = LaunchConfiguration("core_params_file")
    camera_params_file = LaunchConfiguration("camera_params_file")

    # Load the YAML files
    package_dir = get_package_share_directory("follow_the_leader")
    core_params_path = os.path.join(package_dir, "config", "ftl_ur5e.yaml")
    camera_params_path = PythonExpression(["'{}'.format(r'", camera_params_file, "')"])

    params_arg = DeclareLaunchArgument(
        "core_params_file",
        default_value=core_yaml_path,
        description="Path to the YAML file containing node parameters",
    )
    camera_params_arg = DeclareLaunchArgument(
        "camera_params_file",
        default_value=camera_yaml_path,
        description="Path to the YAML file containing camera parameters",
    )

    state_manager_node = Node(
        package="follow_the_leader",
        executable="state_manager",
        output="screen",
        parameters=[core_params_file],
    )

    image_processor_node = Node(
        package="follow_the_leader",
        executable="image_processor",
        output="screen",
        parameters=[core_params_file, camera_params_file],
    )

    point_tracker_node = Node(
        package="follow_the_leader",
        executable="point_tracker",
        output="screen",
        parameters=[core_params_file, camera_params_file],
    )

    modeling_node = Node(
        package="follow_the_leader",
        executable="model",
        output="screen",
        parameters=[core_params_file, camera_params_file],
    )

    controller_node = Node(
        package="follow_the_leader",
        executable="controller_3d",
        # output='screen',
        parameters=[core_params_file],
    )

    servoing_node = Node(
        package="follow_the_leader",
        executable="visual_servoing",
        output="screen",
        parameters=[core_params_file],
    )    
    
    tf_node_b = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments="0.0 0 0.0 0.5 -0.5 0.5 0.5 tool0 camera_link".split()
    )

    tf_node_c = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments="0 0 0 0.5 -0.5 0.5 -0.5 camera_link camera_color_optical_frame".split()
    )

    return LaunchDescription(
        [
            params_arg,
            camera_params_arg,
            # state_manager_node,
            # image_processor_node,
            point_tracker_node,
            modeling_node,
            # controller_node,
            # servoing_node,
            # tf_node_b,
            # tf_node_c
        ]
    )
