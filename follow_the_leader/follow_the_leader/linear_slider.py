#!/usr/bin/env python3

"""
header:
  stamp:
    sec: 1701123797
    nanosec: 663113284
  frame_id: tool0
twist:
  linear:
    x: -0.0005139466833269264
    y: -0.14999897028036757
    z: 0.0002115978168702471
  angular:
    x: 0.0
    y: 0.0
    z: 0.0
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor, ExternalShutdownException
from rclpy.parameter import Parameter
from std_msgs.msg import Float32
from geometry_msgs.msg import TwistStamped

import socket
import json
import sys

class LinearSliderNode(Node):
    def __init__(self) -> None:
        super().__init__(node_name="linear_slider_node")

        # Network parameters
        self.SERVER_HOST = "0.0.0.0"
        self.SERVER_PORT = 8888
        self.clearcore_controller_ip = self.declare_parameter("clearcore_controller_ip", '169.254.97.177')
        self.clearcore_controller_port = self.declare_parameter("clearcore_controller_port", 8888)

        self.CLEARCORE_HOST = self.clearcore_controller_ip.get_parameter_value().string_value
        self.CLEARCORE_PORT = self.clearcore_controller_port.get_parameter_value().integer_value

        # Linear slider actual velocity
        self.current_velocity_publisher = self.create_publisher(
            msg_type=Float32,
            topic="/linear_slider_current_velocity",
            qos_profile=1
        )
        self.current_velocity_timer_period = 0.00001
        self.current_velocity_publisher_timer = self.create_timer(
            timer_period_sec=self.current_velocity_timer_period,
            callback=self.current_velocity_publisher_timer_cb
        )

        # Server
        self.current_velocity_publisher_server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP specifier
        self.current_velocity_publisher_server_socket.bind((self.SERVER_HOST, self.SERVER_PORT))

        # Linear slider target velocity
        self.target_velocity_publisher = self.create_publisher(
            msg_type=Float32,
            topic="/linear_slider_target_velocity",
            qos_profile=1
        )
        # self.target_velocity_timer_period = 0.00001
        # self.target_velocity_publisher_timer = self.create_timer(
        #     timer_period_sec=self.target_velocity_timer_period,
        #     callback=self.target_velocity_publisher_timer_cb
        # )
        
        # Client
        self.target_velocity_publisher_client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Twist command subscriber
        self.target_velocity_subscriber = self.create_subscription(
            msg_type=TwistStamped,
            topic='/servo_node/delta_twist_cmds',
            callback=self.target_velocity_subscriber_cb,
            qos_profile=1
        )
        return

    def current_velocity_publisher_timer_cb(self) -> None:
        """
        Callback method for the current velocity publisher timer of the linear slider
        Receives byte data from the ClearCore controller via the socket server and publishes to the topic.
        """
        msg = Float32()
        try:
            raw_data, addr = self.current_velocity_publisher_server_socket.recvfrom(1024)
            data: dict = json.loads(raw_data)
            msg.data = float(data["servo_velocity"])
            self.current_velocity_publisher.publish(msg=msg)
        except ValueError as e:
            print(f"{e}: Could not convert msg type to float.")
        
        # self.get_logger().info(f"Linear slider current velocity: {msg.data}")
        return
    
    # def target_velocity_publisher_timer_cb(self) -> None:
    #     """
    #     Callback method for the target velocity publisher timer of the linear slider.
    #     Sends byte data from the ClearCore controller via the socket client and publishes to the topic.
    #     """
    #     msg = Float32()

    #     self.target = 0.0

    #     # Publish data
    #     msg.data = self.target
    #     self.target_velocity_publisher.publish(msg=msg)

    #     # self.get_logger().info(f"Linear slider target velocity: {msg.data}")

    #     # Send data to the ClearCore controller
    #     self.target_velocity_publisher_client_socket.sendto(
    #         f"{msg.data}".encode(),
    #         (self.CLEARCORE_HOST, self.CLEARCORE_PORT)
    #     )
    #     return
    
    def target_velocity_subscriber_cb(self, msg):
        """
        Callback method for the target velocity subscriber to get the target velocity from the 3D Controller
        TODO: ADD SAFETY STOPS! When the scan ends or gets paused, the /delta_twist_cmds does not send out a 0.0
        
        """
        pub_msg = Float32()
        pub_msg.data = msg.twist.linear.x
        
        # Publish data
        self.target_velocity_publisher.publish(msg=pub_msg)

        # Convert data to RPM and send to the ClearCore controller
        # 1 cm per 2 rotations = 1 m per 200 rotations
        # data is in m/s
        rpm_speed = pub_msg.data * 60 / 200

        self.target_velocity_publisher_client_socket.sendto(
            f"{rpm_speed}".encode(),
            (self.CLEARCORE_HOST, self.CLEARCORE_PORT)
        )
        self.get_logger().info(f"RPM speed: {rpm_speed}")
        return


def main():
    rclpy.init()

    executor = MultiThreadedExecutor()

    linear_slider = LinearSliderNode()

    rclpy.spin(node=linear_slider, executor=executor)

    return


if __name__ == "__main__":
    main()