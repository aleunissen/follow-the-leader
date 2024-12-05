import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import numpy as np

class JointStatePrinter(Node):
    def __init__(self):
        super().__init__('joint_state_printer')
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10)
        self.subscription  # Prevent unused variable warning
        self.joint_positions_printed = False  # Ensure the node only prints once        
        
        # Define the desired joint order
        self.desired_order = [
                "joint_1",
                "joint_2",
                "joint_3",
                "joint_4",
                "joint_5",
                "joint_6",
            ]        
        # self.desired_order = [
        #     "shoulder_pan_joint",
        #     "shoulder_lift_joint",
        #     "elbow_joint",
        #     "wrist_1_joint",
        #     "wrist_2_joint",
        #     "wrist_3_joint"
        # ]
    
    def joint_state_callback(self, msg: JointState):
        if not self.joint_positions_printed:
            # Extract joint names and positions from the message
            name_to_position = dict(zip(msg.name, msg.position))
            
            # Reorder positions based on the desired joint order
            ordered_positions = [
                name_to_position[joint] for joint in self.desired_order
            ]
            
            # Format the output
            formatted_positions = ', '.join(f"{pos:.5f}" for pos in ordered_positions)
            formatted_output = f"home_joints = [{formatted_positions}]"
            
            # Print the formatted output
            self.get_logger().info(formatted_output)         
           


def main(args=None):
    rclpy.init(args=args)
    node = JointStatePrinter()
    rclpy.spin_once(node)
    # Shut down the node
    rclpy.shutdown()

if __name__ == '__main__':
    main()
