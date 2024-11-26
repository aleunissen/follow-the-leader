import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
import rclpy_message_converter.message_converter as msgconvert
import yaml
from ament_index_python.packages import get_package_share_directory


class CameraInfoLowNode(Node):
    def __init__(self):
        super().__init__('camera_info_low_node')
        

        self.camera_info_low_pub = self.create_publisher(CameraInfo, '/camera/color/camera_info_low', 10)
        
        self.pub_timer = self.create_timer(
            0.03, self.publish_camera_info_callback)


    def publish_camera_info_callback(self):
        camera_info_low_msg = self.load_camera_info_to_msg()
        camera_info_low_msg.header.stamp = self.get_clock().now().to_msg()
        self.camera_info_low_pub.publish(camera_info_low_msg)

    def load_camera_info_to_msg(self):
        pkg_folder = get_package_share_directory("follow_the_leader")
        yaml_string = self.load_yaml_string(pkg_folder+"/config/camera_info_d405_low.yaml")
        msgdict = yaml.load(yaml_string, Loader=yaml.FullLoader)
        msg = msgconvert.convert_dictionary_to_ros_message(CameraInfo,msgdict)
        return msg 

    def load_yaml_string(self, file_path):
        with open(file_path, 'r') as file:
            yaml_string = file.read()
        return yaml_string




def main(args=None):
    rclpy.init(args=args)
    node = CameraInfoLowNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
