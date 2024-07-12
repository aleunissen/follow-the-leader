#!/usr/bin/env python3
import os
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from follow_the_leader.networks.flowgan import FlowGAN
from cv_bridge import CvBridge
import tf2_ros
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
import time

bridge = CvBridge()

class CameraPose:
    def __init__(self, meta, mat):
        self.metadata = meta
        self.pose = mat

    def __str__(self):
        return 'Metadata : ' + ' '.join(map(str, self.metadata)) + '\n' + \
            "Pose : " + "\n" + np.array_str(self.pose)

def read_trajectory(filename):
    traj = []
    with open(filename, 'r') as f:
        metastr = f.readline()
        while metastr:
            metadata = list(map(int, metastr.split()))
            mat = np.zeros(shape=(4, 4))
            for i in range(4):
                matstr = f.readline()
                mat[i, :] = np.fromstring(matstr, dtype=float, sep=' \t')
            traj.append(CameraPose(metadata, mat))
            metastr = f.readline()
    return traj

class ImagePublisherNode(Node):
    def __init__(self, image_folder):
        super().__init__('image_publisher_node')
        self.image_folder = image_folder
        size=(848, 480)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.image_publisher = self.create_publisher(Image, '/camera/color/image_raw', 10)
        self.cam_info_pub = self.create_publisher(CameraInfo, "/camera/color/camera_info", 1)

        # Setup camera
        
        self.cam_info = CameraInfo(
            height= 240,
            width= 424,
            distortion_model="plumb_bob",
            binning_x=0,
            binning_y=0,
            d=[0.05469128489494324, 0.05773274227976799, 7.857435412006453e-05, 0.0003967129159718752, -0.018736450001597404],
            r=np.identity(3).flatten(),
            k=[437.00222778, 0.0, 418.9420166, 0.0, 439.22055054, 240.41038513,0.0,0.0,1.0],
            p=[437.00222778, 0.0, 418.9420166, 0.0, 0.0, 439.22055054, 240.41038513, 0.0, 0.0, 0.0, 1.0, 0.0])
        self.cam_info.header.frame_id = "camera_color_optical_frame"


    def publish_pose(self, camera_pose):
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = 'base_link'
        transform.child_frame_id = 'tool0'
        transform.transform.translation.x = camera_pose[0, 3]
        transform.transform.translation.y = camera_pose[2, 3]
        transform.transform.translation.z = camera_pose[1, 3]*-1
        quat = Rotation.from_matrix(camera_pose[:3, :3]).as_quat()
        angles_deg = [0, 180, 0]  # Roll, Pitch, Yaw
        r = Rotation.from_euler('zyx', angles_deg, degrees=True)
        quat = Rotation.__mul__(r,Rotation.from_matrix(camera_pose[:3, :3])).as_quat()
        # if np.linalg.norm(quat) > np.radians(0.5):
        #     print("ROTATION TOO BIGGGG")
        transform.transform.rotation.x = quat[0]
        transform.transform.rotation.y = quat[1]
        transform.transform.rotation.z = quat[2]
        transform.transform.rotation.w = quat[3]
        self.tf_broadcaster.sendTransform(transform)
        # transform.child_frame_id = 'camera_color_optical_frame'
        # self.tf_broadcaster.sendTransform(transform)

    def publish_images(self,img_name, camera_pose):        
        init_image_path = os.path.join(self.image_folder, img_name)
        img = cv2.imread(init_image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = img
        # result = np.zeros_like(mask)
        # # Define the left and right regions
        # left_region = (0, 0, mask.shape[1] // 3, mask.shape[0])
        # right_region = (2 * mask.shape[1] // 3, 0, mask.shape[1], mask.shape[0])
        # # Fill the left and right regions with white color (255)
        # cv2.rectangle(result, left_region[:2], left_region[2:], (255), -1)
        # cv2.rectangle(result, right_region[:2], right_region[2:], (255), -1)
        # # Combine the mask and result image using bitwise AND
        # result = cv2.bitwise_and(result, mask)
        resized_image = cv2.resize(img, (424, 240))
        msg = bridge.cv2_to_imgmsg(resized_image, encoding="rgb8")
        self.image_publisher.publish(msg)        
        self.cam_info_pub.publish(self.cam_info)
        self.publish_pose(camera_pose)
        time.sleep(0.3)

def main(args=None):
    rclpy.init(args=args)
    image_folder = "/home/ubuntu/FTL/data/color"
    trajectory_file = "/home/ubuntu/FTL/data/trajectory.log"
    camera_poses = read_trajectory(trajectory_file) 

    node = ImagePublisherNode(image_folder)
    image_files = sorted([filename for filename in os.listdir(image_folder)])  # Sort the files of the color frames to process sequentially
    for i in range(len(image_files)-1):
        node.publish_images(image_files[i],camera_poses[i].pose)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

