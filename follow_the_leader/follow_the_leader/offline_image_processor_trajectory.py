#!/usr/bin/env python3
import os
import cv2
import numpy as np
from follow_the_leader.networks.flowgan import FlowGAN
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation

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

class ImageProcessorNode:
    def __init__(self, image_folder):
        self.image_folder = image_folder
        size=(848, 480)
        self.image_processor = FlowGAN(size,size , # Update size as per your requirements
            use_flow=True,
            gan_name="synthetic_flow_pix2pix",
            gan_input_channels=6,
            gan_output_channels=1,
        )
        self.movement_threshold = 0.0075     
        self.just_activated = False
        self.last_image = None
        self.last_pose = None
        self.last_skipped = False

    def process_images(self, output_folder, img_name, camera_pose):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        if self.movement_threshold:
            tf_mat = camera_pose
            pos = tf_mat[:3, 3]
            if self.last_pose is None:
                self.last_pose = tf_mat
            else:
                # If the camera has rotated too much, we assume we get bad optical flows
                rotation = Rotation.from_matrix(self.last_pose[:3, :3].T @ tf_mat[:3, :3]).as_euler("XYZ")
                if np.linalg.norm(rotation) > np.radians(0.5):
                    self.last_pose = tf_mat
                    self.last_skipped = True
                    return

                last_pos = self.last_pose[:3, 3]
                diff = pos - last_pos
                if np.linalg.norm(diff) < self.movement_threshold:
                    return

                movement = np.linalg.inv(tf_mat[:3, :3]) @ diff
                movement /= np.linalg.norm(movement)
                self.last_pose = tf_mat
        init_image_path = os.path.join(self.image_folder, img_name)
        img = cv2.imread(init_image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = self.image_processor.process(img).mean(axis=2).astype(np.uint8)
        
        # Save the mask
        mask_file = os.path.join(output_folder, img_name.replace(".jpg", "_flow" + ".png"))
        cv2.imwrite(mask_file, mask)

def main():
    image_folder = "/media/ubuntu/062ACAAC2ACA9857/Red_currant/20230322_145916/color"
    output_folder = "/media/ubuntu/062ACAAC2ACA9857/Red_currant/20230322_145916/optical_flow"
    trajectory_file = "/media/ubuntu/062ACAAC2ACA9857/Red_currant/20230322_145916/trajectory.log"
    camera_poses = read_trajectory(trajectory_file) 

    node = ImageProcessorNode(image_folder)
    image_files = sorted([filename for filename in os.listdir(image_folder)])  # Sort the files of the color frames to process sequentially
    for i in range(len(image_files)-1):
        node.process_images(output_folder,image_files[i],camera_poses[i].pose)

if __name__ == "__main__":
    main()

