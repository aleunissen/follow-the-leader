#!/usr/bin/env python3
import os
import cv2
import numpy as np
from follow_the_leader.networks.flowgan import FlowGAN
from cv_bridge import CvBridge

bridge = CvBridge()

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

    def process_images(self, output_folder, image_files, index, skip_frames=0):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        
        for i in range(skip_frames):
            init_image = image_files[index]
            init_image_path = os.path.join(self.image_folder, init_image)
            img = cv2.imread(init_image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            second_image_path = os.path.join(self.image_folder, image_files[index+i])
            second_img = cv2.imread(second_image_path)
            second_img = cv2.cvtColor(second_img, cv2.COLOR_BGR2RGB)
            third_image_path = os.path.join(self.image_folder, image_files[index+i])
            third_img = cv2.imread(third_image_path)
            third_img = cv2.cvtColor(third_img, cv2.COLOR_BGR2RGB)

            # Process the image
            init_mask = self.image_processor.process(img).mean(axis=2).astype(np.uint8)
            inbetween_mask = self.image_processor.process(second_img).mean(axis=2).astype(np.uint8)
            target_mask = self.image_processor.process(third_img).mean(axis=2).astype(np.uint8)
            self.image_processor.reset()

            # Save the mask
            mask_file = os.path.join(output_folder, init_image.replace(".png", "_diff_" + str(i) + ".png"))
            cv2.imwrite(mask_file, target_mask)

def main():
    image_folder = "/media/ubuntu/062ACAAC2ACA9857/Red_currant/Skipping_test_2"
    output_folder = "/media/ubuntu/062ACAAC2ACA9857/Red_currant/Skipping_test_2/output"

    node = ImageProcessorNode(image_folder)
    image_files = sorted([filename for filename in os.listdir(image_folder) if "_color" in filename])  # Sort the files of the color frames to process sequentially
    for i in range(10):
        output_subfolder = os.path.join(output_folder, image_files[i].replace("_color.png",""))
        node.process_images(output_subfolder,image_files,i,10)

if __name__ == "__main__":
    main()

