import  pyrealsense2 as rs
import numpy as np
import cv2
import os

def get_ply_file():
    # filters the .ply files there exists on directory
    plys = []
    for file in os.listdir():
        if '.ply' in file:
            plys.append(file)
    # return list with all .ply files 
    return plys

class PointCloud:
    def __init__(self):

        # Declare pointcloud object, for calculating pointclouds and texture mappings
        self.pc = rs.pointcloud()
        # We want the points object to be persistent so we can display the last cloud when a frame drops
        self.points = rs.points()

        # Declare RealSense pipeline, encapsulating the actual device and sensors
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Enable color and depth stream
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        return

    def render_deph (self):
        # Start streaming with chosen configuration
        self.pipeline.start(self.config)
        try:
            while True:
                self.frames = self.pipeline.wait_for_frames()
                self.depth_frame = self.frames.get_depth_frame()
                self.color_frame = self.frames.get_color_frame()

                # Convert images to numpy arrays
                depth_image = np.asanyarray(self.depth_frame.get_data())
                color_image = np.asanyarray(self.color_frame.get_data())

                # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                
                # Show images
                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense', np.hstack((color_image, depth_colormap)))
                cv2.waitKey(1)

        finally:
            self.pipeline.stop()

if __name__ == "__main__":
    pc = PointCloud()
    pc.render_deph()
