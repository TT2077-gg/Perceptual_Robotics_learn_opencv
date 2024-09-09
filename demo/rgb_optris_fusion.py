#!/usr/bin/python3

import cv2
import numpy as np
import pyrealsense2 as rs
import threading
import time
from ctypes.util import find_library
import ctypes as ct

# Define EvoIRFrameMetadata structure
class EvoIRFrameMetadata(ct.Structure):
    _fields_ = [
        ("counter", ct.c_uint),
        ("counterHW", ct.c_uint),
        ("timestamp", ct.c_longlong),
        ("timestampMedia", ct.c_longlong),
        ("flagState", ct.c_int),
        ("tempChip", ct.c_float),
        ("tempFlag", ct.c_float),
        ("tempBox", ct.c_float)
    ]

# Load IR library
libir = ct.cdll.LoadLibrary(ct.util.find_library("irdirectsdk"))

# Initialize IR camera variables
pathFormat, pathLog, pathXml = b'/usr/include/libirimager', b'logfilename', b'./test.xml'
palette_width, palette_height = ct.c_int(), ct.c_int()
thermal_width, thermal_height = ct.c_int(), ct.c_int()
serial = ct.c_ulong()
metadata = EvoIRFrameMetadata()

# Initialize IR library
ret = libir.evo_irimager_usb_init(pathXml, pathFormat, pathLog)
libir.evo_irimager_get_serial(ct.byref(serial))
libir.evo_irimager_get_thermal_image_size(ct.byref(thermal_width), ct.byref(thermal_height))
libir.evo_irimager_get_palette_image_size(ct.byref(palette_width), ct.byref(palette_height))

# Initialize thermal data container
np_thermal = np.zeros([thermal_width.value * thermal_height.value], dtype=np.uint16)
npThermalPointer = np_thermal.ctypes.data_as(ct.POINTER(ct.c_ushort))

# Initialize image container
np_img = np.zeros([palette_width.value * palette_height.value * 3], dtype=np.uint8)
npImagePointer = np_img.ctypes.data_as(ct.POINTER(ct.c_ubyte))

# Load calibration data
rgbIntrinsic = np.load("./calib_results/thermal_realsense_RGB_intrinsic.npy")
rgbDistortion = np.load("./calib_results/thermal_realsense_RGB_distortion.npy")
thermalIntrinsic = np.load("./calib_results/thermal_realsense_thermal_intrinsic.npy")
thermalDistortion = np.load("./calib_results/thermal_realsense_thermal_distortion.npy")
relative_R = np.load("./calib_results/thermal_realsense_Ralative_rotation_matrix.npy")
relative_T = np.load("./calib_results/thermal_realsense_Ralative_translation_matrix.npy")

# Setup RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

# Align depth to color
align_to = rs.stream.color
align = rs.align(align_to)

def get_rgbd_stream():
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    return depth_image, color_image

def get_thermal_stream():
    thermal_ret = libir.evo_irimager_get_thermal_palette_image_metadata(
        thermal_width, thermal_height, npThermalPointer,
        palette_width, palette_height, npImagePointer, ct.byref(metadata)
    )
    thermal_img_raw = np_img.copy().reshape(palette_height.value, palette_width.value, 3)[:,:,::-1]
    thermal_img_gray = np.uint8(cv2.cvtColor(thermal_img_raw, cv2.COLOR_RGB2BGR))
    
    if len(thermal_img_gray.shape) == 3:
        thermal_img_gray = cv2.cvtColor(thermal_img_gray, cv2.COLOR_BGR2GRAY)
    
    thermal_img = cv2.resize(thermal_img_gray[:,:], (640, 480))
    return thermal_img

def project_thermal_to_rgb(depth_image, thermal_image, color_image):
    # Camera intrinsics and extrinsics
    fx = rgbIntrinsic[0,0]
    fy = rgbIntrinsic[1,1]
    cx = rgbIntrinsic[0,2]
    cy = rgbIntrinsic[1,2]
    
    # Create image coordinate grids
    x = np.arange(0, 640)
    y = np.arange(0, 480)
    x_c, y_c = np.meshgrid(x, y)
    
    # Calculate real world coordinates
    x_real_imgplane = (x_c - cx) / fx
    y_real_imgplane = (y_c - cy) / fy
    
    # Calculate 3D points
    w_x = np.multiply(x_real_imgplane, depth_image)
    w_y = np.multiply(y_real_imgplane, depth_image)
    real_point = np.stack((w_x, w_y, depth_image), axis=-1)
    
    real_point = real_point.reshape(-1, 3)
    bad_obj_points = np.where(real_point[:,2] == 0)
    
    # Project 3D points to thermal image plane
    relative_Rvct = cv2.Rodrigues(relative_R)[0]
    imagePoints, _ = cv2.projectPoints(real_point, relative_Rvct, relative_T, thermalIntrinsic, thermalDistortion)
    
    imagePoints = imagePoints[:,0,:]
    imagePoints = np.round(imagePoints).astype(int)
    imagePoints = np.flip(imagePoints, axis=1)
    
    # Clip coordinates to image boundaries
    row = np.clip(imagePoints[:,0], 0, 479)
    col = np.clip(imagePoints[:,1], 0, 639)
    
    # Sample thermal image
    projected_image = thermal_image[row, col]
    projected_image[bad_obj_points] = 0
    projected_image = projected_image.reshape(480, 640)
    
    # Apply color map to thermal image
    projected_img_color = cv2.applyColorMap(projected_image, cv2.COLORMAP_HOT)
    
    # Blend RGB and thermal images
    fusion_img = cv2.addWeighted(color_image, 0.5, projected_img_color, 0.5, 0)
    
    return fusion_img


if __name__ == "__main__":
    try:
        while True:
            depth_image, color_image = get_rgbd_stream()
            thermal_image = get_thermal_stream()
            
            fused_image = project_thermal_to_rgb(depth_image, thermal_image,color_image)
            
            
            cv2.imshow("RGB Image", color_image)
            cv2.imshow("Thermal Image", thermal_image)
            cv2.imshow("Fused Image", fused_image)
            
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        pipeline.stop()
        libir.evo_irimager_terminate()
        cv2.destroyAllWindows()
