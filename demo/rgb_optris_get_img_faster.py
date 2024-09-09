import cv2
import numpy as np
import pyrealsense2 as rs
import threading
import time
from ctypes.util import find_library
import ctypes as ct
import datetime
import queue
import os

## Camera setup and initialization

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
if os.name == 'nt':
    libir = ct.CDLL('.\\libirimager.dll')
else:
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

# Global variables for thread communication
rgb_frame = None
thermal_frame = None
rgb_queue = queue.Queue(maxsize=1)
thermal_queue = queue.Queue(maxsize=1)
exit_event = threading.Event()

def realsense_thread():
    global rgb_frame
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    try:
        while not exit_event.is_set():
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if color_frame:
                color_image = np.asanyarray(color_frame.get_data())
                if not rgb_queue.full():
                    rgb_queue.put(color_image)
    finally:
        pipeline.stop()

def thermal_thread():
    global thermal_frame
    try:
        while not exit_event.is_set():
            thermal_ret = libir.evo_irimager_get_thermal_palette_image_metadata(
                thermal_width, thermal_height, npThermalPointer,
                palette_width, palette_height, npImagePointer, ct.byref(metadata)
            )
            thermal_img_raw = np_img.copy().reshape(palette_height.value, palette_width.value, 3)[:,:,::-1]
            thermal_img_gray = np.uint8(cv2.cvtColor(thermal_img_raw, cv2.COLOR_RGB2BGR))
            
            if len(thermal_img_gray.shape) == 3:
                thermal_img_gray = cv2.cvtColor(thermal_img_gray, cv2.COLOR_BGR2GRAY)
            
            thermal_img_low_res = cv2.cvtColor(thermal_img_gray, cv2.COLOR_GRAY2BGR)
            thermal_img = cv2.resize(thermal_img_low_res[:,:], (640, 480))
            
            if not thermal_queue.full():
                thermal_queue.put(thermal_img)
            
            time.sleep(0.01)
    finally:
        libir.evo_irimager_terminate()

def detectCircle(colorImg):
    colorImgCopy = cv2.cvtColor(colorImg, cv2.COLOR_BGR2BGRA)
    grayColor = cv2.cvtColor(colorImg, cv2.COLOR_BGR2GRAY)
    
    params = cv2.SimpleBlobDetector_Params()
    params.maxArea = 500
    params.minArea = 40
    params.minDistBetweenBlobs = 5
    detector = cv2.SimpleBlobDetector_create(params)
    
    CHECKERBOARD = (7, 5)
    
    try:
        ret, corners = cv2.findCirclesGrid(grayColor, CHECKERBOARD, cv2.CALIB_CB_SYMMETRIC_GRID, blobDetector=detector)
        if ret:
            circle_det = cv2.drawChessboardCorners(colorImg, CHECKERBOARD, corners, ret)
            return circle_det, corners, colorImgCopy
        else:
            return colorImg, [], colorImgCopy
    except:
        return colorImg, [], colorImgCopy

def visualization_thread():
    index_p = 0
    rgbTwoDPoints = []
    thermalTwoDPoints = []

    while not exit_event.is_set():
        try:
            rgb_frame = rgb_queue.get(timeout=1)
            thermal_frame = thermal_queue.get(timeout=1)
        except queue.Empty:
            continue

        circle_rgb, corner_rgb, raw_rgb = detectCircle(rgb_frame)
        circle_thermal, Corner_thermal, raw_thermal = detectCircle(thermal_frame)

        imgs = np.hstack([circle_rgb, circle_thermal])
        cv2.imshow("calibration", imgs)

        key = cv2.waitKey(1)
        if key & 0xFF == ord("s"):
            if len(corner_rgb) != 0 and len(Corner_thermal) != 0:
                rgbTwoDPoints.append(corner_rgb)
                thermalTwoDPoints.append(Corner_thermal)
                cv2.imwrite(f'/home/gg/calib_all/calib_results/rgb_thermal/rs{index_p}.png', circle_rgb)
                cv2.imwrite(f'/home/gg/calib_all/calib_results/rgb_thermal/thermal{index_p}.png', circle_thermal)
                cv2.imwrite(f'/home/gg/calib_all/calib_results/rgb_thermal/rs_raw{index_p}.png', raw_rgb)
                cv2.imwrite(f'/home/gg/calib_all/calib_results/rgb_thermal/thermal_raw{index_p}.png', raw_thermal)
                print(f"take picture {index_p}")
                index_p += 1
                if index_p == 55:
                    break
            else:
                print("Undetect feature point...")
        elif key == 27:
            break

    np.save("/home/gg/calib_all/calib_results/rgb_thermal_pixel_coordinate.npy", np.array(rgbTwoDPoints))
    np.save("/home/gg/calib_all/calib_results/thermal_pixel_coordinate.npy", np.array(thermalTwoDPoints))
    exit_event.set()

if __name__ == "__main__":
    realsense_thread = threading.Thread(target=realsense_thread)
    thermal_thread = threading.Thread(target=thermal_thread)
    vis_thread = threading.Thread(target=visualization_thread)

    realsense_thread.start()
    thermal_thread.start()
    vis_thread.start()

    realsense_thread.join()
    thermal_thread.join()
    vis_thread.join()

    cv2.destroyAllWindows()
