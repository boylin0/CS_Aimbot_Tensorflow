import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

import cv2

#----------------
from PIL import ImageGrab
import win32gui, win32ui, win32con, win32api
import time
import ctypes

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def grab_screen(region=None):

    hwin = win32gui.GetDesktopWindow()

    if region:
            left,top,x2,y2 = region
            width = x2 - left + 1
            height = y2 - top + 1
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)
    
    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (height,width,4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

#----------------

from modules.utils import label_map_util
from modules.utils import visualization_utils as vis_util
PATH_TO_CKPT = os.path.join('model', 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join('model', 'label_map.pbtxt')
NUM_CLASSES = 90


# Load a (frozen) Tensorflow model into memory.

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# In[10]:
height = 760
width = 1280
start_x = 0
start_y = 40

win_center_x = start_x + (width/2)
win_center_y = start_y + (height/2)

def mouse_click():
  #ctypes.windll.user32.SetCursorPos(x, y)
  ctypes.windll.user32.mouse_event(2, 0, 0, 0,0) # left down
  ctypes.windll.user32.mouse_event(4, 0, 0, 0,0) # left up

class _point_t(ctypes.Structure):
    _fields_ = [
                ('x',  ctypes.c_long),
                ('y',  ctypes.c_long),
               ]
               
def mouse_move_abs(x,y):
  point = _point_t()
  result = ctypes.windll.user32.GetCursorPos(ctypes.pointer(point))
  if result and x != 0 and y !=0:
    ctypes.windll.user32.SetCursorPos(point.x + x, point.y + y)
    mouse_click()
  
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    while True:
      last_time = time.time()
      image_np = np.array(grab_screen(region=(start_x,start_y,width,height)))
      image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      # Visualization of the results of a detection.
      '''
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
      '''
      abs_x = 0
      abs_y = 0
      target_x = 0
      target_y = 0
      isFirst = False
      for index, value in enumerate(classes[0]):
        if scores[0, index] > 0.8:
          if value == 1:
            ymin = boxes[0][index][0]*height
            xmin = boxes[0][index][1]*width
            ymax = boxes[0][index][2]*height
            xmax = boxes[0][index][3]*width
            cv2.rectangle(image_np,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(0,255,0),3)
            target_x = int(xmin + ((xmax-xmin) / 2))
            target_y = int(ymin + ((ymax-ymin) / 2))
            if isFirst==False:
              abs_x = int(target_x - win_center_x)
              abs_y = int(target_y - win_center_y)
              isFirst = True
            if abs(int(target_x - win_center_x)) < abs(abs_x):
              abs_x = int(target_x - win_center_x)
            if abs(int(target_y - win_center_y)) < abs(abs_y):
              abs_y = int(target_y - win_center_y)
            print('target_x:{} target_y: {} relative_x:{} relative_y:{}'.format(target_x,  target_y, abs_x, abs_y))
            
      if win32api.GetAsyncKeyState(win32con.VK_MBUTTON):
        mouse_move_abs(abs_x,abs_y)
        
      image_np = cv2.resize(image_np, ( int(width / 1.5) , int(height / 1.5) ))
      cv2.imshow('Object Detection', image_np)
      if cv2.waitKey(1) and win32api.GetAsyncKeyState(win32con.VK_F1) and win32api.GetAsyncKeyState(win32con.VK_CONTROL):
        cv2.destroyAllWindows()
        break
      '''
      if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
      '''
      