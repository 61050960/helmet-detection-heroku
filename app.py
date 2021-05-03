import streamlit as st
import numpy as np
import pandas as pd
import tempfile
import cv2 
import os
import tensorflow as tf
import sys
from random import randint
from streamlit.server.server import Server
from utils import label_map_util
from utils import visualization_utils as vis_util
import utils.SessionState as SessionState

sys.path.append("..")
st.set_page_config(layout="wide")
def parameter_sliders(key, enabled = True, value = None):
    conf = custom_slider("Model Confidence", 
                        minVal = 0, maxVal = 100, InitialValue= value[0], enabled = enabled)

        
    return(conf)

def trigger_rerun():
    """
    mechanism in place to force resets and update widget states
    """
    session_infos = Server.get_current()._session_info_by_id.values() 
    for session_info in session_infos:
        this_session = session_info.session
    this_session.request_rerun()

def main():
    checkStart = ""
    run = False
    # st.set_page_config(page_title = "Traffic Flow Counter", 
    # page_icon=":vertical_traffic_light:")

    # # obj_detector = load_obj_detector(config, wt_file)
    # # tracker = tc.CarsInFrameTracker(num_previous_frames = 10, frame_shape = (720, 1080))

    state = SessionState.get(upload_key = None, enabled = True, 
    start = False, conf = 70, run = False)
    # hide_streamlit_widgets()

    st.markdown("<h1 style='text-align: center; font-size: 65px;'>Helmet Detection </h1>", unsafe_allow_html=True)
    """
    # Helmet Detection :mortar_board: 
    This website is a part of the presentation in computer vision (05506088) courses. 
    Please Upload a video file to track helmet. Don't forget to change parameters to tune the model!
  
    """
    st.markdown(''' **Submitted by Mr.Pornthawee THaweesin (610504246), Mrs.Anongnart Pongpatana (61050960).** ''')

 
    """
    # :floppy_disk: Parameters 
    """
    state.conf = st.slider("Model Confidence",60,100)
    conf = state.conf/ 100
    # st.text("")
    # st.text("")
    # st.text("")


    #set model confidence and nms threshold 
    # if (state.nms is not None):
    #     obj_detector.nms_threshold = state.nms/ 100 


    """
    # :video_camera: Upload Video
    """

    upload = st.empty()
    start_button = st.empty()
    stop_button = st.empty()

    with upload:
        f = st.file_uploader('Upload Video file (mpeg/mp4 format)', key = state.upload_key)
    if f is not None:
        tfile  = tempfile.NamedTemporaryFile(delete = True)
        tfile.write(f.read())

        upload.empty()
        vf = cv2.VideoCapture(tfile.name)

        if not state.run:
          start = start_button.button("start")
          state.start = start
        
        if state.start:
            start_button.empty()
            #state.upload_key = str(randint(1000, int(1e6)))
            state.enabled = False
            if state.run:
                st.write("Confident : ", conf*100)
                tfile.close()
                f.close()
                state.upload_key = str(randint(1000, int(1e6)))
                state.enabled = True
                state.run = False
                ProcessFrames(vf,stop_button,conf)
            else:
                state.run = True
                trigger_rerun()


def ProcessFrames(vf,stop,conf):
  # Name of the directory containing the object detection module we're using
  MODEL_NAME = 'inference_graph'
  VIDEO_NAME = 'VDO3.mp4'

  # Grab path to current working directory
  CWD_PATH = os.getcwd()

  # Path to frozen detection graph .pb file, which contains the model that is used
  # for object detection.
  PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

  # Path to label map file
  PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

  # Path to video
  PATH_TO_VIDEO = os.path.join(CWD_PATH,VIDEO_NAME)

  # Number of classes the object detector can identify
  NUM_CLASSES = 2

  # Load the label map.
  # Label maps map indices to category names, so that when our convolution
  # network predicts `5`, we know that this corresponds to `king`.
  # Here we use internal utility functions, but anything that returns a
  # dictionary mapping integers to appropriate string labels would be fine
  label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
  categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
  category_index = label_map_util.create_category_index(categories)

  # Load the Tensorflow model into memory.
  detection_graph = tf.Graph()
  with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
          serialized_graph = fid.read()
          od_graph_def.ParseFromString(serialized_graph)
          tf.import_graph_def(od_graph_def, name='')

      sess = tf.Session(graph=detection_graph)

  # Define input and output tensors (i.e. data) for the object detection classifier

  # Input tensor is the image
  image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

  # Output tensors are the detection boxes, scores, and classes
  # Each box represents a part of the image where a particular object was detected
  detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

  # Each score represents level of confidence for each of the objects.
  # The score is shown on the result image, together with the class label.
  detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
  detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

  # Number of objects detected
  num_detections = detection_graph.get_tensor_by_name('num_detections:0')

  # Open video file
  #video = cv2.VideoCapture(PATH_TO_VIDEO)
  _stop = stop.button("stop")
  stframe = st.empty()
  while(vf.isOpened()):

      # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
      # i.e. a single-column array, where each item in the column has the pixel RGB value
      ret, frame = vf.read()
      frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      frame_expanded = np.expand_dims(frame_rgb, axis=0)
      if _stop:
        break

      # Perform the actual detection by running the model with the image as input
      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: frame_expanded})

      # Draw the results of the detection (aka 'visulaize the results')
      vis_util.visualize_boxes_and_labels_on_image_array(
          frame_rgb,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8,
          min_score_thresh=conf)

      # All the results have been drawn on the frame, so it's time to display it.
      #cv2.imshow('Object detector', frame)
      stframe.image(frame_rgb, width = 720)

main()