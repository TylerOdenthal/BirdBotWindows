######## BirdBot ML GUI Software Using Tensorflow Object Detection #########
#
# Author: Tyler Odenthal
# Date: 5/19/21
# Description: 
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier and uses it to perform object detection on a video.
# It draws boxes, scores, and labels around the objects of interest in each
# frame of the video.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## Some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import time
import keyboard
import textwrap 
import xml.etree.cElementTree as ET

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util
from tkinter import *
from tkinter import filedialog
from datetime import datetime
from xml.dom import minidom


def BirdBotVideo(filename):

    # Name of the directory containing the object detection module we're using
    MODEL_NAME = 'BirdModel'
    VIDEO_NAME = filename
    min_score_thresh = 0.60
    min_frames_thresh = 28
    max_frames_thresh = 30
    print(filename)
	
	#Gets root file name of video choosen in Tkinter directory
    file_save_name = filename.split('object_detection/')[1].lstrip().split('.')[0]
    print(file_save_name)

    # Grab path to current working directory
    CWD_PATH = os.getcwd()

    # Path to frozen detection graph .pb file, which contains the model that is used
    # for object detection.
    PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH,'data','object-detection.pbtxt')

    # Path to video
    PATH_TO_VIDEO = os.path.join(CWD_PATH,VIDEO_NAME)

    # Number of classes the object detector can identify
    NUM_CLASSES = 12

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
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(graph=detection_graph, config=config)

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
    video = cv2.VideoCapture(PATH_TO_VIDEO)

    imW = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(imW)
    imH = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(imH)
    
    frameCount = 0
    species_frame_count = 0
    seen_species_array = []
    wrapped_seen_species = ''
    approval_species_array = []
    timer_species_array = []
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(file_save_name + '-ML.mp4', fourcc, 30.0, (imW, imH)) # Make sure (width,height) is the shape of input frame from video

    while(video.isOpened()):

        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        ret, frame = video.read()
        
		#Start frame-rate timer
        t1 = cv2.getTickCount()
		
        if frame is None:
            out.release()
            video.release()
            cv2.destroyAllWindows()
            print("END OF VIDEO")
            break
        
        # frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_expanded = np.expand_dims(frame_rgb, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})
        
        object_count = 0
        
        # START PROCESSING THE DETECTION DATA WITH MY CODE
        processedScores = []
        processedClasses = []
        processedBoxes = []
        currentSpecies = []
        blk = np.zeros(frame.shape, np.uint8)
        
        cv2.rectangle(blk, (20, imH-20), (160, imH-50), (255, 255, 255), cv2.FILLED)
        
        # if (scores[0][object_count] < min_score_thresh):
            # print("No birds detected")
            
        # Create a running object name array that resets every 30 frames or 1 second.
        # print(frameCount % max_frames_thresh)
        if frameCount % max_frames_thresh == 0:
            approval_species_array = []
            species_frame_count = 0
        
        # IF SCORE ARRAY IS GREATER THAN MIN SCORE AND LESS THAN 1 ADD TO PROCESS ARRAY
        while ((scores[0][object_count] > min_score_thresh) and (scores[0][object_count] <= 1.0)):
            
            object_name = str(category_index.get(classes[0][object_count]).get('name'))
            
            processedScores.append(str(scores[0][object_count]))
            processedClasses.append(str(category_index.get(classes[0][object_count]).get('name')))
            processedBoxes.append(str(boxes[0][object_count]))
            
            # print('Predictions: ' + str(object_count))
            # print(str(category_index.get(classes[0][object_count]).get('name')))
            
            # print(np.squeeze(boxes[0][object_count])) 
            
            top = int(max(1, imH * boxes[0][object_count][0]))
            # print(top)
            left = int(max(1, imW * boxes[0][object_count][1]))
            # print(left)
            bottom = int(min(imH, imH * boxes[0][object_count][2]))
            # print(bottom)
            right = int(min(imW, imW * boxes[0][object_count][3]))
            # print(right)
            
            if "Song Sparrow" in object_name:
            
                blue = 10
                green = 255
                red = 0
                
                cv2.rectangle(frame, (left,top), (right,bottom), (blue, green, red), 4)
            elif "Chestnut-backed Chickadee" in object_name:
            
                blue = 24
                green = 70
                red = 87
                
                cv2.rectangle(frame, (left,top), (right,bottom), (blue, green, red), 4)
            elif "European Starling" in object_name:
            
                blue = 96
                green = 96
                red = 96
            
                cv2.rectangle(frame, (left,top), (right,bottom), (blue, green, red), 4)
            elif "Golden-crowned Sparrow" in object_name:
            
                blue = 0
                green = 205
                red = 205
                
                cv2.rectangle(frame, (left,top), (right,bottom), (blue, green, red), 4)
            elif "Spotted Towhee" in object_name:
            
                blue = 47
                green = 38
                red = 29
            
                cv2.rectangle(frame, (left,top), (right,bottom), (47, 38, 29), 4)
            elif "American Goldfinch" in object_name:
            
                blue = 150
                green = 255
                red = 255
            
                cv2.rectangle(frame, (left,top), (right,bottom), (blue, green, red), 4)
            elif "Black-capped Chickadee" in object_name:
                    
                blue = 23
                green = 29
                red = 29
           
                cv2.rectangle(frame, (left,top), (right,bottom), (blue, green, red), 4)
            elif "Dark-eyed Junco" in object_name:
            
                blue = 57
                green = 47
                red = 45
            
                cv2.rectangle(frame, (left,top), (right,bottom), (blue, green, red), 4)
            elif "Pine Siskin" in object_name:
            
                blue = 137
                green = 110
                red = 97
            
                cv2.rectangle(frame, (left,top), (right,bottom), (blue, green, red), 4)
            elif "House Finch" in object_name:
            
                blue = 212
                green = 170
                red = 255
            
                cv2.rectangle(frame, (left,top), (right,bottom), (blue, green, red), 4)
            elif "Steller's Jay" in object_name:
            
                blue = 191
                green = 95
                red = 0
            
                cv2.rectangle(frame, (left,top), (right,bottom), (blue, green, red), 4)
            elif "American Robin" in object_name:
            
                blue = 54
                green = 101
                red = 163
            
                cv2.rectangle(frame, (left,top), (right,bottom), (blue, green, red), 4) 
            elif "Red-breasted Nuthatch" in object_name:
            
                blue = 58
                green = 95
                red = 196
            
                cv2.rectangle(frame, (left,top), (right,bottom), (blue, green, red), 4)
            elif "Black-headed Grosbeak" in object_name:
            
                blue = 0
                green = 106
                red = 188
            
                cv2.rectangle(frame, (left,top), (right,bottom), (blue, green, red), 4)
            elif "House Sparrow" in object_name:
            
                blue = 10
                green = 66
                red = 89
            
                cv2.rectangle(frame, (left,top), (right,bottom), (blue, green, red), 4)
            else:
            
                blue = 10
                green = 255
                red = 0
            
                cv2.rectangle(frame, (left,top), (right,bottom), (blue, green, red), 4)  
                
            # Draw label on bird bounding boxes
            label = '%s: %d%%' % (object_name, int(scores[0][object_count]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 2) # Get font size
            label_ymin = max(top, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (left-1, label_ymin-labelSize[1]-10), (left+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (left-1, label_ymin-7), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2) # Draw label text
            
            currentSpecies = processedClasses
            
            approval_species_array.append(object_name)
            
            if object_name in approval_species_array:
                
                species_frame_count = approval_species_array.count(object_name)
                
                if species_frame_count >= min_frames_thresh and object_name not in seen_species_array:

                    seen_species_array.append(object_name)
                    wrapped_seen_species = textwrap.wrap('Seen Species: '+str(seen_species_array), width=70)
                
                # print(object_name + ": " + str(approval_species_array.count(object_name)))
                     
            object_count += 1
        
        # Draws the white box for wrapped seen species         
        for i, line in enumerate(wrapped_seen_species):
                        
            textsize = cv2.getTextSize(line, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 2)[0]

            gap = textsize[1] + 16

            y = 121 + i * gap
            x = 40 + textsize[0]
            
            if i >= 0:
                cv2.rectangle(blk, (20, 100), (x, y+10), (255, 255, 255), cv2.FILLED)
        
        # Create variables for drawing metrics on frames
        countLabel = 'Bird Count: '+str(object_count)  # Example: 5
        countLabelSize = cv2.getTextSize(countLabel, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 2)
        # print(countLabelSize[1])
        
        countSpeciesFrames = "Frame Threshold " + "[" + str(min_frames_thresh) + "]: " + str(species_frame_count)  # Example: 5
        countSpeciesFramesSize = cv2.getTextSize(countSpeciesFrames, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 2)
        # print(countLabelSize[1])
        
        countSpeciesArray = 'Current Species: '+str(currentSpecies)  # Example: [Bald Eagle, Black-capped Chickadee, Steller's Jay]
        countSpeciesArraySize = cv2.getTextSize(countSpeciesArray, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 2)
        # print(countSpeciesArraySize[0][0])
        
        # countSeenSpeciesArray = 'Seen Species: '+str(seen_species_array) # Example: [Bald Eagle, Black-capped Chickadee, Steller's Jay]
        # countSeenSpeciesArraySize = cv2.getTextSize(countSeenSpeciesArray, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 2)
        # print(countSeenSpeciesArraySize[0][0])
        
        # Draw white boxes for metrics
        cv2.rectangle(blk, (20, 10), (210, 40), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
        
        if species_frame_count >= 0:
            cv2.rectangle(blk, (20, 40), (40 + countSpeciesFramesSize[0][0], 70), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
        
        if len(currentSpecies) >= 0:
            cv2.rectangle(blk, (20, 70), (40 + countSpeciesArraySize[0][0], 100), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in

        # if len(seen_species_array) >= 0:
            # cv2.rectangle(blk, (20, 100), (40 + countSeenSpeciesArraySize[0][0], 130), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in

        frame = cv2.addWeighted(frame, 1.0, blk, 0.40, 1)
        
        # Put text in boxes
        cv2.putText(frame, countLabel, (30, 31), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2) # Draw object count text
        cv2.putText(frame, countSpeciesFrames, (30, 61), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2) # Draw object count text
        cv2.putText(frame, countSpeciesArray, (30, 91), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2) # Draw object count text
        # cv2.putText(frame, countSeenSpeciesArray, (30, 121), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2) # Draw object count text

        for i, line in enumerate(wrapped_seen_species):
                        
            textsize = cv2.getTextSize(line, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 2)[0]

            gap = textsize[1] + 16

            y = 121 + i * gap
            x = int((frame.shape[1] - textsize[0]) / 2)

            cv2.putText(frame, line, (30, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,0), 2)        
        
        processedScores = []
        processedClasses = []
        processedBoxes = []
		
        cv2.putText(frame, 'FPS: ' + str(round(frame_rate_calc, 2)), (30, imH-30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)
        
        out.write(frame) 
	
        # Calculate framerate
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc= 1/time1
        print(frame_rate_calc)
        
        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector', frame)
        
        frameCount += 1

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            out.release()
            video.release()
            cv2.destroyAllWindows()
            break 
  
def BirdBotGenerateXML(filename):

    # Name of the directory containing the object detection module we're using
    MODEL_NAME = 'BirdModel'
    VIDEO_NAME = filename
    min_score_thresh = 0.60
    captured_frames = 10
    min_frames_thresh = 28
    max_frames_thresh = 30
    print(filename)
    
    now = datetime.now()
    current_time = now.strftime("%M-%S")

    # Grab path to current working directory
    CWD_PATH = os.getcwd()

    # Path to frozen detection graph .pb file, which contains the model that is used
    # for object detection.
    PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH,'data','object-detection.pbtxt')

    # Path to video
    PATH_TO_VIDEO = os.path.join(CWD_PATH,VIDEO_NAME)

    # Number of classes the object detector can identify
    NUM_CLASSES = 12

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
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(graph=detection_graph, config=config)

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
    video = cv2.VideoCapture(PATH_TO_VIDEO)

    imW = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(imW)
    imH = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(imH)
    
    frameCount = 0
    species_frame_count = 0
    seen_species_array = []
    approval_species_array = []
    timer_species_array = []

    while(video.isOpened()):

        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        ret, frame = video.read()
        ret2, trainframe = video.retrieve()
        
        if frame is None:
            video.release()
            cv2.destroyAllWindows()
            print("END OF VIDEO")
            break
        
        # frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_expanded = np.expand_dims(frame_rgb, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})
        
        object_count = 0
        
        # START PROCESSING THE DETECTION DATA WITH MY CODE
        processedScores = []
        processedClasses = []
        processedBoxes = []
        currentSpecies = []
        blk = np.zeros(frame.shape, np.uint8)
        
        objectHolder = {}
                
        root = ET.Element("annotation")

        # uncomment for training photos
        training = str(category_index.get(classes[0][object_count]).get('name')) + str(frameCount) + "-" + current_time + '.jpg'
        
        xmlFileName = str(category_index.get(classes[0][object_count]).get('name')) + str(frameCount) + "-" + current_time + '.xml'
        
        print(training)
        
        # write training photo every 10 frames on successful id
            
        folder = ET.SubElement(root, "folder")
        folder.text = "images"
        filename = ET.SubElement(root, "filename") 
        filename.text = training
        path = ET.SubElement(root, "path")  
        path.text = "images/"+training
        source = ET.SubElement(root, "source")
        
        database = ET.SubElement(source, "database")
        database.text = "unspecified"
        
        size = ET.SubElement(root, "size")
        
        width = ET.SubElement(size, "width")
        width.text = str(imW)
        height = ET.SubElement(size, "height")
        height.text = str(imH)
        depth = ET.SubElement(size, "depth")
        depth.text = str(3)
        
        # IF SCORE ARRAY IS GREATER THAN MIN SCORE AND LESS THAN 1 ADD TO PROCESS ARRAY
        while ((scores[0][object_count] > min_score_thresh) and (scores[0][object_count] <= 1.0)):
            
            object_name = str(category_index.get(classes[0][object_count]).get('name'))
            
            top = int(max(1, imH * boxes[0][object_count][0]))
            # print(top)
            left = int(max(1, imW * boxes[0][object_count][1]))
            # print(left)
            bottom = int(min(imH, imH * boxes[0][object_count][2]))
            # print(bottom)
            right = int(min(imW, imW * boxes[0][object_count][3]))
            # print(right)
            
            objectHolder["objectXml"+str(object_count)] = ET.SubElement(root, "object")

            objectHolder["nameXml"+str(object_count)] = ET.SubElement(objectHolder["objectXml"+str(object_count)], "name")
            objectHolder["nameXml"+str(object_count)].text = object_name
            objectHolder["pose"+str(object_count)] = ET.SubElement(objectHolder["objectXml"+str(object_count)], "pose")
            objectHolder["pose"+str(object_count)].text = "Unspecified"
            objectHolder["truncated"+str(object_count)] = ET.SubElement(objectHolder["objectXml"+str(object_count)], "truncated") 
            objectHolder["truncated"+str(object_count)].text = "Unspecified"
            objectHolder["difficult"+str(object_count)] = ET.SubElement(objectHolder["objectXml"+str(object_count)], "difficult")
            objectHolder["difficult"+str(object_count)].text = "Unspecified"

            objectHolder["bndbox"+str(object_count)] = ET.SubElement(objectHolder["objectXml"+str(object_count)], "bndbox") 

            objectHolder["xminXml"+str(object_count)] = ET.SubElement(objectHolder["bndbox"+str(object_count)], "xmin")
            objectHolder["xminXml"+str(object_count)].text = str(left)
            objectHolder["yminXml"+str(object_count)] = ET.SubElement(objectHolder["bndbox"+str(object_count)], "ymin") 
            objectHolder["yminXml"+str(object_count)].text = str(top)
            objectHolder["xmaxXml"+str(object_count)] = ET.SubElement(objectHolder["bndbox"+str(object_count)], "xmax") 
            objectHolder["xmaxXml"+str(object_count)].text = str(right)
            objectHolder["ymaxXml"+str(object_count)] = ET.SubElement(objectHolder["bndbox"+str(object_count)], "ymax") 
            objectHolder["ymaxXml"+str(object_count)].text = str(bottom)
            
            # print(objectHolder["ymaxXml"+str(object_count)].text)
            
            # print (ET.tostring(root))
            
            if "Song Sparrow" in object_name:
            
                blue = 10
                green = 255
                red = 0
                
                cv2.rectangle(frame, (left,top), (right,bottom), (blue, green, red), 4)
            elif "Chestnut-backed Chickadee" in object_name:
            
                blue = 24
                green = 70
                red = 87
                
                cv2.rectangle(frame, (left,top), (right,bottom), (blue, green, red), 4)
            elif "European Starling" in object_name:
            
                blue = 96
                green = 96
                red = 96
            
                cv2.rectangle(frame, (left,top), (right,bottom), (blue, green, red), 4)
            elif "Golden-crowned Sparrow" in object_name:
            
                blue = 0
                green = 205
                red = 205
                
                cv2.rectangle(frame, (left,top), (right,bottom), (blue, green, red), 4)
            elif "Spotted Towhee" in object_name:
            
                blue = 47
                green = 38
                red = 29
            
                cv2.rectangle(frame, (left,top), (right,bottom), (47, 38, 29), 4)
            elif "American Goldfinch" in object_name:
            
                blue = 150
                green = 255
                red = 255
            
                cv2.rectangle(frame, (left,top), (right,bottom), (blue, green, red), 4)
            elif "Black-capped Chickadee" in object_name:
                    
                blue = 23
                green = 29
                red = 29
           
                cv2.rectangle(frame, (left,top), (right,bottom), (blue, green, red), 4)
            elif "Dark-eyed Junco" in object_name:
            
                blue = 57
                green = 47
                red = 45
            
                cv2.rectangle(frame, (left,top), (right,bottom), (blue, green, red), 4)
            elif "Pine Siskin" in object_name:
            
                blue = 137
                green = 110
                red = 97
            
                cv2.rectangle(frame, (left,top), (right,bottom), (blue, green, red), 4)
            elif "House Finch" in object_name:
            
                blue = 212
                green = 170
                red = 255
            
                cv2.rectangle(frame, (left,top), (right,bottom), (blue, green, red), 4)
            elif "Steller's Jay" in object_name:
            
                blue = 191
                green = 95
                red = 0
            
                cv2.rectangle(frame, (left,top), (right,bottom), (blue, green, red), 4)
            elif "American Robin" in object_name:
            
                blue = 54
                green = 101
                red = 163
            
                cv2.rectangle(frame, (left,top), (right,bottom), (blue, green, red), 4) 
            elif "Red-breasted Nuthatch" in object_name:
            
                blue = 58
                green = 95
                red = 196
            
                cv2.rectangle(frame, (left,top), (right,bottom), (blue, green, red), 4)
            elif "Black-headed Grosbeak" in object_name:
            
                blue = 0
                green = 106
                red = 188
            
                cv2.rectangle(frame, (left,top), (right,bottom), (blue, green, red), 4)
            elif "House Sparrow" in object_name:
            
                blue = 10
                green = 66
                red = 89
            
                cv2.rectangle(frame, (left,top), (right,bottom), (blue, green, red), 4)
            else:
            
                blue = 10
                green = 255
                red = 0
            
                cv2.rectangle(frame, (left,top), (right,bottom), (blue, green, red), 4)  
                
            # Draw label
            label = '%s: %d%%' % (object_name, int(scores[0][object_count]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 2) # Get font size
            label_ymin = max(top, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (left-1, label_ymin-labelSize[1]-10), (left+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (left-1, label_ymin-7), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2) # Draw label text
            
            currentSpecies = processedClasses
                     
            object_count += 1
                    
        # Create variables for drawing metrics on frames
        countLabel = 'Bird Count: '+str(object_count)  # Example: 5
        countLabelSize = cv2.getTextSize(countLabel, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 2)
        # print(countLabelSize[1])
        
        countSpeciesFrames = "Frame Threshold " + "[" + str(min_frames_thresh) + "]: " + str(species_frame_count)  # Example: 5
        countSpeciesFramesSize = cv2.getTextSize(countSpeciesFrames, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 2)
        # print(countLabelSize[1])
        
        countSpeciesArray = 'Current Species: '+str(currentSpecies)  # Example: [Bald Eagle, Black-capped Chickadee, Steller's Jay]
        countSpeciesArraySize = cv2.getTextSize(countSpeciesArray, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 2)
        # print(countSpeciesArraySize[0][0])
        
        # Draw white boxes for metrics
        cv2.rectangle(blk, (20, 10), (210, 40), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
        
        if species_frame_count >= 0:
            cv2.rectangle(blk, (20, 40), (40 + countSpeciesFramesSize[0][0], 70), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
        
        if len(currentSpecies) >= 0:
            cv2.rectangle(blk, (20, 70), (40 + countSpeciesArraySize[0][0], 100), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in

        frame = cv2.addWeighted(frame, 1.0, blk, 0.30, 1)
        
        # Put text in boxes
        cv2.putText(frame, countLabel, (30, 31), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2) # Draw object count text
        cv2.putText(frame, countSpeciesFrames, (30, 61), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2) # Draw object count text
        cv2.putText(frame, countSpeciesArray, (30, 91), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2) # Draw object count text  
        
        processedScores = []
        processedClasses = []
        processedBoxes = []
        
        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector', frame)
        
        frameCount += 1

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            video.release()
            cv2.destroyAllWindows()
            break
            
        if frameCount % captured_frames == 0 and object_count > 0:
                
            cv2.imwrite("B:/BirdBot/BirdModelTraining/train_images/" + training, trainframe)
        
            print (ET.tostring(root))
        
            xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
        
            with open('B:/BirdBot/BirdModelTraining/train_images/' + xmlFileName, "w") as f:
            
                print("wrote file!!! XML")
                f.write(xmlstr)
            
            a_file = open('B:/BirdBot/BirdModelTraining/train_images/' + xmlFileName, "r")
            
            lines = a_file.readlines()
            a_file.close()

            del lines[0]
        
            new_file = open('B:/BirdBot/BirdModelTraining/train_images/' + xmlFileName, "w+")
        
            for line in lines:
                new_file.write(line)

            new_file.close()
  
def RealTimeMode():

    # Name of the directory containing the object detection module we're using
    MODEL_NAME = 'BirdModel'

    min_score_thresh = 0.60
    min_frames_thresh = 28
    max_frames_thresh = 30
    NatureMode = False
    StreamingMode = True
    RecordMode = False
    
    # Grab path to current working directory
    CWD_PATH = os.getcwd()

    # Path to frozen detection graph .pb file, which contains the model that is used
    # for object detection.
    PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH,'data','object-detection.pbtxt')

    # Number of classes the object detector can identify
    NUM_CLASSES = 12

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
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(graph=detection_graph, config=config)

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

    # Open video input and set to 1080p
    video = cv2.VideoCapture(1)
    video.set(3, 1920)
    video.set(4, 1080)
    
    # Print if 1080p was set on camera, some DSLR cameras will not set to new resolution
    imW = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    imW2 = 1920
    print(imW)
    print(imW2)
    imH = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    imH2 = 1080
    print(imH)
    print(imH2)

    frameCount = 0
    species_frame_count = 0
    seen_species_array = []
    wrapped_seen_species = ''
    approval_species_array = []
    timer_species_array = []

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('realtime.avi', fourcc, 30.0, (imW, imH)) # Make sure (width,height) is the shape of input frame from video
    
    # cv2.startWindowThread()
    # cv2.namedWindow("Object detector")

    while StreamingMode is True:
    
        while NatureMode is False:

            # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
            # i.e. a single-column array, where each item in the column has the pixel RGB value
            ret, frame = video.read()
            frame = cv2.resize(frame,(imW2, imH2), 0, 0, interpolation=cv2.INTER_LINEAR)
            # frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_expanded = np.expand_dims(frame_rgb, axis=0)

            # Perform the actual detection by running the model with the image as input
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: frame_expanded})
            
            object_count = 0
            
            # START PROCESSING THE DETECTION DATA WITH MY CODE
            processedScores = []
            processedClasses = []
            processedBoxes = []
            currentSpecies = []
            blk = np.zeros(frame.shape, np.uint8)
            
            now = datetime.now()
            current_time = now.strftime("%H-%M-%S")
            
            cv2.rectangle(blk, (20, imH-20), (160, imH-50), (255, 255, 255), cv2.FILLED)
            
            # if (scores[0][object_count] < min_score_thresh):
                # print("No birds detected")
                
            if frameCount % max_frames_thresh == 0:
                approval_species_array = []
                species_frame_count = 0
            
            # IF SCORE ARRAY IS GREATER THAN MIN SCORE AND LESS THAN 1 ADD TO PROCESS ARRAY
            while ((scores[0][object_count] > min_score_thresh) and (scores[0][object_count] <= 1.0)):
                
                object_name = str(category_index.get(classes[0][object_count]).get('name'))
            
                processedScores.append(str(scores[0][object_count]))
                processedClasses.append(str(category_index.get(classes[0][object_count]).get('name')))
                processedBoxes.append(str(boxes[0][object_count]))
                
                # print('Predictions: ' + str(object_count))
                # print(str(category_index.get(classes[0][object_count]).get('name')))
                
                # print(np.squeeze(boxes[0][object_count])) 
                
                top = int(max(1, imH * boxes[0][object_count][0]))
                # print(top)
                left = int(max(1, imW * boxes[0][object_count][1]))
                # print(left)
                bottom = int(min(imH, imH * boxes[0][object_count][2]))
                # print(bottom)
                right = int(min(imW, imW * boxes[0][object_count][3]))
                # print(right)
                
                if "Song Sparrow" in object_name:
                
                    blue = 10
                    green = 255
                    red = 0
                    
                    cv2.rectangle(frame, (left,top), (right,bottom), (blue, green, red), 4)
                elif "Chestnut-backed Chickadee" in object_name:
                
                    blue = 24
                    green = 70
                    red = 87
                    
                    cv2.rectangle(frame, (left,top), (right,bottom), (blue, green, red), 4)
                elif "European Starling" in object_name:
                
                    blue = 96
                    green = 96
                    red = 96
                
                    cv2.rectangle(frame, (left,top), (right,bottom), (blue, green, red), 4)
                elif "Golden-crowned Sparrow" in object_name:
                
                    blue = 0
                    green = 205
                    red = 205
                    
                    cv2.rectangle(frame, (left,top), (right,bottom), (blue, green, red), 4)
                elif "Spotted Towhee" in object_name:
                
                    blue = 47
                    green = 38
                    red = 29
                
                    cv2.rectangle(frame, (left,top), (right,bottom), (47, 38, 29), 4)
                elif "American Goldfinch" in object_name:
                
                    blue = 150
                    green = 255
                    red = 255
                
                    cv2.rectangle(frame, (left,top), (right,bottom), (blue, green, red), 4)
                elif "Black-capped Chickadee" in object_name:
                        
                    blue = 23
                    green = 29
                    red = 29
               
                    cv2.rectangle(frame, (left,top), (right,bottom), (blue, green, red), 4)
                elif "Dark-eyed Junco" in object_name:
                
                    blue = 57
                    green = 47
                    red = 45
                
                    cv2.rectangle(frame, (left,top), (right,bottom), (blue, green, red), 4)
                elif "Pine Siskin" in object_name:
                
                    blue = 137
                    green = 110
                    red = 97
                
                    cv2.rectangle(frame, (left,top), (right,bottom), (blue, green, red), 4)
                elif "House Finch" in object_name:
                
                    blue = 212
                    green = 170
                    red = 255
                
                    cv2.rectangle(frame, (left,top), (right,bottom), (blue, green, red), 4)
                elif "Steller's Jay" in object_name:
                
                    blue = 191
                    green = 95
                    red = 0
                
                    cv2.rectangle(frame, (left,top), (right,bottom), (blue, green, red), 4)
                elif "American Robin" in object_name:
                
                    blue = 54
                    green = 101
                    red = 163
                
                    cv2.rectangle(frame, (left,top), (right,bottom), (blue, green, red), 4) 
                elif "Red-breasted Nuthatch" in object_name:
                
                    blue = 58
                    green = 95
                    red = 196
                
                    cv2.rectangle(frame, (left,top), (right,bottom), (blue, green, red), 4)
                elif "Black-headed Grosbeak" in object_name:
                
                    blue = 0
                    green = 106
                    red = 188
                
                    cv2.rectangle(frame, (left,top), (right,bottom), (blue, green, red), 4)
                elif "House Sparrow" in object_name:
                
                    blue = 10
                    green = 66
                    red = 89
                
                    cv2.rectangle(frame, (left,top), (right,bottom), (blue, green, red), 4)
                else:
                
                    blue = 10
                    green = 255
                    red = 0
                
                    cv2.rectangle(frame, (left,top), (right,bottom), (blue, green, red), 4)  
                    
                # Draw label on bird bounding boxes
                label = '%s: %d%%' % (object_name, int(scores[0][object_count]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 2) # Get font size
                label_ymin = max(top, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (left-1, label_ymin-labelSize[1]-10), (left+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (left-1, label_ymin-7), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2) # Draw label text
                
                currentSpecies = processedClasses
                
                approval_species_array.append(object_name)
                
                if object_name in approval_species_array:
                    
                    species_frame_count = approval_species_array.count(object_name)
                    
                    if species_frame_count >= min_frames_thresh and object_name not in seen_species_array:

                        seen_species_array.append(object_name)
                        wrapped_seen_species = textwrap.wrap('Seen Species: '+str(seen_species_array), width=70)
                    
                    # print(object_name + ": " + str(approval_species_array.count(object_name)))
                         
                object_count += 1
            
            # Draws the white box for wrapped seen species         
            for i, line in enumerate(wrapped_seen_species):
                            
                textsize = cv2.getTextSize(line, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 2)[0]

                gap = textsize[1] + 16

                y = 121 + i * gap
                x = 40 + textsize[0]
                
                if i >= 0:
                    cv2.rectangle(blk, (20, 100), (x, y+10), (255, 255, 255), cv2.FILLED)
            
            # Create variables for drawing metrics on frames
            countLabel = 'Bird Count: '+str(object_count)  # Example: 5
            countLabelSize = cv2.getTextSize(countLabel, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 2)
            # print(countLabelSize[1])
            
            countSpeciesFrames = "Frame Threshold " + "[" + str(min_frames_thresh) + "]: " + str(species_frame_count)  # Example: 5
            countSpeciesFramesSize = cv2.getTextSize(countSpeciesFrames, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 2)
            # print(countLabelSize[1])
            
            countSpeciesArray = 'Current Species: '+str(currentSpecies)  # Example: [Bald Eagle, Black-capped Chickadee, Steller's Jay]
            countSpeciesArraySize = cv2.getTextSize(countSpeciesArray, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 2)
            # print(countSpeciesArraySize[0][0])
            
            # countSeenSpeciesArray = 'Seen Species: '+str(seen_species_array) # Example: [Bald Eagle, Black-capped Chickadee, Steller's Jay]
            # countSeenSpeciesArraySize = cv2.getTextSize(countSeenSpeciesArray, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 2)
            # print(countSeenSpeciesArraySize[0][0])
            
            # Draw white boxes for metrics
            cv2.rectangle(blk, (20, 10), (210, 40), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            
            if species_frame_count >= 0:
                cv2.rectangle(blk, (20, 40), (40 + countSpeciesFramesSize[0][0], 70), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            
            if len(currentSpecies) >= 0:
                cv2.rectangle(blk, (20, 70), (40 + countSpeciesArraySize[0][0], 100), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in

            # if len(seen_species_array) >= 0:
                # cv2.rectangle(blk, (20, 100), (40 + countSeenSpeciesArraySize[0][0], 130), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in

            frame = cv2.addWeighted(frame, 1.0, blk, 0.40, 1)
            
            # Put text in boxes
            cv2.putText(frame, countLabel, (30, 31), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2) # Draw object count text
            cv2.putText(frame, countSpeciesFrames, (30, 61), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2) # Draw object count text
            cv2.putText(frame, countSpeciesArray, (30, 91), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2) # Draw object count text
            # cv2.putText(frame, countSeenSpeciesArray, (30, 121), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2) # Draw object count text
            cv2.putText(frame, current_time, (30, imH-30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)

            for i, line in enumerate(wrapped_seen_species):
                            
                textsize = cv2.getTextSize(line, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 2)[0]

                gap = textsize[1] + 16

                y = 121 + i * gap
                x = int((frame.shape[1] - textsize[0]) / 2)

                cv2.putText(frame, line, (30, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,0), 2)        
            
            processedScores = []
            processedClasses = []
            processedBoxes = []
            
            # Draw the results of the detection (aka 'visulaize the results') - ARRAY Based Method, probably faster than what I'm doing above, but less control.
            # vis_util.visualize_boxes_and_labels_on_image_array(
                # frame,
                # np.squeeze(boxes),
                # np.squeeze(classes).astype(np.int32),
                # np.squeeze(scores),
                # category_index,
                # use_normalized_coordinates=True,
                # line_thickness=1,
                # min_score_thresh=0.50)
            
            if RecordMode is True:
            
                out.write(frame) 
            
            # All the results have been drawn on the frame, so it's time to display it.
            # cv2.resizeWindow("Object detector", imW2,imH2)
            cv2.imshow('Object detector', frame)
            
            frameCount += 1
            
            # print("Nature Mode: " + str(NatureMode))
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
            
                StreamingMode = False
                cv2.destroyAllWindows()
                break
            
            if cv2.waitKey(1) & 0xFF == ord('n'):
            
                NatureMode = True
                
            if cv2.waitKey(1) & 0xFF == ord('r'):
            
                RecordMode = True
        
        while NatureMode is True:
            
            ret, frame = video.read()
        
            # print("Nature Mode: " + str(NatureMode))
        
            cv2.imshow('Object detector', frame)
        
            if cv2.waitKey(10) & 0xFF == ord('q'):
            
                StreamingMode = False
                cv2.destroyAllWindows()
                break
        
            if cv2.waitKey(10) & 0xFF == ord('n'):
            
                NatureMode = False

def processVideo():
    filename = filedialog.askopenfilename(initialdir = "B:/BirdBot/BirdModel/models/research/object_detection", title = "Select a Video File",filetypes = (("all video format","*.mp4*"),("all files","*.*")))
    BirdBotVideo(filename)
    
def generateSpeciesData():
    filename = filedialog.askopenfilename(initialdir = "B:/BirdBot/BirdModel/models/research/object_detection", title = "Select a Video File",filetypes = (("all video format","*.mp4*"),("all files","*.*")))
    BirdBotGenerateXML(filename)
    
# Create the root window
window = Tk()

# Set window title
window.title('BirdBot ML Software Explorer')

# Set window size
window.geometry("700x500")

#Set window background color
window.config(background = "white")

# Create a File Explorer label
label_file_explorer = Label(window,
                            text = "BirdBot ML Software Explorer - Please Use Buttons Below!",
                            width = 100, height = 4,
                            fg = "blue")
                        

button_process_video = Button(window,
                        text = "Process Video",
                        command = processVideo)
                        
button_realtime_mode = Button(window,
                        text = "Real-Time Mode",
                        command = RealTimeMode)
                        
button_get_species = Button(window,
                        text = "Generate Species Data",
                        command = generateSpeciesData)

button_exit = Button(window,
                    text = "Exit",
                    command = exit)

# Grid method is chosen for placing
# the widgets at respective positions
# in a table like structure by
# specifying rows and columns
label_file_explorer.grid(row = 0, column = 0, pady = 10)

button_process_video.grid(row = 1, column = 0, columnspan = 2, pady = 5)

button_realtime_mode.grid(row = 2, column = 0, columnspan = 2, pady = 5)

button_get_species.grid(row = 3, column = 0, columnspan = 2, pady = 5)

button_exit.grid(row = 4, column = 0, columnspan = 2, pady = 5)

# Let the window wait for any events
window.mainloop()