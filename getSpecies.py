# The steps implemented in the object detection sample code: 
# 1. for an image of width and height being (w, h) pixels, resize image to (w', h'), where w/h = w'/h' and w' x h' = 262144
# 2. resize network input size to (w', h')
# 3. pass the image to network and do inference
# (4. if inference speed is too slow for you, try to make w' x h' smaller, which is defined with DEFAULT_INPUT_SIZE (in object_detection.py or ObjectDetection.cs))
import sys
import tensorflow as tf
import numpy as np
import moviepy.video.io.ImageSequenceClip
from PIL import Image
from object_detection import ObjectDetection
import cv2
import importlib.util
import os
import argparse
import time
import xml.etree.cElementTree as ET
from xml.dom import minidom

PhotoName = 'BBN2'
MODEL_FILENAME = 'model.tflite'
LABELS_FILENAME = 'labels.txt'
VIDEO_PATH = '/home/pi/tflite1/'+PhotoName+'.mp4'
min_conf_threshold = 0.3
fps=30
Day = "-Day3"

class TFLiteObjectDetection(ObjectDetection):
    """Object Detection class for TensorFlow Lite"""
    def __init__(self, model_filename, labels):
        super(TFLiteObjectDetection, self).__init__(labels)
        self.interpreter = tf.lite.Interpreter(model_path=model_filename)
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]['index']
        self.output_index = self.interpreter.get_output_details()[0]['index']

    def predict(self, preprocessed_image):
        inputs = np.array(preprocessed_image, dtype=np.float32)[np.newaxis, :, :, (2, 1, 0)]  # RGB -> BGR and add 1 dimension.

        # Resize input tensor and re-allocate the tensors.
        self.interpreter.resize_tensor_input(self.input_index, inputs.shape)
        self.interpreter.allocate_tensors()
        
        self.interpreter.set_tensor(self.input_index, inputs)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_index)[0]


def main(image_filename):
    # Load labels
    with open(LABELS_FILENAME, 'r') as f:
        labels = [l.strip() for l in f.readlines()]

    od_model = TFLiteObjectDetection(MODEL_FILENAME, labels)
    interpreter = tf.lite.Interpreter(MODEL_FILENAME)
    
    interpreter.allocate_tensors()
    
    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    
    floating_model = (input_details[0]['dtype'] == np.float32)
    
    # print(floating_model)
    
    input_mean = 127.5
    input_std = 127.5

    image = Image.open(image_filename)
    predictions = od_model.predict_image(image)
    
    x = 0
    boxes = []
    tagId = []
    classes = []
    scores = []
    species_array = []
    seen_species_array = []
    timer_species_array = []
    
    while x < len(predictions):
        
        # print("This is prediction #" + str(x) + ":")
        # print(predictions[x])
        
        classes.append(predictions[x]['tagName'])
        # print("Class Name: " + classes[x])
        
        boxes.append(predictions[x]['boundingBox'])
        # print("Boxes: " + str(boxes[x]))
        
        tagId.append(predictions[x]['tagId'])
        # print("Index ID: " + str(tagId[x]))
        
        scores.append(round(predictions[x]['probability'],3))
        # print("Probability: " + str(scores[x]))
        
        x += 1
        object_count = 0
    
    # print("This is the prediction array:")
    # print(boxes)
    # print(tagId)
    # print(scores)
    
    video = cv2.VideoCapture(VIDEO_PATH)
    imW = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    # print(imW)
    imH = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # print(imH)
    
    while(video.isOpened()):
        
        # Acquire frame and resize to expected shape [1xHxWx3]
        ret, frame = video.read()
        if not ret:
            print('Reached the end of the video!')
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)
        
        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std
        
        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()
        
        # frame
        currentframe = 0
        
        while (True):

            # reading from frame
            ret, frame = video.read()
            trainframe = frame
            
            # if video is still left continue creating images
            name = PhotoName + str(currentframe) + '.jpg'
            
            cv2.imwrite(name, frame)
            
            if ret:
                
                videoImage = Image.open("/home/pi/tflite1/BirdBotTL/python/"+name)
                
                predictions = od_model.predict_image(videoImage)
                
                x = 0
                
                while x < len(predictions):
        
                    # print("This is prediction #" + str(x) + ":")
                    # print(predictions[x])
        
                    classes.append(predictions[x]['tagName'])
                    # print("Class Name: " + classes[-1])
        
                    boxes.append(predictions[x]['boundingBox'])
                    # print("Boxes: " + str(boxes[-1]))
        
                    tagId.append(predictions[x]['tagId'])
                    # print("Index ID: " + str(tagId[-1]))
        
                    scores.append(round(predictions[x]['probability'],3))
                    # print("Probability: " + str(scores[-1]))
        
                    x += 1
                    object_count = 0
                
                # Write training photo if there is no bird detected
                # print(predictions)
                # if not predictions and currentframe % 25 == 0:
                    # training = 'null' + str(currentframe) + 'train' + '.jpg'
                    # cv2.imwrite("/home/pi/tflite1/BirdBotTL/python/Training/"+training, trainframe)
                    
                objectHolder = {}
                
                root = ET.Element("annotation")
        
                # UNCOMMENT for training photos
                training = PhotoName + str(currentframe) + Day + '.jpg'
                
                # Write training photo every 5 frames on successful ID
                    
                folder = ET.SubElement(root, "folder")
                folder.text = "images"
                filename = ET.SubElement(root, "filename") 
                filename.text = training
                path = ET.SubElement(root, "path")  
                path.text = "images/"+training
                source = ET.SubElement(root, "source")
                
                database = ET.SubElement(source, "database")
                database.text = "Unspecified"
                
                size = ET.SubElement(root, "size")
                
                width = ET.SubElement(size, "width")
                width.text = str(imW)
                height = ET.SubElement(size, "height")
                height.text = str(imH)
                depth = ET.SubElement(size, "depth")
                depth.text = str(3)
                    
                for i in range(len(scores)):
                    if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
        
                        # print(boxes[i])
        
                        object_name = labels[int(tagId[i])] # Look up object name from "labels" array using class index
                        
                        ymin = int(max(1, imH * boxes[i]['top']))
                        # print(ymin)
                        xmin = int(max(1, boxes[i]['left'] * imW))
                        # print(xmin)
                        ymax = int(min(imH,(imH * (boxes[i]['top'] + boxes[i]['height']))))
                        # print(ymax)
                        xmax = int(min(imW, ((imW * (boxes[i]['left'] +  boxes[i]['width'])))))
                        # print(xmax)
                   
                        objectHolder["objectXml"+str(i)] = ET.SubElement(root, "object")
                        
                        objectHolder["nameXml"+str(i)] = ET.SubElement(objectHolder["objectXml"+str(i)], "name")
                        objectHolder["nameXml"+str(i)].text = object_name
                        objectHolder["pose"+str(i)] = ET.SubElement(objectHolder["objectXml"+str(i)], "pose")
                        objectHolder["pose"+str(i)].text = "Unspecified"
                        objectHolder["truncated"+str(i)] = ET.SubElement(objectHolder["objectXml"+str(i)], "truncated") 
                        objectHolder["truncated"+str(i)].text = "Unspecified"
                        objectHolder["difficult"+str(i)] = ET.SubElement(objectHolder["objectXml"+str(i)], "difficult")
                        objectHolder["difficult"+str(i)].text = "Unspecified"
                        
                        objectHolder["bndbox"+str(i)] = ET.SubElement(objectHolder["objectXml"+str(i)], "bndbox") 
                        
                        objectHolder["xminXml"+str(i)] = ET.SubElement(objectHolder["bndbox"+str(i)], "xmin")
                        objectHolder["xminXml"+str(i)].text = str(xmin)
                        objectHolder["yminXml"+str(i)] = ET.SubElement(objectHolder["bndbox"+str(i)], "ymin") 
                        objectHolder["yminXml"+str(i)].text = str(ymin)
                        objectHolder["xmaxXml"+str(i)] = ET.SubElement(objectHolder["bndbox"+str(i)], "xmax") 
                        objectHolder["xmaxXml"+str(i)].text = str(xmax)
                        objectHolder["ymaxXml"+str(i)] = ET.SubElement(objectHolder["bndbox"+str(i)], "ymax") 
                        objectHolder["ymaxXml"+str(i)].text = str(ymax)
                        print(objectHolder["ymaxXml"+str(i)].text)
                        
                        print (ET.tostring(root))
                        
                        # DRAW RECT on FRAME at (cords) (cords), with (color) and THICKNESS
                        # COLOR is specific to bird species! ADD new bird colors!
                        if "Song Sparrow" in object_name:
                            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 4)
                        elif "Chestnut-backed Chickadee" in object_name:
                            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (24, 70, 87), 4)
                        elif "European Starling" in object_name:
                            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (96, 96, 96), 4)
                        elif "Golden-crowned Sparrow" in object_name:
                            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (0, 205, 205), 4)
                        elif "Spotted Towhee" in object_name:
                            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (47, 38, 29), 4)
                        elif "American Goldfinch" in object_name:
                            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (150, 255, 255), 4)
                        elif "Black-capped Chickadee" in object_name:
                            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (23, 29, 29), 4)
                        elif "Dark-eyed Junco" in object_name:
                            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (57, 47, 45), 4)
                        elif "Pine Siskin" in object_name:
                            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (137, 110, 97), 4)
                        elif "House Finch" in object_name:
                            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (212, 170, 255), 4)
                        elif "Steller's Jay" in object_name:
                            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (191, 95, 0), 4)
                        else:
                            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 4)                    

                        # Draw label
                        label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_TRIPLEX, 0.7, 2) # Get font size
                        label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                        cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                        cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                        
                        object_count += 1
                        
                if currentframe % 2 == 0 and object_count > 0:
                    
                    cv2.imwrite("/home/pi/tflite1/BirdBotTL/python/Training/" + training, trainframe)
                    
                    print (ET.tostring(root))
                    
                    xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
                    
                    with open('/home/pi/tflite1/BirdBotTL/python/Training/' + PhotoName + str(currentframe) + Day + '.xml', "w") as f:
                        
                        f.write(xmlstr)
                        
                    a_file = open('/home/pi/tflite1/BirdBotTL/python/Training/' + PhotoName + str(currentframe) + Day + '.xml', "r")
                        
                    lines = a_file.readlines()
                    a_file.close()
                    
                    del lines[0]
                    
                    new_file = open('/home/pi/tflite1/BirdBotTL/python/Training/' + PhotoName + str(currentframe) + Day + '.xml', "w+")
                    
                    for line in lines:
                        new_file.write(line)
                    
                    new_file.close()
                                            
                # RESET VARIABLES
                boxes = []
                tagId = []
                classes = []
                scores = []
                object_count = 0
                species_array = []
                
                # if currentframe == 0:
                #     seen_species_array = []
                
                # cv2.imshow('Object detector', frame)
                
                # print('Creating...' + name)                
                # writing the extracted images
                # cv2.imwrite(name, frame)
                # os.replace("/home/pi/tflite1/BirdBotTL/python/"+name, "/home/pi/tflite1/BirdBotTL/python/Images/"+name)
                
                # increasing counter so that it will
                # show how many frames are created
                currentframe += 1
                
                os.remove(name)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            else:
                break
    
    # CLEAR VIDEO AND WINDOWS 
    video.release()
    cv2.destroyAllWindows()
    

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('USAGE: {} image_filename'.format(sys.argv[0]))
    else:
        main(sys.argv[1])
