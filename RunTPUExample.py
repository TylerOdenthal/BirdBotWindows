import os
import pathlib
import cv2
import numpy as np
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify
from PIL import Image

# Specify the TensorFlow model, labels, and image
script_dir = pathlib.Path(__file__).parent.absolute()
model_file = os.path.join(script_dir, 'BirdModelTPU/ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite')
label_file = os.path.join(script_dir, 'BirdModelTPU/tpulabelmap.txt')

# Initialize the TF interpreter
interpreter = edgetpu.make_interpreter(model_file)
interpreter.allocate_tensors()

# Open video file
video = cv2.VideoCapture('BBTT35.mp4')

frame_rate_calc = 1
freq = cv2.getTickFrequency()
print(freq)

imW = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
print(imW)
imH = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(imH)

while(video.isOpened()):
	
	ret, frame = video.read()
	
    #Start frame-rate timer
	t1 = cv2.getTickCount()
	
	if frame is None:
		video.release()
		cv2.destroyAllWindows()
		print("END OF VIDEO")
		break
	
	# frame = cv2.flip(frame, 1)
	frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame_expanded = np.expand_dims(frame_rgb, axis=0)
	frame_resize = cv2.resize(frame_rgb,(320, 320), 0, 0, interpolation=cv2.INTER_LINEAR)
	
	# Resize the image
	size = common.input_size(interpreter)
	# image = Image.open(frame).convert('RGB').resize(size, Image.ANTIALIAS)

	# Run an inference
	common.set_input(interpreter, frame_resize)
	interpreter.invoke()
	classes = classify.get_classes(interpreter, top_k=1)

	# Print the result
	labels = dataset.read_label_file(label_file)
	for c in classes:
	  print('%s: %.5f' % (labels.get(c.id, c.id), c.score))
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
            
                StreamingMode = False
                cv2.destroyAllWindows()
                break
				
	cv2.putText(frame, 'FPS: ' + str(frame_rate_calc), (30,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)
	
	# Calculate framerate
	t2 = cv2.getTickCount()
	time1 = (t2-t1)/freq
	frame_rate_calc= 1/time1
	print(frame_rate_calc)
	
		  
	cv2.imshow('Object detector', frame)
  
