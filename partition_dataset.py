""" usage: partition_dataset.py [-h] [-i IMAGEDIR] [-o OUTPUTDIR] [-r RATIO] [-x]
Partition dataset of images into training and testing sets
optional arguments:
  -h, --help            show this help message and exit
  -i IMAGEDIR, --imageDir IMAGEDIR
                        Path to the folder where the image dataset is stored. If not specified, the CWD will be used.
  -o OUTPUTDIR, --outputDir OUTPUTDIR
                        Path to the output folder where the train and test dirs should be created. Defaults to the same directory as IMAGEDIR.
  -r RATIO, --ratio RATIO
                        The ratio of the number of test images over the total number of images. The default is 0.1.
  -x, --xml             Set this flag if you want the xml annotation files to be processed and copied over.
"""

import os
import re
from shutil import copyfile
import argparse
import math
import random

from os import path
from os.path import exists

def iterate_dir(source, dest, ratio, copy_xml):
    source = source.replace('\\', '/')
    dest = dest.replace('\\', '/')
    train_dir = os.path.join(dest, 'train')
    test_dir = os.path.join(dest, 'test')

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    images = [f for f in os.listdir(source)
              if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(.jpg|.jpeg|.png)$', f)]
			  
    xmlFiles = [f for f in os.listdir(source)
              if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(.xml)$', f)]

    num_images = len(images)
    print(num_images)
    num_xmlFiles = len(xmlFiles)
    print(num_xmlFiles)
    num_test_images = math.ceil(ratio*num_images)
	
    # CHECKS IF XML FILE EXISTS FOR JPG FILE, IF NOT THEN DELETE FILES (BUT RANDOM)
    for i in range(num_test_images):
    
        idx = random.randint(0, len(images)-1)
        filename = images[idx]
        
        xml_filename = filename[:-4] + ".xml"
        xml_path = "B:/BirdBot/BirdModelTraining/birdbotwindows/images/" + str(xml_filename)
		
        # print("B:\BirdBot\BirdModelTraining\birdbotwindows\images\\" + xml_filename)
        # print(str(path.exists("B:\BirdBot\BirdModelTraining\birdbotwindows\images\\" + xml_filename)))
		
        print(str(path.exists(xml_path)))
        
        if str(path.exists(xml_path)) is "True":
        
            # print(xml_filename)
            copyfile(os.path.join(source, xml_filename),
                    os.path.join(test_dir, xml_filename))
                    
            # print(filename)
            copyfile(os.path.join(source, filename),
                    os.path.join(test_dir, filename))
                    
        else:
            print(str(filename) + " DID NOT HAVE XML FILE!")
            try:    
                os.remove("B:/BirdBot/BirdModelTraining/birdbotwindows/images/" + str(filename))
                print("Deleted file")
            except:
                print(xml_filename)
                print("Error or file not found")
    
	# CHECKS IF XML FILE EXISTS FOR JPG FILE, IF NOT THEN DELETE FILES
    for filename in images:
        
        xml_filename = filename[:-4] + ".xml"
        xml_path = "B:/BirdBot/BirdModelTraining/birdbotwindows/images/" + str(xml_filename)
		
        print(str(path.exists(xml_path)))
        
        if str(path.exists(xml_path)) is "True":
        
            # print(xml_filename)
            copyfile(os.path.join(source, xml_filename),
                    os.path.join(train_dir, xml_filename))
                    
            # print(filename)
            copyfile(os.path.join(source, filename),
                    os.path.join(train_dir, filename))
                    
        else:
            print(filename + " DID NOT HAVE XML FILE!")
            try:    
                os.remove("B:/BirdBot/BirdModelTraining/birdbotwindows/images/" + str(filename))
                print("Deleted JPG file")
            except:
                print("Error or JPG file not found")
	
	# CHECKS FOR EXTRA XML FILES THAT HAVE NO JPG FILE			
    # for i in range(num_xmlFiles):
        
        # xml_filename = xmlFiles[i]
        # filename = xml_filename = filename[:-4] + ".jpg"
		
        # if str(path.exists("B:\BirdBot\BirdModelTraining\birdbotwindows\images\\" + xml_filename)) is "True":
            # print('FILE FOUND: ' + str(filename))
        # else:
            # print(str(filename) + " DID NOT HAVE .JPG FILE!")
            # try:    
                # os.remove("B:\BirdBot\BirdModelTraining\birdbotwindows\images\\" + str(xml_filename))
                # print("Deleted XML file")
            # except:
                # print("Error or XML file not found")


def main():

    # Initiate argument parser
    parser = argparse.ArgumentParser(description="Partition dataset of images into training and testing sets",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-i', '--imageDir',
        help='Path to the folder where the image dataset is stored. If not specified, the CWD will be used.',
        type=str,
        default="images"
    )
    parser.add_argument(
        '-o', '--outputDir',
        help='Path to the output folder where the train and test dirs should be created. '
             'Defaults to the same directory as IMAGEDIR.',
        type=str,
        default="images"
    )
    parser.add_argument(
        '-r', '--ratio',
        help='The ratio of the number of test images over the total number of images. The default is 0.1.',
        default=0.20,
        type=float)
    parser.add_argument(
        '-x', '--xml',
        help='Set this flag if you want the xml annotation files to be processed and copied over.',
        action='store_true'
    )
    args = parser.parse_args()

    if args.outputDir is None:
        args.outputDir = args.imageDir

    # Now we are ready to start the iteration
    iterate_dir(args.imageDir, args.outputDir, args.ratio, args.xml)


if __name__ == '__main__':
    main()