import os
import numpy as np
import json
from PIL import Image

def detect_red_light(I):
    '''
    This function takes a numpy array <I> and returns a list <bounding_boxes>.
    The list <bounding_boxes> should have one element for each red light in the
    image. Each element of <bounding_boxes> should itself be a list, containing
    four integers that specify a bounding box: the row and column index of the
    top left corner and the row and column index of the bottom right corner (in
    that order). See the code below for an example.

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''


    bounding_boxes = [] # This should be a list of lists, each of length 4. See format example below.
    bounding_boxes_transpose = [] # This is incase their bounding boxes go other way? Based on Slack

    '''
    BEGIN YOUR CODE
    '''

    '''
    As an example, here's code that generates between 1 and 5 random boxes
    of fixed size and returns the results in the proper format.
    '''

    picture_width, picture_height = I.size
    #picture_width = 640
    #picture_height = 480

    match_dictionary = {}

    height = 0
    for j in range(0, picture_height - light1_height,10):
        for i in range(0, picture_width - light1_width, 10):
            #print(j)
            location = (i, height + j, i + light1_width, height + j + light1_height)
            picture_match_location = I.crop(location)
            picture_match_vector = np.asarray(picture_match_location).flatten()
            normed_picture_match_vector = picture_match_vector/(np.linalg.norm(picture_match_vector, ord=2))
            match_prob = np.inner(normed_picture_match_vector, normed_light1)
            match_dictionary[location] = match_prob
            if(match_prob > 0.88):
                bounding_boxes.append(location)
                bounding_boxes_transpose.append((height + j, i, height + j + light1_height, i + light1_width)) #



    '''
    END YOUR CODE
    '''

    for i in range(len(bounding_boxes)):
        assert len(bounding_boxes[i]) == 4

    return bounding_boxes

test_path = 'C:/data/CS 148/test_images'
test_names = sorted(os.listdir(test_path))
light1 = Image.open(os.path.join(test_path, test_names[3]))
light1_vector = np.asarray(light1).flatten()
light1_width, light1_height = light1.size
normed_light1 = light1_vector/np.linalg.norm(light1_vector, ord=2)





# set the path to the downloaded data:
data_path = 'C:/data/CS 148/RedLights2011_Medium'

# set a path for saving predictions:
preds_path = 'C:/data/hw01_preds'
os.makedirs(preds_path,exist_ok=True) # create directory if needed

# get sorted list of files:
file_names = sorted(os.listdir(data_path))

# remove any non-JPEG files:
file_names = [f for f in file_names if '.jpg' in f]

preds = {}
for k in range(len(file_names)):
    print(k)

    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names[k]))

    # convert to numpy array:
    #I = np.asarray(I)

    preds[file_names[k]] = detect_red_light(I)

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds.json'),'w') as f:
    json.dump(preds,f)
