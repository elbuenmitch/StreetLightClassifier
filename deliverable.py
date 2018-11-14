## TODO: Create a brightness feature that takes in an RGB image and outputs a feature vector and/or value
## This feature should use HSV colorspace values
def create_feature(rgb_image):
    
	## TODO: Convert image to HSV color space
    masked_image = mask_image(rgb_image, 1, 3, 8, "hsv")
    
    ## TODO: Create and return a feature value and/or vector
    feature = brigtness_per_row(masked_image)
    
    return feature

####--------- BORRAR!!!!! ---------------------
#Parameters: Estimation tresholds, upper and lower bounds for the masks, HSV layer index for the masking
red_t = 12     #treshold for the label estimation
yel_t = 18     #treshold for the label estimation
lower = 30     #Some number
upper = 256    #Some number
layer = 1      #HSV layer used for the mask
crop_r= 3      #vertical crop
crop_c= 8      #horizontal crop
type_ = "hsv"  #Color format to be used in the masking
####--------- BORRAR!!!!! ---------------------

##------------------------------------------------------------------------
##-------------------------- HELPER METHODS! -----------------------------
##------------------------------------------------------------------------

## Masks an image based on the following params:
# upper_: upper boundary for the pixel's intensity
# lower_: lower boundary for the pixel's intensity
# layer: index on the HSV standard that represents the layer on which the mask is applied: 0, 1 or 2.
# type_: "hsv" or "lab" for the color format
def mask_image(rgb_image, layer_, crop_row, crop_col, type_):
    copy_ = np.copy(rgb_image)
    if type_ == "hsv":
        converted = rgb2hsv(copy_)
    if type_ == "lab":
        converted = rgb2lab(copy_)
    channel = converted[:,:,layer_]
    
    #define the boundaries
    low, high = dynamic_mask_boundries(copy_,layer_, type_)
    new_mask = cv2.inRange(channel, low, high)
    copy_[new_mask == 0] = [0,0,0]
    cropped_copy_ = crop_image(copy_, crop_row, crop_col)
    return cropped_copy_

#compresses an image into a vector by summing up the values on each row
def brigtness_per_row(rgb_image):
    oneD_image = []
    ids = []
    for i in range(len(rgb_image)):
        oneD_image.append(np.sum(rgb_image[i,:]))
        ids.append(i)
    return oneD_image

#Determines the tresholds where the Red, Yellow and Green light are located vertically in the Y-axis of the image. 
#Returns the row of the image where the division changes from Red to Yellow, and from Yellow to Green.
def dynamic_mask_boundries(rgb_image, layer, type_):
    if type_ == "hsv":
        converted = rgb2hsv(rgb_image)
    if type_ == "lab":
        converted = rgb2lab(rgb_image)
    layer_ = converted[:,:,layer]
    low = (np.min(layer_) + np.max(layer_)) / 3
    high = np.max(layer_)
    return int(low), int(high)

#Crops an image symetrically a number of pixel rows and colums
def crop_image(rgb_image, crop_row, crop_col):
    if crop_row != 0 and crop_col != 0:
        image_crop = rgb_image[crop_row:-crop_row, crop_col:-crop_col, :]
    if crop_row == 0:
        image_crop = rgb_image[:, crop_col:-crop_col, :]
    if crop_col == 0:
        image_crop = rgb_image[crop_row:-crop_row, :, :]
    return image_crop

#Returns an HSV colorspace image from the RGB one in the parameters
def rgb2hsv(rgb_image):
    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
#Returns an LAB colorspace image from the RGB one in the parameters
def rgb2lab(rgb_image):
    return cv2.cvtColor(rgb_image, cv2.COLOR_BGR2LAB)

#calculates the tresholds to vertically locate the light
def calc_label_treshnolds(image):
    height = len(image)
    red = int(height/3)+2
    yel = int(2*height/3)-2
    return red, yel



