#All the stuff that I coded but in the end submitted only a fraction. Here is the whole code!

#--------------- YO HICE ESTA CELDA! BORRELA PAPA!!! -----------------

##------------------------------------------------------------------------
##------------------------------ METHODS! --------------------------------
##------------------------------------------------------------------------

def dynamic_mask_boundries(rgb_image, layer, type_):
    if type_ == "hsv":
        converted = rgb2hsv(rgb_image)
    if type_ == "lab":
        converted = rgb2lab(rgb_image)
    layer_ = converted[:,:,layer]
    low = (np.min(layer_) + np.max(layer_)) / 3
    high = np.max(layer_)
    return int(low), int(high)

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

def crop_image(rgb_image, crop_row, crop_col):
    if crop_row != 0 and crop_col != 0:
        image_crop = rgb_image[crop_row:-crop_row, crop_col:-crop_col, :]
    if crop_row == 0:
        image_crop = rgb_image[:, crop_col:-crop_col, :]
    if crop_col == 0:
        image_crop = rgb_image[crop_row:-crop_row, :, :]
    return image_crop

#calculates the feature based on the brightness per row method
def calculate_feature(rgb_image, layer_, crop_row, crop_col, type_):
    #import pdb; pdb.set_trace()
    a = mask_image(rgb_image, layer_, crop_row, crop_col, type_)
    b = brigtness_per_row(a)
    return b.index(max(b))

# Gets the estimated label based on the brightness feature only
def estimate_label(image, red_t_, yel_t_, layer_, crop_row, crop_col, type_):
    estimation = "red"
    x = calculate_feature(image, layer_, crop_row, crop_col, type_)
    r_, y_ = calc_label_treshnolds(image)
    #if x > red_t_:
    #    estimation = "yellow"
    #if x > yel_t_:
    #    estimation = "green"
    if x >= r_:
        estimation = "yellow"
    if x >= y_:
        estimation = "green"
    return one_hot_encode(estimation)

#BORRAR
# Gets the estimated label based on the brightness feature only
def estimate_label2(image, red_t_, yel_t_, layer_, crop_row, crop_col):
    #import pdb; pdb.set_trace()
    estimation = "red"
    x = calculate_feature(image, layer_, crop_row, crop_col)
    r_, y_ = calc_label_treshnolds(image)
    #if x > red_t_:
    #    estimation = "yellow"
    #if x > yel_t_:
    #    estimation = "green"
    if x >= r_:
        estimation = "yellow"
    if x >= y_:
        estimation = "green"
    return one_hot_encode(estimation)

#calculates the tresholds to vertically locate the light
def calc_label_treshnolds(image):
    height = len(image)
    red = int(height/3)+2
    yel = int(2*height/3)-2
    return red, yel

def channels_from_image(image):
    channel_1 = image[:,:,0]
    channel_2 = image[:,:,1]
    channel_3 = image[:,:,2]
    return channel_1, channel_2, channel_3

#returns an array of tuples with: (image, label of the image, index in the standardized list) 
def get_greens_and_yellows(standardized_list):
    greens_yellows = []
    ids = []
    for i in range(len(STANDARDIZED_LIST)):
        lab = STANDARDIZED_LIST[i][1]
        if (lab == one_hot_encode("green") or lab == one_hot_encode("yellow")):
            greens_yellows.append((STANDARDIZED_LIST[i][0],lab,i))
    return greens_yellows 
    
def avg_color(rgb_image, channel):
    ch = rgb_image[:,:,channel]
    ave = np.average(ch[ch != 0])
    return ave 
    
def avg_color_list(list_of_rgb_images):
    red = []
    gre = []
    blu = []
    ids = []
    for i in range(len(list_of_rgb_images)):
        ima = list_of_rgb_images[i][0]
        red.append(avg_color(ima,0))
        gre.append(avg_color(ima,1))
        blu.append(avg_color(ima,2))
        ids.append(i)
    return red, gre, blu, ids
    
    
def rgb2hsv(rgb_image):
    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

def rgb2lab(rgb_image):
    return cv2.cvtColor(rgb_image, cv2.COLOR_BGR2LAB)

# @param hsv_rgb: "rgb" or "hsv" or "lab"
def show_image_per_channel(rgb_image, hsv_or_rgb):
    
    # Sepparate channels
    if hsv_or_rgb == "hsv":
        hsv = rgb2hsv(rgb_image)
        channel_1, channel_2, channel_3 = channels_from_image(hsv)
        chans = ["h","s","v"]
    if hsv_or_rgb == "rgb":
        channel_1, channel_2, channel_3 = channels_from_image(rgb_image)
        chans = ["r","g","b"]
    if hsv_or_rgb == "lab":
        hsv = rgb2lab(rgb_image)
        channel_1, channel_2, channel_3 = channels_from_image(rgb_image)
        chans = ["l","a","b"]
    
    # Plot the original image and the three channels
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,10))
    ax1.set_title('RGB Image')
    ax1.imshow(rgb_image)
    ax2.set_title(chans[0])
    ax2.imshow(channel_1)#, cmap='gray')
    ax3.set_title(chans[1])
    ax3.imshow(channel_2)#, cmap='gray')
    ax4.set_title(chans[2])
    ax4.imshow(channel_3)#, cmap='gray')

#BORRAR?
def enhance_contrast(rgb_image):
    copy_ = np.copy(rgb_image)
    #copy_[rgb_image < 5]
    return 0






#--------------- YO HICE ESTA CELDA! BORRELA PAPA!!! -----------------

#Parameters: Estimation tresholds, upper and lower bounds for the masks, HSV layer index for the masking
red_t = 12     #treshold for the label estimation
yel_t = 18     #treshold for the label estimation
lower = 30     #Some number
upper = 256    #Some number
layer = 1      #HSV layer used for the mask
crop_r= 3      #vertical crop
crop_c= 8      #horizontal crop
type_ = "hsv"  #Color format to be used in the masking

# Estimate labels for all images
estimations = []
estimations.clear()
ids_ = []
ids_.clear()
wrongs = []
wrongs.clear()
wrong_reds = []
wrong_reds.clear()
fatal_mistakes = []
fatal_mistakes.clear()
masked_images = []
features = []
wrong_greens = []
for i in range(len(STANDARDIZED_LIST)):
    ima = STANDARDIZED_LIST[i][0]
    lab = STANDARDIZED_LIST[i][1]
    ##ERASE! ----
    #maskey = 
    masked_images.append(mask_image(ima, layer, crop_r, crop_c, type_))
    ##ERASE! ----
    blurred = cv2.blur(ima,(2,2))
    estim = estimate_label(blurred, red_t, yel_t, layer, crop_r, crop_c, type_)
    estimations.append(estim)
    features.append(calculate_feature(blurred, layer, crop_r, crop_c, type_))
    ids_.append(i) 
    if estim != lab:
        wrongs.append(ima)
        if lab == one_hot_encode("red"):
            wrong_reds.append((ima,lab,estim))
            if estim == one_hot_encode("green"):
                fatal_mistakes.append((ima,lab,estim))
        if lab == one_hot_encode("green"):
            wrong_greens.append((ima,lab,estim))

plt.scatter(ids_,features)
                
print("Accuracy: "+str(1-(len(wrongs)/len(STANDARDIZED_LIST))))
print(str(len(wrongs))+" incorrectly labeled pictures in total")


#-------------



#--------------- YO HICE ESTA CELDA! BORRELA PAPA!!! -----------------
a = 0
misclassifier_red_ima = wrong_reds[a][0]
misclassifier_red_lab = wrong_reds[a][1]
misclassifier_red_est = wrong_reds[a][2]
#plt.imshow(misclassifier_red_ima)
print(str(misclassifier_red_lab)+" was expected but the result was "+str(misclassifier_red_est))
print(str(len(wrong_reds))+" misclassified Red Lights in total")

g = 0
misclassifier_green_ima = wrong_greens[a][0]
misclassifier_green_lab = wrong_greens[a][1]
misclassifier_green_est = wrong_greens[a][2]
#plt.imshow(misclassifier_red_ima)
print(str(misclassifier_green_lab)+" was expected but the result was "+str(misclassifier_green_est))
print(str(len(wrong_greens))+" misclassified Green Lights in total")

b = 0
fatal_mistakes_ima = fatal_mistakes[b][0]
fatal_mistakes_lab = fatal_mistakes[b][1]
fatal_mistakes_est = fatal_mistakes[b][2]
#plt.imshow(fatal_mistakes_ima)
print(str(fatal_mistakes_lab)+" was expected but the result was "+str(fatal_mistakes_est))
print(str(len(fatal_mistakes))+" Red Lights classified as green")

print("feature: "+str(calculate_feature(fatal_mistakes_ima, layer, crop_r, crop_c)))
blurred2 = cv2.blur(fatal_mistakes_ima,(6,6))
aa = mask_image(blurred2, layer, crop_r, crop_c)

#plt.imshow(misclassifier_red_ima)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,7))
ax1.set_title('fatal_mistakes_ima: '+str(a))
ax1.imshow(fatal_mistakes_ima)
ax2.set_title('Mask')
ax2.imshow(aa)#, cmap='gray')

show_image_per_channel(fatal_mistakes_ima,"lab")


lo, hi = dynamic_mask_boundries(fatal_mistakes_ima, 1)
print("low: "+str(lo))
print("hi: "+str(hi))


red_, yel_ = calc_label_treshnolds(fatal_mistakes_ima)
print(red_)
print(yel_)

loko = rgb2hsv(fatal_mistakes_ima)
print("lowest pixel in the S Channel: "+str(np.min(loko[:,:,1])))
print("highest pixel in the S Channel: "+str(np.max(loko[:,:,1])))
estimated_label = estimate_label2(blurred2, red_t, yel_t, 1, crop_r, crop_c)
oneDimage = brigtness_per_row(aa)
#plt.plot(oneDimage)
print(estimated_label)




#------------




# RGB channels
r,g,b = channels_from_image(fatal_mistakes_ima)

r_sum = avg_color(fatal_mistakes_ima,0)
g_sum = avg_color(fatal_mistakes_ima,1)
b_sum = avg_color(fatal_mistakes_ima,2)
print(r_sum)
print(g_sum)
print(b_sum)

    
show_image_per_channel(fatal_mistakes_ima,"rgb")
fatal_hsv = rgb2hsv(fatal_mistakes_ima)
print(fatal_hsv[12,8,1])

sobel_x3 = np.array([[ 0, 2, 0], 
                     [ 2, -8, 2], 
                     [ 0, 2, 0]])

# 3x3 array for edge detection
sobel_vertical = np.array([[-1, -1, -1], 
                           [ 0, 0, 0], 
                           [ 1, 1, 1]])
                   
## TODO: Create and apply a Sobel x operator
sobel_horizontal = np.array([[ -1, 0, 1], 
                             [ -1, 0, 1], 
                             [ -1, 0, 1]])

def apply_filter(rgb_image, sobel):
    return cv2.filter2D(rgb_image, -1, sobel)

filtered_image = apply_filter(fatal_mistakes_ima,sobel_x3)
#plt.imshow(filtered_image)
show_image_per_channel(filtered_image,"hsv")

def dynamic_crop(rgb_image, sobel_h, sobel_v):
    
    ##Horizontal edge finder
    filtered_image_h = apply_filter(rgb_image,sobel_h)
    height = int(len(filtered_image_h)/2)
    #Left
    left = 0
    av = filtered_image_h[height,left]
    for i in range(1,len(filtered_image_h)):
        new_average = (av + filtered_image_h[height,i])/2
        if new_average >
        
    #Right
    
    ##Vertical edge finder
    filtered_image_v = apply_filter(rgb_image,sobel_v)
    #Top
    
    #Bottom
    
    
    return 0



