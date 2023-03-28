import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import operator
import pytorch
from PIL import Image, ImageStat


def imagefile(file_paths: list)-> str:
    """This function applies PIL verify to validate images and return all the image formats.

    Args:
        file_paths: a list of files paths containing image files.
    
    Returns:
        print a statement if all images are valid and as well as all the formats found.
    """

    all_formats = []

    for file in file_paths:
        
        img = Image.open(file)

        try:
            img.verify()
            format = img.format
            all_formats.append(format)

        except Exception:
            print('Invalid Image Type: {}'.format(file))

    unique = set(all_formats)

    print('Images are valid and formats found are {}'.format(unique))

def imagemode(file_paths: list)-> str:
    """This function checks the folder of images for it's image mode

    Args:
        file_paths: a list of files paths containing image files.

    Returns:
        print a statement if all images are valid and as well as all the modes found
    """

    all_modes = []

    for file in file_paths:

        img = Image.open(file)
        mode = img.mode
        all_modes.append(mode)

    unique = set(all_modes)

    print('Images mode found are {}'.format(unique))

def minmax_height(file_paths: list)-> str:
    """This function calculates the min and max height of all the images in a list.

    Args:
        file_paths: a list of file paths containing image files.

    Return:
        print a statement of the min and max height in a list of images.
    """

    heights = []

    for file in file_paths:

        img = Image.open(file)
        height = img.height
        heights.append(height)

    print('Min Height is {} and Max Height is {}'.format(min(heights),max(heights)))

def minmax_width(file_paths: list)-> str:
    """This function calculates the min and max width of all the images in a list.

    Args:
        file_paths: a list of file paths containing image files.

    Return:
        print a statement of the min and max width in a list of images.

    """

    widths = []

    for file in file_paths:

        img = Image.open(file)
        width = img.width
        widths.append(width)

    print('Min Width is {} and Max Width is {}'.format(min(widths),max(widths)))

def avg_mean(file_paths: list)-> float:
    """This function calculates the average mean of all images in a list.

    Args:
        file_paths: a list of file paths containing image files.
        
    Return:
        average: the average mean of the images in the list
    """

    accumulate_mean = np.zeros(3)

    for file in file_paths:

        img = Image.open(file)
        stat = ImageStat.Stat(img)
        file_mean = np.array(stat.mean)
        accumulate_mean += file_mean

    average = accumulate_mean/len(file_paths)

    return average

def whiteblack_ratio(file_paths: list)-> float:
    """This function calculates the average white/black pixels ratio of a list of images.

    Args:
        file_paths: a list of file paths containing image files.

    Return:
        ratio: average white/black ratio across all the images in the list
    """

    ratio = 0

    for file in file_paths:

        img = Image.open(file)
        data = np.array(img.getdata())
        count = np.unique(data, return_counts=True)

        if count[0][0] != 0:
            print('Expected [0,255] is incorrect')
        if count[0][1] != 255:
            print('Expected [0,255] is incorrect')

        black = count[1][0]
        white = count[1][1]

        ratio += white/black

    return ratio/len(file_paths)

def wb_ratio_df(file_paths: list)-> pd.DataFrame:
    """This function calculates the white and black pixels ratio of a list of images

    Args:
        file_paths: a list of file paths containing image files.

    Return:
        df: a Dataframe of the image index and it's white back ratio
    """

    df = pd.DataFrame(columns=['Index','Ratio'])

    i = 1

    for file in file_paths:

        img = Image.open(file)
        data = np.array(img.getdata())
        count = np.unique(data, return_counts=True)

        if count[0][0] != 0:
            print('Expected [0,255] is incorrect')
        if count[0][1] != 255:
            print('Expected [0,255] is incorrect')

        black = count[1][0]
        white = count[1][1]

        ratio = white/black
        
        new_row = pd.DataFrame([[i,ratio]], columns=['Index','Ratio'])
        df = pd.concat([df, new_row], ignore_index=True)

        i += 1

    return df

def black_df(file_paths: list)-> pd.DataFrame:
    """This function calculates the black pixel count of all images in a list

    Args:
        file_paths: a list of file paths containing image files.

    Return:
        df: a Dataframe of the index and black pixel count of all images in a list
    """

    df = pd.DataFrame(columns=['Index','Black'])

    i = 1

    for file in file_paths:

        img = Image.open(file)
        data = np.array(img.getdata())
        count = np.unique(data, return_counts=True)

        if count[0][0] != 0:
            print('Expected [0,255] is incorrect')

        black = count[1][0]
        new_row = pd.DataFrame([[i,black]], columns=['Index','Black'])
        df = pd.concat([df, new_row], ignore_index=True)

        i += 1

    return df

def white_df(file_paths: list)-> pd.DataFrame:
    """This function calculates the white pixel count of all images in a list

    Args:
        file_paths: a list of file paths containing image files.

    Return:
        df: a Dataframe of the index and white pixel count of all images in a list
    """
    
    df = pd.DataFrame(columns=['Index','White'])

    i = 1

    for file in file_paths:

        img = Image.open(file)
        data = np.array(img.getdata())
        count = np.unique(data, return_counts=True)

        if count[0][1] != 255:
            print('Expected [0,255] is incorrect')

        white = count[1][1]
        new_row = pd.DataFrame([[i,white]], columns=['Index','White'])
        df = pd.concat([df, new_row], ignore_index=True)

        i += 1

    return df

# Think of ways to make mask_path optional
def get_rgb_df(image_path: str, mask_path: str)-> pd.DataFrame:
    """This function creates a df of pixel counts in each of the 3 channels.

    Args:
        image_path: str of an image path
        mask_path: str of a mask path

    Return:
        rgb_df: a dataframe of pixel counts of each of the 3 channels of 1 image
    """

    image = Image.open(image_path)
    mask = Image.open(mask_path)

    r, g, b = image.split()

    r_channel_df = pd.DataFrame(data=np.array(r.histogram(mask=mask)), 
                                                columns=['r_count'])

    g_channel_df = pd.DataFrame(data=np.array(g.histogram(mask=mask)), 
                                                columns=['g_count'])

    b_channel_df = pd.DataFrame(data=np.array(b.histogram(mask=mask)),
                                                columns=['b_count'])

    rgb_df = pd.concat([r_channel_df, g_channel_df, b_channel_df], 
                                                axis=1, ignore_index=True)

    rgb_df = rgb_df.rename(columns={0: "r_counts", 1: "g_counts", 2:"b_counts"})

    return rgb_df

def get_avg_rgb_df(image_paths: list, mask_paths:list)-> pd.DataFrame:
    """This calculates average pixel value of each channel across a list of images

    Args:
        image_paths: a list of color images paths.
        mask_paths: a list of corresponding mask paths. This helps the function
            to calculate pixel values only where mask value is non-zero.
    
    Return:
        rgb_df: a dataframe of average pixel values by each channels of a list of image
    """

    image_list = []
    mask_list = []

    for image_path in image_paths:
        image = Image.open(image_path)
        image_list.append(image)

    for mask_path in mask_paths:
        mask = Image.open(mask_path)
        mask_list.append(mask)

    i = 0

    r_accu = np.zeros(256)
    g_accu = np.zeros(256)
    b_accu = np.zeros(256)

    for i in range(len(image_list)):
        r, g, b = image_list[i].split()
        r_hist = np.array(r.histogram(mask=mask_list[i]))
        g_hist = np.array(g.histogram(mask=mask_list[i]))
        b_hist = np.array(b.histogram(mask=mask_list[i]))

        r_accu += r_hist
        g_accu += g_hist
        b_accu += b_hist

        i += 1

    r_avg = r_accu/len(image_list)
    g_avg = g_accu/len(image_list)
    b_avg = b_accu/len(image_list)

    r_channel_df = pd.DataFrame(data=r_avg, columns=['r_count'])
    g_channel_df = pd.DataFrame(data=g_avg, columns=['g_count'])
    b_channel_df = pd.DataFrame(data=b_avg, columns=['b_count'])

    rgb_df = pd.concat([r_channel_df, g_channel_df, b_channel_df], 
                                        axis=1, ignore_index=True)

    rgb_df = rgb_df.rename(columns={0: "r_counts", 1: "g_counts", 2:"b_counts"})

    return rgb_df

def get_all_rgb_dfs(image_mask_dict: dict)-> pd.DataFrame:
    """This function creates a df of all individual image pixel counts of each channel

    Args:
        image_paths: a list of color images paths.
        mask_paths: a list of corresponding mask paths. This helps the function
            to calculate pixel values only where mask value is non-zero.
    
    Return:
        all_rgb_df: a dataframe of all pixel counts by each channels 
            of a each individual image in a list.
    """

    all_rgb_dfs = []

    key_list = list(image_mask_dict.keys())

    for i in range(len(image_mask_dict)):
        rgb_df = get_rgb_df(key_list[i], image_mask_dict[key_list[i]])
        all_rgb_dfs.append(rgb_df)

    return all_rgb_dfs

def rgb_deviate_pct(rgb_dfs: list, 
                    avg_rgb_df:pd.DataFrame, 
                    total_pixels: int)-> list:
    """This function subtracts image pixel count by avg pixel count 
        to find deviation percentage.
    
    Args:
        rgb_dfs: a list of dataframes of individual images rgb pixel counts.
        avg_rgb_df: a dataframe of average rgb pixel counts
        total_pixels: an integer of the total number of pixels in the image.

    Return:
        list_pct: a list of float indicating the deviation in pixel count 
            by each image represented by 1 dataframe each.
    """

    list_pct = []

    for rgb_df in rgb_dfs:
        rgb_deviation = rgb_df.subtract(avg_rgb_df).abs()
        r_d, g_d, b_d = rgb_deviation.sum()
        deviation_pct = (r_d + g_d + b_d)/total_pixels
        list_pct.append(deviation_pct)

    return list_pct

#####

def display_images(filelist: list):
    """Function to display images using ipyplot

    Args:
        filelist (list): list of image paths, can be RGB or greyscale
    """
    images = [cv2.imread(image)[...,::-1] for image in filelist]
    rows = 2
    cols = 7
    for i in range(0, len(images), rows*cols):
        fig = plt.figure(figsize=(40,10))
        for j in range(0, rows*cols):
            fig.add_subplot(rows,cols, j+1)
            plt.imshow(images[i+j])

    plt.show()  
    plt.tight_layout()

def get_pixels_array(filelist: list)-> np.array:
    """Function to obtain pixels array of image in filelist
    
    Args:
        filelist (list): list of image paths, can be RGB or greyscale
    
    Returns:
        np.array(list_pixels_arr): Array of list of arrays with pixel values
        of each image
    """

    list_pixels_arr = []

    for img in filelist:
        im = torch.array(Image.open(img))
        list_pixels_arr.append(im)

    return list_pixels_arr

def get_pixels_mean(pixel_array: np.array)-> np.array:
    """This function gets the pixel mean from an array of image arrays.

    Args:
        pixel_array: An array of Image pixel arrays.

    Return:
        pixel_mean: An array of Image pixel mean array.
    """
    
    pixel_mean = np.mean(pixel_array, axis=0)

    return pixel_mean

def get_pixels_std(pixel_array: np.array)-> np.array:
    """This function gets the pixel std from an array of image arrays

    Args:
        pixel_array: An array of Image pixel arrays

    Return:
        pixel_std: An array of Image pixel_std array.
    """

    pixel_std = np.std(pixel_array, axis=0)

    return pixel_std

def get_average_image(pixel_mean_array: np.array)-> Image:
    """This function converts image pixel array into a displayable image format

    Args:
        pixel_mean_array: An array of Image pixel mean array.

    Return:
        average_image: an image ready to be displayed. 
    """

    average_image = Image.fromarray(np.array(pixel_mean_array, dtype=np.uint8))

    return average_image

def get_deviation(filelist: list, pixel_mean: np.array, channel: int)-> np.array:
    """Calculate pixel-wise deviation of current pixel against pixel_mean and 
    append to list of total_deviation. Converts total_deviation to absolute total_deviation
    Calculates sum of deviation across all pixels for each image, 
    with axis (1,2) for greyscale images and axis (1,2,3) for RGB images

    Args:
        filelist (list): list of image paths, can be RGB or greyscale
            pixel_mean (np.array): mean pixels across all images and channels 
            wrt height and weight, with shape (height, width, channels = 3 if RGB) 
            or (height, width) for greyscale

    Returns:
        abs_total_deviation (np.array): absolute total_deviation 
            with shape (no.of images, height, width, channels = 3 if RGB) 
            or (no.of images, height, width) for greyscale
            sum_deviations (np.array): single array of shape (no.of images)
    """
    total_deviation = []

    for img in filelist:
        current_pixel = np.array(Image.open(img))
        pixel_deviation = current_pixel - pixel_mean
        total_deviation.append(pixel_deviation)
    
    total_deviation = np.array(total_deviation)
    abs_total_deviation = abs(total_deviation)
    
    if channel == 1:
        sum_deviations = np.sum(abs_total_deviation, axis= (1,2))
    elif channel == 3:
        sum_deviations = np.sum(abs_total_deviation, axis= (1,2,3))
    
    return abs_total_deviation, sum_deviations

def min_max_deviation(sum_deviations: np.array, filelist)-> int:
    """Obtain lower and upper bounds of deviations 

    Args:
        sum_deviations (np.array): sum of deviation across all pixels for each image

    Returns:
        min_index (int): index of image that has the minimum deviation from average pixels
        max_index (int): index of image that has the maximum deviation from average pixels
        mean_deviation (array): mean of sum_deviations
    """
    
    min_index, min_value = min(enumerate(sum_deviations), key=operator.itemgetter(1))
    max_index, max_value = max(enumerate(sum_deviations), key=operator.itemgetter(1))
    mean_deviation = np.mean(sum_deviations)
    print(f'min index is {min_index} and the corresponding min value is {min_value} \
        \nmax index is {max_index} and corresponding max value is {max_value}')
    print(f'Mean deviation is {mean_deviation}')
    print(f'Min image is {filelist[min_index]} and max image is {filelist[max_index]}')
    
    return min_index, max_index, mean_deviation


def plot_image(deviations: np.array, total_pixels: int)-> pd.DataFrame:
    """function to compute dataframe showing 

    Args:
        deviations (np.array): array of sum_deviations
        total_pixels (_type_): image height x width x channel

    Returns:
        df: dataframe consisting of sum_deviations and avg_pixel_deviation for each image
    """

    df = pd.DataFrame(deviations, columns = ['sum_deviations'])

    df = df.assign(avg_pixel_deviation = lambda x: ((x['sum_deviations']/total_pixels)))
    ax = df.plot(y="avg_pixel_deviation", kind="bar", figsize=(9, 8))
    ax.set_xlabel("Image index")
    plt.show()
    
    return df

def show_actual_images(index: int, distribution: np.array)-> Image:
    """Function to show image

    Args:
        index (int): index of to iterate through list
        distribution (np.array): pixel distribution of images, can be pixel_distribution or abs_total_deviated

    Returns:
        output_image: image
    """
    
    array= np.array(distribution[index,:,:], dtype=np.uint8)
    output_image = Image.fromarray(array)
    
    return output_image
