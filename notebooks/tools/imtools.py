import numpy as np
from PIL import Image
from scipy.ndimage import filters
import os
import functools

import itertools
import matplotlib.pyplot as plt



project_path = os.path.abspath(os.pardir)


def create_folders(new_folder_path):
    def inner_function(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            os.system("if [ ! -d " + new_folder_path +
                      " ]; then mkdir -p " + new_folder_path + "; fi")
            func(*args, **kwargs)
        return wrapper
    return inner_function


def transform_random(angle, sigma):
    def inner_function(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_output = func(*args, **kwargs)
            for random_imgs in list(func_output.keys()):
                for elem in func_output[random_imgs]:
                    img = Image.open(kwargs["folder_path"] + "/" +
                                     random_imgs + "/" + elem)
                    img.rotate(angle).save(kwargs["folder_path"]+"/"+
                                           random_imgs+"/"+"r"+elem)

                    im = np.array(img.convert('L'))
                    im_blur = filters.gaussian_filter(im, sigma)
                    Image.fromarray(im_blur).save(kwargs["folder_path"]+"/"+
                                                         random_imgs+"/"+"b"+elem)
        return wrapper
    return inner_function


@create_folders(project_path+"/data/gray_scale_pictures")
def resize_img_list(folder_path, new_folder_path, size=(50, 50)):
    for number in os.listdir(folder_path):
        os.system("mkdir " + new_folder_path + "/" + number)
        for pctr in os.listdir(folder_path + "/" + number):
            img = Image.open(folder_path + "/" + number + "/" + pctr)
            img.convert('L').resize(size).save(new_folder_path + "/" + number + "/" + pctr)


@transform_random(45, 0.75)
def choose_random_img_set(folder_path, size=10):
    l = {}
    for nmbr in os.listdir(folder_path):
        l[nmbr] = np.random.choice(os.listdir(folder_path+"/"+nmbr), size=size, replace=False)
    return l


def histeq(im_array, nbr_bins=256):
    """
    Histogram equalization of a grayscale image
    :param im:
    :param nbr_bins:
    :return:
    """
    imhist, bins = np.histogram(im_array.flatten(), nbr_bins, normed=True)
    cdf = imhist.cumsum()
    cdf = 255 * cdf / cdf[-1]
    im2 = np.interp(im_array.flatten(), bins[:-1], cdf)
    return im2.reshape(im_array.shape), cdf


@create_folders(project_path+"/data/hist_eq")
def final_prepar_img_list(folder_path):
    for number in os.listdir(folder_path):
        for pctr in os.listdir(folder_path + "/" + number):
            img = Image.open(folder_path + "/" + number + "/" + pctr)
            vector, _ = histeq(np.array(img))
            Image.fromarray(np.uint8(vector)).save(project_path+"/data/hist_eq/"+number+"_"+pctr)


def pca(X):
    """
    Principal Component Analysis
    input: X, matrix with training data stored as flattened arrays in rows
    return: projection matrix (with important dimensions first), variance
    and mean
    :param X:
    :return:
    """

    # get dimensions
    num_data, dim = X.shape

    # center data
    mean_X = X.mean(axis=0)
    X = X - mean_X

    if dim > num_data:
        # PCA - compact trick used
        M = np.dot(X,X.T) # covariance matrix
        e, EV = np.linalg.eigh(M) # eigenvalues and eigenvectors
        tmp = np.dot(X.T,EV).T # this is the compact trick
        V = tmp[::-1] # reverse since last eigenvectors are the ones we want
        S = np.sqrt(e)[::-1] # reverse since eigenvalues are in increasing order
    for i in range(V.shape[1]):
      V[:, i] /= S
    else:
        # PCA - SVD used
        U, S, V = np.linalg.svd(X)
        V = V[:num_data] # only makes sense to return the first num_data

    # return the projection matrix, the variance and the mean
    return V, S, mean_X


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()