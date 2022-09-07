import numpy as np
import cv2
from matplotlib import pyplot as plt
import pywt

def show_side_by_side(im1, im2):
    plt.subplot(121), plt.imshow(im1)
    plt.subplot(122), plt.imshow(im2)
    plt.show()


def get_clahe(img):
    image_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # The declaration of CLAHE
    # clipLimit -> Threshold for contrast limiting
    clahe = cv2.createCLAHE(clipLimit=5)
    final_img = clahe.apply(image_bw) + 30
    return final_img


def get_rgb(img):
    r = image.copy()
    # set green and red channels to 0
    r[:, :, 1] = 0
    r[:, :, 2] = 0

    g = image.copy()
    # set blue and red channels to 0
    g[:, :, 0] = 0
    g[:, :, 2] = 0

    b = image.copy()
    # set blue and green channels to 0
    b[:, :, 0] = 0
    b[:, :, 1] = 0

    return r, g, b


def fuseCoeff(cooef1, cooef2, method):
    if (method == 'mean'):
        cooef = (cooef1 + cooef2) / 2
    elif (method == 'min'):
        cooef = np.minimum(cooef1, cooef2)
    elif (method == 'max'):
        cooef = np.maximum(cooef1, cooef2)
    else:
        cooef = []

    return cooef

def channelTransform(ch1, ch2, shape):
    cooef1 = pywt.dwt2(ch1, 'db5', mode='periodization')
    cooef2 = pywt.dwt2(ch2, 'db5', mode='periodization')
    cA1, (cH1, cV1, cD1) = cooef1
    cA2, (cH2, cV2, cD2) = cooef2

    cA = (cA1 + cA2) / 2
    cH = (cH1 + cH2) / 2
    cV = (cV1 + cV2) / 2
    cD = (cD1 + cD2) / 2
    fincoC = cA, (cH, cV, cD)
    outImageC = pywt.idwt2(fincoC, 'db5', mode='periodization')
    outImageC = cv2.resize(outImageC, (shape[0], shape[1]))
    return outImageC


def fusion(img1, img2):
    # Params
    FUSION_METHOD = 'mean'  # Can be 'min' || 'max || anything you choose according theory

    # Read the two image
    I1 = img1
    I2 = img2

    ## Seperating channels
    iR1 = I1.copy()
    iR1[:, :, 1] = iR1[:, :, 2] = 0
    iR2 = I2.copy()
    iR2[:, :, 1] = iR2[:, :, 2] = 0

    iG1 = I1.copy()
    iG1[:, :, 0] = iG1[:, :, 2] = 0
    iG2 = I2.copy()
    iG2[:, :, 0] = iG2[:, :, 2] = 0

    iB1 = I1.copy()
    iB1[:, :, 0] = iB1[:, :, 1] = 0
    iB2 = I2.copy()
    iB2[:, :, 0] = iB2[:, :, 1] = 0

    shape = (I1.shape[1], I1.shape[0])
    # Wavelet transformation on red channel
    outImageR = channelTransform(iR1, iR2, shape)
    outImageG = channelTransform(iG1, iG2, shape)
    outImageB = channelTransform(iB1, iB2, shape)

    outImage = I1.copy()
    outImage[:, :, 0] = outImage[:, :, 1] = outImage[:, :, 2] = 0
    outImage[:, :, 0] = outImageR[:, :, 0]
    outImage[:, :, 1] = outImageG[:, :, 1]
    outImage[:, :, 2] = outImageB[:, :, 2]

    outImage = np.multiply(np.divide(outImage - np.min(outImage), (np.max(outImage) - np.min(outImage))), 255)
    outImage = outImage.astype(np.uint8)

    return outImage

def get_inhanced_image(image):
    # Resizing the image for compatibility
    image = cv2.resize(image, (500, 600))

    r, g, b = get_rgb(image)
    r_c = get_clahe(r)
    g_c = get_clahe(g)
    b_c = get_clahe(b)

    rgb = image.copy()
    # set blue and green channels to 0
    rgb[:, :, 0] = b_c
    rgb[:, :, 1] = g_c
    rgb[:, :, 2] = r_c
    # show_side_by_side(image[..., ::-1], rgb)

    blur_img = cv2.GaussianBlur(rgb, (0, 0), 5)
    usm = cv2.addWeighted(rgb, 1.5, blur_img, -0.5, 0)
    # show_side_by_side(rgb, usm)

    xxx = fusion(image[..., ::-1], usm)
    # show_side_by_side(usm, xxx)

    # denoising of image saving it into dst image
    dst = cv2.fastNlMeansDenoisingColored(xxx, None, 10, 10, 7, 15)
    return dst
    # show_side_by_side(image[..., ::-1], dst)