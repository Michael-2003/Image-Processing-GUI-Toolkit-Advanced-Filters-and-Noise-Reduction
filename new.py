import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from scipy.fftpack import fft , fft2 ,fftshift , ifftshift , ifft2

import cv2
def filter():
# Load the image and convert to grayscale
    image = plt.imread('noise.png')
    gray_image = np.mean(image, axis=-1)

    # Compute the 2D Fourier Transform of the image
    f_transform = np.fft.fft2(gray_image)
    f_transform_shifted = np.fft.fftshift(f_transform)

    # Visualize the magnitude spectrum
    magnitude_spectrum = np.abs(f_transform_shifted)
    plt.imshow(np.log(1 + magnitude_spectrum), cmap='gray')
    plt.title('Log Magnitude Spectrum')
    plt.show()

    # Identify peaks corresponding to the periodic noise
    # (You may need a more sophisticated peak identification algorithm)
    peaks = np.where(magnitude_spectrum < 700)

    # Set the threshold based on the identified peaks
    threshold = np.max(magnitude_spectrum[peaks])

    # Create a filter based on the threshold
    filter = np.ones_like(gray_image)
    filter[magnitude_spectrum > threshold] = 0

    # Apply the filter in the frequency domain
    f_transform_filtered = f_transform_shifted * filter

    # Inverse Fourier Transform to get the filtered image
    image_filtered = np.fft.ifft2(np.fft.ifftshift(f_transform_filtered)).real

    # Display the original, magnitude spectrum, and filtered images
    plt.subplot(121), plt.imshow(gray_image, cmap='gray'), plt.title('Original Image')

    plt.subplot(122), plt.imshow(image_filtered, cmap='gray'), plt.title('Filtered Image')
    plt.show()


def notch_clown():
    f = cv2.imread('Lenna.jpeg',0)


    # add periodic noise to the image

    from scipy import misc
    import numpy as np

    shape = f.shape[0], f.shape[1]
    noise = np.zeros(shape, dtype='float64')

    x, y = np.meshgrid(range(0, shape[0]), range(0, shape[1]))
    s=1+np.sin(x+y/1.5)
    noisyImage=((f)/128+s)/4


#get image in frequency domain
    spectrum = fft2(noisyImage)
    spectrum = fftshift(spectrum)

    freq_noisyImage=20*np.log(np.abs(spectrum))

    noise_filter = np.ones(shape=(474,474))
    noise_filter [175:205 , 160:185]=0
    noise_filter [280:310 , 160:185]=0
    noise_filter [175:205 , 300:320]=0
    noise_filter [280:310 , 300:320]=0

    # impelemet the notch filter

    denoised_image = ifft2(fftshift(spectrum*noise_filter))

    denoised_image_mag=np.abs(denoised_image)
    freq_denoised_mag=20*np.log(np.abs(spectrum))*noise_filter

    plt.subplot(121), plt.imshow(denoised_image_mag, cmap='gray'), plt.title('Original Image')

    plt.subplot(122), plt.imshow(freq_denoised_mag, cmap='gray'), plt.title('Filtered Image')
    plt.show()



def band_reject():

    band_filter = np.ones(shape=(512,512))

    x, y = np.meshgrid(np.arange(512), np.arange(512))
    center = (512//2, 512//2)

    radius = 30
    band_filter[ ((x - center[0])*2 + (y - center[1])**2 <= (radius*2))] = 0

    radius = 26
    band_filter[ ((x - center[0])*2 + (y - center[1])**2 <= radius*2) ] = 1
    denoised_image2 = ifft2(fftshift(spectrum * band_filter))
    denoised_image_magntiud2 = np.abs(denoised_image2)

    freq_denoied_img2 = 20 * np.log(np.abs(spectrum)) * band_filter

    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(14,7))

    ax0.imshow(denoised_image_magntiud2, cmap='gray')
    ax0.set_title("Noisy image")
    ax0.axis('off')


    ax1.imshow(freq_denoied_img2, cmap='gray')
    ax1.set_title("freq of noisy image")
    ax1.axis('off');

