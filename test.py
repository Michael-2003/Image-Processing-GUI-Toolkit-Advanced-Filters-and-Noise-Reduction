import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import tkinter as tk
from PIL import Image, ImageTk, ImageFilter, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import tkinter.filedialog
from scipy.ndimage import median_filter
import random
import cv2
from scipy import fftpack
from new import filter , notch_clown

class MyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("My Tkinter App")
        self.root.geometry("600x500")

        # Title
        title_label = tk.Label(root, text="Welcome to our app", font=("Helvetica", 16))
        title_label.pack(pady=10)

        # Dropdown Menu
        options = ["Add Salt and pepper noise","Remove Salt and Pepper", "Display histogram", "Histogram Equalization","Apply Sobel","Apply Laplace","Fourier Transform","Add Periodic Noise","Remove periodic noise mask","Remove periodic Notch","Custom highpass filter","Notch on specific picture"]
        self.selected_option = tk.StringVar()
        dropdown_menu = ttk.Combobox(root, textvariable=self.selected_option, values=options)
        dropdown_menu.pack(pady=10)


        self.image = None
        # Image Display
        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)

        # Buttons
        submit_button = tk.Button(root, text="Submit", command=self.submit_action)
        submit_button.pack(pady=10)

        upload_button = tk.Button(root, text="Upload Image", command=self.upload_image)
        upload_button.pack(pady=10)



        self.file_path=None
        self.salt_pepper_noisy=None #s&p image
        self.periodic_noisy=None #periodic normal
        self.noisy_fourier=None # fourier of periodic
        self.mask_coordinates=[]
        self.noise_fourier=None # fourier of periodic noise itself




    def display_image(self, img):
        img = ImageTk.PhotoImage(img)
        self.image_label.configure(image=img)
        self.image_label.image = img

    def submit_action(self):
        selected_option = self.selected_option.get()

        if selected_option == "Add Salt and pepper noise":
            self.open_salt_and_pepper_window()
        elif selected_option == "Remove Salt and Pepper":
            self.open_remove_salt_window()
        elif selected_option == "Display histogram":
            self.display_histogram()
        elif selected_option == "Histogram Equalization":
            self.histogram_equalization()
        elif selected_option == "Apply Sobel":
            self.apply_sobel()
        elif selected_option == "Apply Laplace":
            self.apply_laplace()
        elif selected_option == "Fourier Transform":
            self.display_fourier_transform()
        elif selected_option == "Add Periodic Noise":
            self.open_periodic_window()
        elif selected_option == "Remove periodic noise mask":
            self.mask_remove_periodic_noise()
        elif selected_option == "Remove periodic Notch":
            self.apply_notch_filter()
        elif selected_option == "Custom highpass filter":
            filter()
        elif selected_option == "Notch on specific picture":
            notch_clown()

    
    def open_salt_and_pepper_window(self):
        
        salt_and_pepper_window = tk.Toplevel(self.root)
        salt_and_pepper_window.title("Salt and pepper")

        white_degree_label = tk.Label(salt_and_pepper_window, text="Enter Amount of white pixels")
        white_degree_label.pack(pady=10)

        white_degree_entry = tk.Entry(salt_and_pepper_window)
        white_degree_entry.pack(pady=10)

        black_degree_label = tk.Label(salt_and_pepper_window, text="Enter Amount of black pixels")
        black_degree_label.pack(pady=10)

        black_degree_entry = tk.Entry(salt_and_pepper_window)
        black_degree_entry.pack(pady=10)

        submit_button = tk.Button(salt_and_pepper_window, text="Submit", command=lambda: self.apply_salt_and_pepper([white_degree_entry.get(),black_degree_entry.get()]))
        submit_button.pack(pady=10)
        
        
        

    def open_remove_salt_window(self):
        
        remove_window = tk.Toplevel(self.root)
        remove_window.geometry("200x200")
        remove_window.title("Salt and pepper")

        kernel_size_label = tk.Label(remove_window, text="Enter kernel size")
        kernel_size_label.pack(pady=10)

        kernel_size_entry = tk.Entry(remove_window)
        kernel_size_entry.pack(pady=10)

       

        submit_button = tk.Button(remove_window, text="Remove noise", command= lambda: self.remove_salt_pepper_noise(kernel_size_entry.get()))
        submit_button.pack(pady=10)
        return
    def remove_salt_pepper_noise(self,filtersize):
        if self.image:
            pixels = np.array(self.image)
            filter_size = int(int(filtersize))
            denoised_img = median_filter(pixels, size=filter_size)  # Apply median filter to remove salt and pepper noise
            denoised_img = Image.fromarray(denoised_img)

            self.display_image(denoised_img)
        

    def apply_salt_and_pepper(self, degree):
        if self.image:
            white_pixels = int(degree[0])
            black_pixels = int(degree[1])

            pixels = np.array(self.image)
            noisy_pixels = self.add_noise(pixels, white_pixels, black_pixels)
            noisy_img = Image.fromarray(noisy_pixels)
            self.salt_pepper_noisy=noisy_img.copy()
            plt.imsave('salt.png', noisy_img, cmap='gray')

            self.display_image(noisy_img)
    
    def add_noise(self, img, num_white_pixels, num_black_pixels):
        row, col = img.shape[:2]  # Getting the dimensions of the image

        for _ in range(num_white_pixels):
            y_coord = np.random.randint(0, row)  # Pick a random y coordinate
            x_coord = np.random.randint(0, col)  # Pick a random x coordinate
            img[y_coord][x_coord] = 255  # Color that pixel to white

        for _ in range(num_black_pixels):
            y_coord = np.random.randint(0, row)  # Pick a random y coordinate
            x_coord = np.random.randint(0, col)  # Pick a random x coordinate
            img[y_coord][x_coord] = 0  # Color that pixel to black

        return img
    



    def calculate_histogram(self, img):
        if img.mode == 'RGB':
            r, g, b = img.split()
            histogram_r = np.array(r.histogram())
            histogram_g = np.array(g.histogram())
            histogram_b = np.array(b.histogram())
            histogram = np.array([histogram_r, histogram_g, histogram_b])
        else:
            histogram = np.array(img.histogram())
            histogram = histogram.reshape(1, -1)  # Convert to a 2D array
        
        return histogram

    def plot_histogram(self, histogram):
        plt.figure(figsize=(6, 4))
        plt.title('Histogram')
        if histogram.shape[0] == 3:  # For RGB images
            colors = ('Red', 'Green', 'Blue')
            for i, color in enumerate(colors):
                plt.plot(histogram[i], color=color)
        else:  # For grayscale or other image modes
            plt.plot(histogram[0], color='black')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.show()

    def display_histogram(self):
        if self.image:
            histogram = self.calculate_histogram(self.image)
            self.plot_histogram(histogram)
    

    def histogram_equalization(self):
        def equalize_and_display(image):
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            hist_original = cv2.calcHist([image], [0], None, [256], [0, 256])

            equalized_image = cv2.equalizeHist(image)

            hist_equalized = cv2.calcHist([equalized_image], [0], None, [256], [0, 256])

            plt.subplot(2, 2, 1), plt.imshow(image, cmap='gray')
            plt.title('Original Image'), plt.xticks([]), plt.yticks([])

            plt.subplot(2, 2, 2), plt.imshow(equalized_image, cmap='gray')
            plt.title('Equalized Image'), plt.xticks([]), plt.yticks([])
            plt.imsave('equalizedimage.png', equalized_image, cmap='gray')


            plt.subplot(2, 2, 3)
            plt.plot(hist_original, color='black')
            plt.title('Histogram Before Equalization'), plt.xlim([0, 256])

            plt.subplot(2, 2, 4)
            plt.plot(hist_equalized, color='black')
            plt.title('Histogram After Equalization'), plt.xlim([0, 256])

            plt.show()

    
        your_image_array = cv2.imread(self.file_path)  
        equalize_and_display(your_image_array)
    def apply_sobel(self):
        sobel_window = tk.Toplevel(self.root)
        sobel_window.title("Apply Sobel Filter")

        # X-axis Sobel kernel size
        x_label = tk.Label(sobel_window, text="Enter X-axis Sobel Kernel Size:")
        x_label.pack(pady=5)
        x_entry = tk.Entry(sobel_window)
        x_entry.pack(pady=5)

        # Y-axis Sobel kernel size
        y_label = tk.Label(sobel_window, text="Enter Y-axis Sobel Kernel Size:")
        y_label.pack(pady=5)
        y_entry = tk.Entry(sobel_window)
        y_entry.pack(pady=5)

        # Apply button
        apply_button = tk.Button(sobel_window, text="Apply Sobel", command=lambda: self.apply_sobel_filter(x_entry.get(), y_entry.get()))
        apply_button.pack(pady=10)

    def apply_sobel_filter(self, x_kernel_size, y_kernel_size):
        if self.image:
            try:
                x_kernel = int(x_kernel_size)
                y_kernel = int(y_kernel_size)
                # Apply Sobel filter to the image
                sobel_img = self.apply_sobel_operation(self.image, x_kernel, y_kernel)
                # Display the filtered image
                self.display_image(sobel_img)
            except ValueError:
                print("Please enter valid kernel sizes.")
                
    def apply_sobel_operation(self, image, x_kernel, y_kernel):
        grayscale_img = image.convert('L') if image.mode != 'L' else image
        img_array = np.array(grayscale_img)
        
        
        sobel_x = cv2.Sobel(img_array, cv2.CV_64F, 1, 0, ksize=x_kernel)
                                                        
        sobel_y = cv2.Sobel(img_array, cv2.CV_64F, 0, 1, ksize=y_kernel)
        
        sobel_img = np.sqrt(sobel_x*2 + sobel_y*2)
        
        sobel_img = cv2.normalize(sobel_img, None, 0, 255, cv2.NORM_MINMAX)
        sobel_img = np.uint8(sobel_img)
        return Image.fromarray(sobel_img)

    def apply_laplace(self):
        laplace_window = tk.Toplevel(self.root)
        laplace_window.title("Apply Laplace Filter")
        kernel_label = tk.Label(laplace_window, text="Enter Laplace Kernel Size:")
        kernel_label.pack(pady=5)
        kernel_entry = tk.Entry(laplace_window)
        kernel_entry.pack(pady=5)
        apply_button = tk.Button(laplace_window, text="Apply Laplace", command=lambda: self.apply_laplace_filter(kernel_entry.get()))
        apply_button.pack(pady=10)

    def apply_laplace_filter(self, kernel_size):
        if self.image:
            try:
                laplace_kernel = int(kernel_size)
                laplace_img = self.apply_laplace_operation(self.image, laplace_kernel)
                self.display_image(laplace_img)
            except ValueError:
                print("Please enter a valid kernel size.")
    def apply_laplace_operation(self, image, kernel_size):
        grayscale_img = image.convert('L') if image.mode != 'L' else image
        img_array = np.array(grayscale_img)
        
        laplace_img = cv2.Laplacian(img_array, cv2.CV_64F, ksize=kernel_size)
        laplace_img = cv2.normalize(laplace_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        enhanced_img = cv2.convertScaleAbs(laplace_img)
        return Image.fromarray(enhanced_img)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])

        if file_path:
            self.file_path=file_path
            print(self.file_path)
            self.image = Image.open(file_path)
            self.image.thumbnail((220, 220))  
            photo = ImageTk.PhotoImage(self.image)

            self.image_label.configure(image=photo)
            self.image_label.image = photo
            
    def display_fourier_transform(self):
        if self.image:
            gray_img = self.image.convert('L')  
            pixels = np.array(gray_img)
            f_transform = np.fft.fft2(pixels)  
            f_transform_shifted = np.fft.fftshift(f_transform)  

            magnitude_spectrum = 20 * np.log(np.abs(f_transform_shifted))  

            magnitude_spectrum *= 255.0 / np.max(magnitude_spectrum)

            fourier_img = Image.fromarray(magnitude_spectrum.astype(np.uint8))
            self.noisy_fourier= fourier_img.copy()
            plt.imshow(fourier_img,cmap="gray")
            
            plt.title("Your Image")
            plt.axis("off")  
            plt.show()
            
            
    def open_periodic_window(self):
        
        remove_window = tk.Toplevel(self.root)
        remove_window.geometry("200x200")
        remove_window.title("Periodic noise")

        frequency_label = tk.Label(remove_window, text="Enter frequency")
        frequency_label.pack(pady=10)

        frequency_entry = tk.Entry(remove_window)
        frequency_entry.pack(pady=10)

        amplitude_label = tk.Label(remove_window, text="Enter amplitude of the sin function")
        amplitude_label.pack(pady=10)

        amplitude_entry = tk.Entry(remove_window)
        amplitude_entry.pack(pady=10)
       

        submit_button = tk.Button(remove_window, text="Add noise", command= lambda: self.periodic(int(frequency_entry.get()),int(amplitude_entry.get())))
        submit_button.pack(pady=10)
    def periodic(self,frequency,amplitude):

        def add_periodic_noise(image, frequency, amplitude):
            rows, cols = image.shape
            x = np.arange(cols)
            y = np.arange(rows)

            x, y = np.meshgrid(x, y)

            noise = amplitude * np.sin(2 * np.pi * frequency[0] * x / cols + 2 * np.pi * frequency[1] * y / rows)

            noisy_image = image + noise

            noisy_image = np.clip(noisy_image, 0, 255)

            self.periodic_noisy=noisy_image.copy()

            return noisy_image, noise

        def plot_image_and_fourier(image, noise, title):
            f_transform = fftpack.fft2(image)
            f_transform_shifted = fftpack.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_transform_shifted) + 1)
            self.noisy_fourier=magnitude_spectrum

            f_transforms = fftpack.fft2(noise)
            f_transform_shifteds = fftpack.fftshift(f_transforms)
            magnitude_spectrums = np.log(np.abs(f_transform_shifteds) + 1)    
            self.noise_fourier= magnitude_spectrums.copy()        
            plt.figure(figsize=(18, 6))

            plt.subplot(141), plt.imshow(image, cmap='gray'), plt.title('Original Image')
            plt.subplot(142), plt.imshow(noise, cmap='gray'), plt.title('Periodic Noise')
            plt.subplot(143), plt.imshow(self.noisy_fourier, cmap='gray'), plt.title('Fourier Transform of noisy picture')
            plt.subplot(144), plt.imshow(magnitude_spectrums, cmap='gray'), plt.title('Fourier Transform of noise')
            plt.imsave('noise.png', image, cmap='gray')
            plt.show()

        original_image = cv2.imread(self.file_path, cv2.IMREAD_GRAYSCALE)
        frequency=(frequency,frequency)

        noisy_image, noise = add_periodic_noise(original_image, frequency, amplitude)

        plot_image_and_fourier(noisy_image, noise, 'Image with Periodic Noise')


    def mask_remove_periodic_noise(self):
        def on_ft_click(event):
            if event.inaxes:
              
                x, y = int(event.xdata), int(event.ydata)
                self.mask_coordinates.append((x, y))
                if len(self.mask_coordinates) >= 3:
                    
                    plt.close()
                    self.remove_periodic_noise_with_mask()
                    mask_window.destroy()

        if True:
            mask_coordinates = []
            mask_window = tk.Toplevel(self.root)
            mask_window.title("Mask Method: Select 2 points on Fourier Transform")
            message_label = tk.Label(mask_window, text="Click on 2 points to create a mask.")
            message_label.pack(pady=10)
            self.mask_coordinates = mask_coordinates
            plt.imshow(self.noisy_fourier, cmap='gray')
            plt.title("Fourier Transform of Noisy Image")
            plt.axis("off")
            plt.gcf().canvas.mpl_connect('button_press_event', on_ft_click)
            plt.show()
    
    
    def remove_periodic_noise_with_mask(self):
            x1, y1 = self.mask_coordinates[1]
            x2, y2 = self.mask_coordinates[2]

            noisy_img = np.array(self.image)

            mask = np.ones_like(noisy_img)

            mask[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)] = 1

            # Apply the mask to remove periodic noise
            f_transform = np.fft.fft2(noisy_img)
            f_transform_shifted = np.fft.fftshift(f_transform)
            filtered_spectrum = f_transform_shifted * mask
            inverse_filtered_img = np.fft.ifft2(np.fft.ifftshift(filtered_spectrum)).real

            cleaned_img = Image.fromarray(inverse_filtered_img.astype(np.uint8))
            
            
            plt.imshow(cleaned_img, cmap='gray')
            plt.title("Cleaned image")
            plt.axis("off")
            plt.show()

    def apply_notch_filter(self):
           
        noisy_img = np.array(self.periodic_noisy.copy())
        height, width = noisy_img.shape

        f_transform = np.fft.fft2(noisy_img)
        f_transform_shifted = np.fft.fftshift(f_transform)

        magnitude_spectrum = np.abs(f_transform_shifted)

        peak_y, peak_x = np.unravel_index(np.argmax(magnitude_spectrum), magnitude_spectrum.shape)

        notch_width = int(max(height, width) * 0.02)

        notch_filter = np.ones_like(noisy_img)
        notch_filter[peak_y - notch_width:peak_y + notch_width + 1,
                    peak_x - notch_width:peak_x + notch_width + 1] = 0

        filtered_spectrum = f_transform_shifted * notch_filter
        inverse_filtered_img = np.fft.ifft2(np.fft.ifftshift(filtered_spectrum)).real

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(noisy_img, cmap="gray")
        plt.title("Original Image")
        plt.axis("off")  

        plt.subplot(1, 2, 2)
        plt.imshow(inverse_filtered_img, cmap="gray")
        plt.title("Filtered Image")
        plt.axis("off")  

        plt.show()

        
       
        
    

if __name__ == "__main__":
    root = tk.Tk()
    app = MyApp(root)
    root.mainloop()
