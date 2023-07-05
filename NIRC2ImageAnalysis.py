import numpy as np
from scipy.ndimage import shift, rotate, median_filter, generic_filter, uniform_filter
from pyklip.instruments import NIRC2
from photutils.centroids import centroid_com, centroid_quadratic
from photutils.centroids import centroid_1dg, centroid_2dg
from IPython.display import clear_output
import matplotlib.pyplot as plt
import matplotlib.colors as co
import astropy.io.fits as fits
import warnings

class NIRC2ImageAnalysis:

    def __init__(self, filenames:list, guess_star:list):
        """
        Creates a new NIRC2ImageAnalysis object to easily reduce images.
        Paramaters:
            - filenames: A sequence containing a list of paths
            - guess_star: The best guess of the star on the images in [y,x] format
        """
        self.filenames = filenames
        self.NIRC2_obj = NIRC2.NIRC2Data()
        self.guess_star = guess_star
        
        # Read the data into the NIRC2 object and create primary star guesses
        self.NIRC2_obj.readdata(self.filenames, find_star=True, guess_star=self.guess_star)
        clear_output()
        print("Images have been read")
        
        # Get properties from the NIRC2 object to our class
        self.primary_centers = self.NIRC2_obj.centers
        if self.primary_centers == []:
            warnings.warn("WARNING: Radon transform was unsucessful. No centers found.")

        # Get the parallactic angle and input images from the NIRC2 object
        self.par_angle = self.NIRC2_obj.PAs
        self.sci_img = self.NIRC2_obj.input
        self.sci_frames = None
        self.final_image = []

        
    def get_centers(self):
        '''
        Returns the center of the primary star in [x,y] format
        '''
        return self.primary_centers
    
    def get_final_image(self):
        """
        Return and show final image if reduction if complete. Otherwise
        nothing will be returned 
        """
        if self.final_image == []:
            warnings.warn("WARNING: Reduction has not been initated")
        else:
            print("Plotting final image")
            plt.imshow(self.final_image, origin='lower')
            plt.show()
            return self.final_image
    
    def save_final(self, path):
        """
        Saves the final image to the path as a fits file
        
        Parameters:
            - path: the desired destination of the fits file
        """
        hdu = fits.PrimaryHDU(self.final_image)
        hdul = fits.HDUList([hdu])
        hdul.writeto(path, overwrite=True)
    
    def reduce(self, dark_path, flat_path, dark_exp_scale, dark_flat_snip=None) -> bool:
        """
        Reads in the dark and flat matrices and scales the dark matrix to be the 
        same exposure time as the images. 

        Paramaters:
            - dark_path: path to the dark matrix 
            - flat_path: path to the flat matrix
            - dark_exp_scale: the scaling factor to scale the exposure time for the
                              dark matrix
            - dark_flat_snip: a tuple containing the new shape of the dark and flat frame.
                              The frames will be cut from the center of the array
        Returns:
            - True: if the reduction was sucessful
            - False: if reduction failed
        """
        # Open master_dark and multiply so exposure times match
        hdul = fits.open(dark_path, ignore_missing_end=True)
        dark = hdul[0].data * dark_exp_scale

        # Open master flat 
        hdul = fits.open(flat_path, ignore_missing_end=True)
        flat = hdul[0].data

        # Snips the dark array by the dark_flat_snip dimensions if given
        dim_d = np.shape(dark)
        dim_f = np.shape(flat)

        if dark_flat_snip != None:
            dark = dark[dim_d[1]//2 - dark_flat_snip[1]//2 : dim_d[1]//2 + dark_flat_snip[1]//2, dim_d[0]//2 - dark_flat_snip[0]//2 : dim_d[0]//2 + dark_flat_snip[0]//2]
            flat = flat[dim_f[1]//2 - dark_flat_snip[1]//2 : dim_f[1]//2 + dark_flat_snip[1]//2, dim_f[0]//2 - dark_flat_snip[0]//2 : dim_f[0]//2 + dark_flat_snip[0]//2]

        # Checks if dimensions are correct:
        dim_test = np.shape(self.sci_img[0])
        for img in self.sci_img[1:]:
            if dim_test != np.shape(img):
                print("Science images not all the same dimension")
                return False
        if np.shape(dark) != dim_test:
            print("Dark dimension and science image incompatible")
            return False

        # Subtract the science images by the dark array to get the science frames
        self.sci_frames = self.sci_img - dark

        # Subtract the science frames by the dark data and divide the result by the flat data 
        self.sci_frames = np.divide(self.sci_frames, flat, out=np.zeros_like(self.sci_frames), where=flat!=0)

        # Take the median across the frames and subtract the respected image with the frame
        for im in range(len(self.sci_frames)):
            myim = self.sci_frames[im]
            avg = np.median(myim)
            myim -= avg
            self.sci_frames[im] = myim
            
        # Run the hot pixel remover on all the frames
        for ind in range(len(self.sci_frames)):
            self.sci_frames[ind] =  self.hot_pixel_remover(np.abs(self.sci_frames[ind]), window_size=5, mult_value=5)     

        # Transform the image by shifting and rotating them
        return self.image_transform()  
    
    def centroid(self, guess, snip_rad=10):
        """
        The function finds the centroid of the image by fitting a 2D Gaussian to the 
        distribution in the image and finds the uncertainty of that value in pixels.
        Note: The primary star must be centered in the image. 

        Parameters:
            - path: the path to the imag
            - guess: target guess should be in the form [x, y]
            - snip_rad: refers to the snip radius of the target. The target must
                        fit within this snipped image.
        
        Returns:
            - a tuple (delta RA, delta declination, RA uncertainty, declination uncertainty)
        """
        x_guess = guess[0]
        y_guess = guess[1]
        dat = self.final_image[y_guess - snip_rad : y_guess + snip_rad, x_guess - snip_rad : x_guess + snip_rad]
        x2, y2 = centroid_quadratic(dat)
        x3, y3 = centroid_1dg(dat)
        x4, y4 = centroid_2dg(dat)
        x_final = x4 + x_guess - snip_rad
        y_final = y4 + y_guess - snip_rad
        plt.imshow(dat, origin='lower') 
        plt.show()
        return (self.final_image.shape[0] - x_final , y_final - self.final_image.shape[1], np.std([x2,x3,x4]), np.std([y2,y3,y4]))
                
    
    def hot_pixel_remover(self, image, window_size=5, mult_value=3):
        """
        Removes the hot pixels in an image by creating a median and standard diviation mask.
        The algorithm would then look pixel by pixel and compare it's value to the masks
        and determine if the pixel is hot. The hot pixel's value would be replaced by the 
        median value (from the median mask).

        Parameters:
            - image: a single 2D numpy array
            - window_size: the size of the window which the median and standard deviation 
                           are calculated over.
            - mult_value: corresponds to the n*sigma for the standard diviation gaussian 
        """
        # Create masks
        median_mask = median_filter(image, size=(window_size,window_size))
        stdev_mask = self._stdev_filter(median_mask, window_size)
        
        # Investigate pixels in image
        return np.where((image > (median_mask + (mult_value * stdev_mask))) | (image < (median_mask - (mult_value * stdev_mask))), median_mask ,image)
        
    def image_transform(self) -> bool:
        """
        Shifts and rotates of the images based on the parallactic angle
        """
        # Now we want to shift our images over using scipy.ndimage.shift, which we read in as shift
        nim = len(self.primary_centers) #number of images
        if nim == 0:
            return False

        centerx = self.sci_img[0].shape[0] // 2
        centery = self.sci_img[0].shape[1] // 2
        delta_x = [centerx - self.primary_centers[i][0] for i in range(nim)]
        delta_y = [centery - self.primary_centers[i][1] for i in range(nim)]

        shifted_input = np.empty(self.sci_img.shape)

        for j in range(nim): #loop through each image
            shifted_im = shift(self.sci_frames[j], (delta_y[j], delta_x[j])) #put shifts in row,column order
            rot_im = rotate(shifted_im, -self.par_angle[j], reshape=False)
            shifted_input[j,:,:] = rot_im

        self.final_image = np.median(shifted_input, axis=0)
        return True

    def _stdev_filter(self, image, window_size=5):
        """
        Creates a standard deviation filter on an image over a window.
        https://nickc1.github.io/python,/matlab/2016/05/17/Standard-Deviation-(Filters)-in-Matlab-and-Python.html

        Parameters: 
            - image: a single 2D numpy array of the image
            - window_size: the size of the window to look around a particular pixel
        """
        r,c = image.shape
        image += np.random.rand(r,c)*1e-6
        c1 = uniform_filter(image, window_size, mode='constant')
        c2 = uniform_filter(image*image, window_size, mode='constant')
        return np.sqrt(c2 - c1*c1)