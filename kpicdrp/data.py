"""
Define datatypes for the KPIC DRP
"""
import os
from multiprocessing import Value
import numpy as np
import astropy.io.fits as fits


class BasicData():
    """
    Base data type from which everything else inherits. Defines common functionality across all data.

    Data can be created by passing in the data/header explicitly, or by passing in a filename to load
    the data/header from disk. 

    Args:
        data (np.array): the actual data
        header: corresponding header
        filepath (str): filepath to a FITS file with data/header information  

    Fields:
        data (np.array): the actual data
        header: corresponding header
        filename (str): filepath that corresponds to the data (where it is read/written)
        type (str): kind of data in string representation
        filesuffix (str): 
    """
    type = "base"

    def __init__(self, data=None, header=None, filepath=""):
        if data is not None:
            self.data = data
            self.header = header
            self.exthdus = None
        elif len(filepath) > 0:
            # load data/header from disk. assume loaded in primary FITS extention
            with fits.open(filepath) as hdulist:
                self.data = hdulist[0].data
                self.header = hdulist[0].header
                # load the extension headers for subclasses to deal with 
                if len(hdulist) > 1:
                    self.exthdus = hdulist[1:]
                else:
                    self.exthdus = None
        else:
            raise ValueError("Either data or filename needs to be specified. Cannot both be empty")
        
        # split up filepath into filename and filedir
        filepath_args = filepath.split(os.path.sep)
        if len(filepath_args) == 1:
            # no directory info in filepath, so current working directory
            self.filedir = "."
            self.filename = filepath_args[0]
        else:
            self.filename = filepath_args[-1]
            self.filedir = os.path.sep.join(filepath_args[:-1])


    def save(self, filename=None, filedir=None):
        """
        Save file to disk with user specified filepath

        Args:
            filename (str): filepath to save to. Use self.filename if not specified
            filedir (str): filedir to save to. Use self.filedir if not specified
        """
        if filename is not None:
            self.filename = filename
        if filedir is not None:
            self.filedir = filedir
        
        filepath = os.path.join(self.filedir, self.filename)

        hdulist = fits.HDUList()
        hdu = fits.PrimaryHDU(data=self.data, header=self.header)
        hdulist.append(hdu)
        hdulist.writeto(filepath, overwrite=True)
        hdulist.close()



class Dataset():
    """
    A sequence of data objects of the same kind. Can be looped over. 
    """
    def __init__(self, data_sequence=None, filelist=None, dtype=None):
        if data_sequence is None and filelist is None:
            raise ValueError("Either data_sequence or filelist needs to be specified")


        if data_sequence is not None:
            # data is already nicely formatted. No need to do anything
            self.data = data_sequence
            self.type = data_sequence[0].type
        else:
            if len(filelist) == 0:
                raise ValueError("Empty filelist passed in")
    
            # read in data from disk
            if dtype is None:
                raise ValueError("Need to specify a dtype when passing in a list of files")
            self.data = []
            for filepath in filelist:
                frame = dtype(filepath=filepath)
                self.data.append(frame)
            
            self.type = self.data[0].type

            # turn lists into np.array for indiexing behavior
            if isinstance(self.data, list):
                self.data = np.array(self.data)

    def __iter__(self):
        self.__count__ = 0
        return self

    def __next__(self):
        if self.__count__ < len(self.data):
            frame = self.data[self.__count__]
            self.__count__ += 1
            return frame
        else:
            raise StopIteration

    def __getitem__(self, indices):
        if isinstance(indices, int):
            # return a single element of the data
            return self.data[indices]
        else:
            # return a subset of the dataset
            return Dataset(data_sequence=self.data[indices])

    def __len__(self):
        return len(self.data)

    def save(self, filedir=None, filenames=None):
        """
        Save each file of data in this dataset into the indicated file directory

        """
        # may need to strip off the dir of filenames to save into filedir. 
        if filenames is None:
            filenames = []
            for frame in self.data:
                filename = frame.filename
                filenames.append(frame.filename)

        for filename, frame in zip(filenames, self.data):
            frame.save(filename=filename, filedir=filedir)

        
class DetectorFrame(BasicData):
    """
    A frame of 2D data from the NIRSPEC detector. Has shape 2048x2048
    """
    type = "2d"

    def __init__(self, data=None, header=None, filepath=""):
        super().__init__(data, header, filepath)
        
        if 'ROTATED' not in self.header:
            # rotate data by -90 degrees to for NIRSPEC "standard"
            self.data = np.rot90(self.data, -1)
            self.header['ROTATED'] = True # mark in the header frame has been rotated


class Background(DetectorFrame):
    """
    A thermal background frame from the NIRSPEC detector. Has shape 2048x2048
    """
    type = "bkgd"

    def __init__(self, data=None, header=None, filepath="", data_noise=None):
        super().__init__(data, header, filepath)

        if data_noise is not None:
            self.noise = data_noise
        elif self.exthdus is not None:
            self.noise = self.exthdus[0].data
        else: 
            self.noise = np.zeros(self.data.shape)

    def save(self, filename=None, filedir=None):
        """
        Save file to disk with user specified filepath

        Args:
            filename (str): filepath to save to. Use self.filename if not specified
        """
        self.header['ISCALIB'] = True
        self.header['CALIBTYP'] = "Background"

        if filename is not None:
            self.filename = filename
        if filedir is not None:
            self.filedir = filedir
        
        filepath = os.path.join(self.filedir, self.filename)

        hdulist = fits.HDUList()
        hdu = fits.PrimaryHDU(data=self.data, header=self.header)
        hdulist.append(hdu)
        exthdu1 = fits.ImageHDU(data=self.noise)
        hdulist.append(exthdu1)
        hdulist.writeto(filepath, overwrite=True)
        hdulist.close()

class BadPixelMap(DetectorFrame):
    """
    A badpixelmap frame from the NIRSPEC detector. Has shape 2048x2048

    Bad pixels are set to np.nan. Good pixels are set to 1. 
    """
    type = "badpixmap"

    def __init__(self, data=None, header=None, filepath=""):
        super().__init__(data, header, filepath)


    def save(self, filename=None, filedir=None):
        """
        Save file to disk with user specified filepath

        Args:
            filename (str): filepath to save to. Use self.filename if not specified
        """
        self.header['ISCALIB'] = True
        self.header['CALIBTYP'] = "BadPixelMap"

        super().save(filename=filename, filedir=filedir)

    def mark_bad(self, frame):
        """
        Mark bad pixels in input frame as np.nan

        Args:
            frame (DetectorFrame): frame to mark bad pixels
        """
        if not isinstance(frame, DetectorFrame):
            raise ValueError("input frame needs to be instance of DetectorFrame")
        
        frame.data *= self.data # either multiplies by 1 or np.nan