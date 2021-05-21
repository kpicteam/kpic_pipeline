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
        filename (str): filepath to a FITS file with data/header information

    Fields:
        data (np.array): the actual data
        header: corresponding header
        filename (str): filepath that corresponds to the data (where it is read/written)
        type (str): kind of data in string representation
        filesuffix (str): 
    """
    type = "base"

    def __init__(self, data=None, header=None, filename=""):
        if data is not None:
            self.data = data
            self.header = header
            self.filename = filename
        elif len(filename) > 0:
            # load data/header from disk. assume loaded in primary FITS extention
            with fits.open(filename) as hdulist:
                self.data = hdulist[0].data
                self.header = hdulist[0].header
            self.filename = filename
        else:
            raise ValueError("Either data or filename needs to be specified. Cannot both be empty")
        



    def save(self, filename=None):
        """
        Save file to disk with user specified filepath

        Args:
            filename (str): filepath to save to. Use self.filename if not specified
        """
        if filename is not None:
            self.filename = filename

        hdulist = fits.HDUList()
        hdu = fits.PrimaryHDU(data=self.data, header=self.header)
        hdulist.append(hdu)
        hdulist.writeto(self.filename, overwrite=True)
        hdulist.close()



class Dataset():
    """
    A sequence of data objects of the same kind. 
    """
    def __init__(self, data_sequence=None, filelist=None, dtype=None):
        if data_sequence is None and filelist is None:
            raise ValueError("Either data_sequence or filelist needs to be specified")

        if data_sequence is not None:
            # data is already nicely formatted. No need to do anything
            self.data = data_sequence
            self.type = data_sequence[0].type
        else:
            # read in data from disk
            if dtype is None:
                raise ValueError("Need to specify a dtype when passing in a list of files")
            self.data = []
            for filename in filelist:
                frame = dtype(filename=filename)
                self.data.append(frame)
            
            self.type = self.data[0].type

    def save(filedir, filenames=None):
        """
        Save each file of data in this dataset into the indicated file directory

        """
        # may need to strip off the dir of filenames to save into filedir. 
        if filenames is None:
            filenames = []
            for frame in self.data:
                filename = frame.filename
                filename_args = filename.split(os.path.sep)
                filename = filename_args[-1]
                filenames.append(filename)

        
class DetectorFrame(BasicData):
    """
    A frame of 2D data from the NIRSPEC detector. Has shape 2048x2048
    """
    type = "2d"

    def __init__(self, data=None, header=None, filename=""):
        super().__init__(data, header, filename)


class Background(DetectorFrame):
    """
    A thermal background frame from the NIRSPEC detector. Has shape 2048x2048
    """
    type = "bkgd"

    def __init__(self, data=None, header=None, filename=""):
        super().__init__(data, header, filename)


    def save(self, filename=None):
        """
        Save file to disk with user specified filepath

        Args:
            filename (str): filepath to save to. Use self.filename if not specified
        """
        self.header['ISCALIB'] = True
        self.header['CALIBTYP'] = "Background"

        super().save(filename)

class BadPixelMap(DetectorFrame):
    """
    A badpixelmap frame from the NIRSPEC detector. Has shape 2048x2048
    """
    type = "badpixmap"

    def __init__(self, data=None, header=None, filename=""):
        super().__init__(data, header, filename)


    def save(self, filename=None):
        """
        Save file to disk with user specified filepath

        Args:
            filename (str): filepath to save to. Use self.filename if not specified
        """
        self.header['ISCALIB'] = True
        self.header['CALIBTYP'] = "BadPixelMap"

        super().save(filename)