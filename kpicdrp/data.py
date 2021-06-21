"""
Define datatypes for the KPIC DRP
"""
import os
from multiprocessing import Value
from astropy.utils import data
import numpy as np
import astropy.io.fits as fits
import astropy.time as time


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
        time_obs (astropy.time): time that data was taken
        filesuffix (str): 
    """
    type = "base"

    def __init__(self, data=None, header=None, filepath=""):
        if data is not None:
            self.data = data
            self.header = header
            self.extdata = None
        elif len(filepath) > 0:
            # load data/header from disk. assume loaded in primary FITS extention
            with fits.open(filepath) as hdulist:
                self.data = np.copy(hdulist[0].data)
                self.header = hdulist[0].header.copy()
                # load the extension headers for subclasses to deal with 
                if len(hdulist) > 1:
                    self.extdata = [np.copy(hdu.data) for hdu in hdulist[1:]]
                else:
                    self.extdata = None
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

        # get time
        date = self.header['DATE-OBS']
        utc = self.header['UTC']
        self.time_obs = time.Time("{0}T{1}Z".format(date, utc))

    # create this field dynamically 
    @property
    def filepath(self):
        return os.path.join(self.filedir, self.filename)

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
    def __init__(self, frames=None, filelist=None, dtype=None):
        if frames is None and filelist is None:
            raise ValueError("Either data_sequence or filelist needs to be specified")


        if frames is not None:
            # data is already nicely formatted. No need to do anything
            self.frames = frames
            self.type = frames[0].type
        else:
            if len(filelist) == 0:
                raise ValueError("Empty filelist passed in")
    
            # read in data from disk
            if dtype is None:
                raise ValueError("Need to specify a dtype when passing in a list of files")
            self.frames = []
            for filepath in filelist:
                frame = dtype(filepath=filepath)
                self.frames.append(frame)
            
            self.type = self.frames[0].type

        # turn lists into np.array for indiexing behavior
        if isinstance(self.frames, list):
            self.frames = np.array(self.frames)

    # create the data field dynamically 
    @property
    def data(self):
        this_data = np.array([frame.data for frame in self.frames])
        return this_data

    def __iter__(self):
        return self.frames.__iter__()

    def __getitem__(self, indices):
        if isinstance(indices, int):
            # return a single element of the data
            return self.frames[indices]
        else:
            # return a subset of the dataset
            return Dataset(frames=self.frames[indices])

    def __len__(self):
        return len(self.frames)

    def save(self, filedir=None, filenames=None):
        """
        Save each file of data in this dataset into the indicated file directory

        """
        # may need to strip off the dir of filenames to save into filedir. 
        if filenames is None:
            filenames = []
            for frame in self.frames:
                filename = frame.filename
                filenames.append(frame.filename)

        for filename, frame in zip(filenames, self.frames):
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
        elif self.extdata is not None:
            self.noise = self.extdata[0]
        else: 
            self.noise = np.zeros(self.data.shape)

    def save(self, filename=None, filedir=None, caldb=None):
        """
        Save file to disk with user specified filepath

        Args:
            filename (str): filepath to save to. Use self.filename if not specified
            caldb (DetectorCalDB object): if specified, calibration database to keep track of files
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

        if caldb is not None:
            self.caldb = caldb
            self.caldb.create_entry(self)
            self.caldb.save()


class BadPixelMap(DetectorFrame):
    """
    A badpixelmap frame from the NIRSPEC detector. Has shape 2048x2048

    Bad pixels are set to np.nan. Good pixels are set to 1. 
    """
    type = "badpixmap"

    def __init__(self, data=None, header=None, filepath=""):
        super().__init__(data, header, filepath)


    def save(self, filename=None, filedir=None, caldb=None):
        """
        Save file to disk with user specified filepath

        Args:
            filename (str): filepath to save to. Use self.filename if not specified
            caldb (DetectorCalDB object): if specified, calibration database to keep track of files
        """
        self.header['ISCALIB'] = True
        self.header['CALIBTYP'] = "BadPixelMap"

        super().save(filename=filename, filedir=filedir)

        if caldb is not None:
            self.caldb = caldb
            self.caldb.create_entry(self)
            self.caldb.save()


    def mark_bad(self, frame):
        """
        Mark bad pixels in input frame as np.nan

        Args:
            frame (DetectorFrame): frame to mark bad pixels
        """
        if not isinstance(frame, DetectorFrame):
            raise ValueError("input frame needs to be instance of DetectorFrame")
        
        frame.data *= self.data # either multiplies by 1 or np.nan

class TraceParams(BasicData):
    """
    Location and widths of fiber traces on the NIRSPEC detector
    """
    type = "trace"

    def __init__(self, locs=None, widths=None, labels=None, header=None, filepath=""):
        if locs is None and widths is None and labels is None and header is None:
            super().__init__(filepath=filepath) # read in file from disk
            self.locs = self.data
            self.widths = self.extdata[0]
            
            self.labels = []
            num_fibs = self.locs.shape[0]
            for i in range(num_fibs):
                self.labels.append(self.header['FIB{0}'.format(i)])
        else:
            if locs is None or widths is None or labels is None or header is None:
                raise ValueError("locs, widths, labels, and header all need to be set as not None in to create a TraceParams")
            
            # not reading in from disk. user has passed in all the necessary components to make a new TraceParams
            super().__init__(data=locs, header=header, filepath=filepath) # just pass in locs as the "data"
        
            self.locs = locs
            self.widths = widths
            self.labels = labels


    def save(self, filename=None, filedir=None):
        """
        Save file to disk with user specified filepath

        Args:
            filename (str): filepath to save to. Use self.filename if not specified
        """
        self.header['ISCALIB'] = True
        self.header['CALIBTYP'] = "TraceParams"

        # write labels to header
        for i, label in enumerate(self.labels):
            self.header['FIB{0}'.format(i)] = label

        if filename is not None:
            self.filename = filename
        if filedir is not None:
            self.filedir = filedir
        
        filepath = os.path.join(self.filedir, self.filename)

        hdulist = fits.HDUList()
        hdu = fits.PrimaryHDU(data=self.locs, header=self.header)
        hdulist.append(hdu)
        exthdu1 = fits.ImageHDU(data=self.widths)
        hdulist.append(exthdu1)
        hdulist.writeto(filepath, overwrite=True)
        hdulist.close()

    def copy(self):
        """
        Makes a copy of itself and return the copy. Deep copy of numpy arrays and headers

        Return:
            copy_trace (TraceParams): a copy of this TraceParams file
        """

        copy_trace = TraceParams(locs=np.copy(self.locs), 
                                widths=np.copy(self.widths), 
                                labels=self.labels.copy(), 
                                header=self.header.copy(), 
                                filepath=self.filepath)

        return copy_trace