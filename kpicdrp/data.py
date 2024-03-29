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

    Attributes:
        data (np.array): the actual data
        header: corresponding header
        filename (str): filepath that corresponds to the data (where it is read/written)
        type (str): kind of data in string representation
        time_obs (astropy.time.Time): time that data was taken
        fiber_goal (str): short str label of the fiber goal. 'none' if not defined. 
                          Uses same convention as the `labels` field in TraceParams and Spectrum
        filesuffix (str): not implemented
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

        # record fiber goal
        if "FIUGNM" in self.header:
            goal_str = self.header["FIUGNM"]
        else:
            goal_str = "Unknown"
        # convert to shorthand str that we use for labeling
        if 'science fiber' in goal_str:
            fibnum = goal_str[-1]
            self.fiber_goal = 's{0}'.format(fibnum)
        else:
            # not one of the science fibers
            self.fiber_goal = 'none'


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

    def add_parent_filenames(self, parent_data):
        if isinstance(parent_data, Dataset):
            # multiple files
            self.header['DRPNFILE'] = len(parent_data)
            for i in range(len(parent_data)):
                self.header['FILE_{0}'.format(i)] = parent_data[i].filename
        elif isinstance(parent_data, BasicData):
            # single frame
            self.header['DRPNFILE'] = 1
            self.header['FILE_{0}'.format(0)] = parent_data.filename


class Dataset():
    """
    A sequence of data objects of the same kind. Can be looped over. 
    User needs to either initalize with the `frames` variable or pass in both `filelist` and `dtype`

    Args:
        frames (list/array of Data): a list or array of any Data class (i.e., BasicData or subclass)
                                     If this is passed in, `filelist` and `dtype` are ignored.
        filelist (list of str): list of filepaths to load in data from disk
        dtype (kpicdrp.data class): BasicData or any subclass

    Attributes:
        frames (np.array of Data): list/array of Data objects
        type (str): string representation of data type
        data (np.array): datacube of all the data from all the frames
        fib_indices (dict): dictionary that maps a fiber label (e.g., 's2') to an array
                            of indices corresponding to frames with that fiber as the goal
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

        # create this only when requested. 
        # a dictionary of indices for the frames corresponding to each 
        # science fiber
        self._fib_indices = None

    # create the data field dynamically 
    @property
    def data(self):
        this_data = np.array([frame.data for frame in self.frames])
        return this_data

    # create this field only when needed
    @property
    def fib_indices(self):
        if self._fib_indices is None:
            # populate this dictionary on the fly
            self._fib_indices = {}
            all_fib_goals = np.array(self.get_dataset_attributes("fiber_goal"))
            uniq_goals = np.unique(all_fib_goals)
            for goal_label in uniq_goals:
                this_indices = np.where(all_fib_goals == goal_label)[0]
                self._fib_indices[goal_label] = this_indices
        return self._fib_indices

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

    def get_header_values(self, key):
        """
        Grabs the header value for a given keyword from all frames of a dataset

        Args:
            key (str): header keyword

        Returns:
            (list): list of values of that keyword for each element of the dataset
        """
        vals = [frame.header[key] for frame in self]
        
        return vals

    def get_dataset_attributes(self, field):
        """
        Grabs a particular attribute from each attribute in this dataset as a list
        
        Args:
            field (str): attribute name (e.g., "filename" to get all frame.filename)

        Returns:
            (list): list of values of that attribute for each element in the dataset
        """
        vals = [frame.__dict__[field] for frame in self]

        return vals

        
class DetectorFrame(BasicData):
    """
    A frame of 2D data from the NIRSPEC detector. Has shape 2048x2048
    """
    type = "2d"

    def __init__(self, data=None, header=None, filepath="", noise=None, ):
        super().__init__(data, header, filepath)
        
        if 'ROTATED' not in self.header:
            # rotate data by -90 degrees to for NIRSPEC "standard"
            self.data = np.rot90(self.data, -1)
            self.header['ROTATED'] = True # mark in the header frame has been rotated

        if noise is not None:
            self.noise = noise
        elif self.extdata is not None:
            # check if it got read in as an extension header through the filepath. 
            self.noise = self.extdata[0]
        else: 
            self.noise = np.zeros(self.data.shape)

    def save(self, filename=None, filedir=None):
        """
        Save file to disk with user specified filepath

        Args:
            filename (str): filepath to save to. Use self.filename if not specified
            caldb (DetectorCalDB object): if specified, calibration database to keep track of files
        """
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

class Background(DetectorFrame):
    """
    A thermal background frame from the NIRSPEC detector. Has shape 2048x2048
    """
    type = "bkgd"

    def __init__(self, data=None, header=None, filepath="", noise=None):
        super().__init__(data, header, filepath, noise)

    def save(self, filename=None, filedir=None, caldb=None):
        """
        Save file to disk with user specified filepath

        Args:
            filename (str): filepath to save to. Use self.filename if not specified
            caldb (DetectorCalDB object): if specified, calibration database to keep track of files
        """
        self.header['ISCALIB'] = True
        self.header['CALIBTYP'] = "Background"

        super().save(filename=filename, filedir=filedir)

        if caldb is not None:
            caldb.create_entry(self)
            caldb.save()

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
            caldb.create_entry(self)
            caldb.save()


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

    Args:
        locs (np.array): location of traces on detector (N_fibers, N_orders, N_x)
        widths(np.array): standard deviation of Gaussian profile of traces (N_fibers, N_orders, N_x)
        labels (list of str): labels for each fiber. See `labels` attribute below for details of how labels are defined
        trace_index (dict): maps label (e.g., 's1') to index in the N_traces (e.g., 0)
        header: FITS header for file
        filepath (str): filepath to read the calibration file from, if applicable

    Attributes: 
        locs (np.array): location of traces on detector (N_fibers, N_orders, N_x)
        widths (np.array): standard deviation of Gaussian profile of traces (N_fibers, N_orders, N_x)
        labels (list of str): labels for each fiber. Each label consists of a character and a number (e.g., 's3')
                                Characters represent the type of fiber, and numbers denote which fiber of that type.
                                A typical label list looks like ['s1', 's2', 's3', 'b1', 'b2', 'b3', 'd1', 'd2']
                                Types of fibers currently defined:
                                    * `s`: science fibers, where we actually get data from the sky
                                    * `b`: background fibers, where we sample the thermal background of the slit
                                    * `d`: dark current fibers, where we sample the counts outside the slit
                                    * `c`: calibration fibers, containing light from calibration sources
    """
    type = "trace"

    def __init__(self, locs=None, widths=None, labels=None, header=None, filepath=""):
        if locs is None and widths is None and labels is None and header is None:
            super().__init__(filepath=filepath) # read in file from disk
            self.widths = self.data
            self.locs = self.extdata[0]
            
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

        # trace indices mapping
        self.trace_index = {}
        for i, label in enumerate(self.labels):
            self.trace_index[label] = i


    def get_sci_indices(self):
        """
        Returns the indices corresponding to just the scinece fibers

        Return:
            list of int: indices correspond to the fibers that are science fibers. 
        """
        return [i for i,val in enumerate(self.labels) if 's' in val]

    def save(self, filename=None, filedir=None, caldb=None): 
        """
        Save file to disk with user specified filepath

        Args:
            filename (str): filepath to save to. Use self.filename if not specified
            caldb (TraceCalDB object): if specified, calibration database to keep track of traces
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
        hdu = fits.PrimaryHDU(data=self.widths, header=self.header)
        hdulist.append(hdu)
        exthdu1 = fits.ImageHDU(data=self.locs)
        hdulist.append(exthdu1)
        hdulist.writeto(filepath, overwrite=True)
        hdulist.close()

        if caldb is not None:
            caldb.create_entry(self)
            caldb.save()

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

class Spectrum(BasicData):
    """
    Extracted spectral data. Supports data from multiple traces and orders.

    Args:
        fluxes (np.array): extracted fluxes. Dimensions are (N_traces x N_orders x N_columns)
        errs (np.array): uncertainties on fluxes. (N_traces x N_orders x N_columns)
        labels (list of str): labels (e.g, 's1', 'b3') for each trace. Same convention as TraceParams
        trace_index (dict): maps label (e.g., 's1') to index in the N_traces (e.g., 0)
        wavecal (Wavecal): wavelength calibration if desired (optional)
        header: FITS header (only needed if creating a new object from scratch)
        filepath (str): path to file

    Attributes:
        fluxes: extracted fluxes. Dimensions are (N_traces x N_orders x N_columns)
        errs: uncertainties on fluxes. (N_traces x N_orders x N_columns)
        wvs: corresponding wavelengths if calibrated. (N_traces x N_orders x N_columns)
    """
    type = "spectrum"

    def __init__(self, fluxes=None, errs=None, labels=None, wavecal=None, header=None, filepath=""):
        super().__init__(fluxes, header, filepath)
        self.fluxes = self.data

        if errs is not None:
            self.errs = errs
        else:
            self.errs = self.extdata[0]

        # grab labels for each fiber
        if labels is not None:
            if len(labels) != self.fluxes.shape[0]:
                raise ValueError("There are {0} traces but {1} labels are passed in".format(self.fluxes.shape[0], len(labels)))
            self.labels = labels
        else:
            num_fibs = self.fluxes.shape[0]
            self.labels = []
            for i in range(num_fibs):
                self.labels.append(self.header['FIB{0}'.format(i)])
        # fiber indices
        self.trace_index = {}
        for i, label in enumerate(self.labels):
            self.trace_index[label] = i

        if wavecal is not None:
            # can wavelength calibrate once labels are set
            self.calibrate_wvs(wavecal)
        elif (self.extdata is not None) and (len(self.extdata) > 1):
            # read in a wavelength calibrated file
            self._wvs = self.extdata[1]
        else:
            self._wvs = None
            self.header['WAVCALIB'] = False

    # create the data field dynamically 
    @property
    def wvs(self):
        if self._wvs is None:
            raise ValueError("This spectrum is not wavelength calibrated.")
        else:
            return self._wvs

    def calibrate_wvs(self, wv_soln):
        """
        Uses the provided wavelength solution to assign wavelengths to each spectral channel

        Args:
            wv_soln (data.Wavecal): wavecal. Needs to have wvsolns for the fibers in self.labels
        """
        # find the indexes of the labels we want from the wavelength solution
        indices = []
        for label in self.labels:
            if label[0] == 's':
                # only valid for science fibers right now
                index = wv_soln.labels.index(label)
                indices.append(index)
            else:
                # background and slit background fibers don't matter so just grab the first one
                indices.append(0)
        self._wvs = wv_soln.wvs[indices]
        self.header['WAVCALIB'] = True
        self.header['WAVEFILE'] = wv_soln.filename

    def save(self, filename=None, filedir=None): 
        """
        Save file to disk with user specified filepath

        Args:
            filename (str): filepath to save to. Use self.filename if not specified
            caldb (TraceCalDB object): if specified, calibration database to keep track of traces
        """
        self.header['DATATYPE'] = "Spectrum"

        if filename is not None:
            self.filename = filename
        if filedir is not None:
            self.filedir = filedir
        
        filepath = os.path.join(self.filedir, self.filename)
        
        # write labels to header
        for i, label in enumerate(self.labels):
            self.header['FIB{0}'.format(i)] = label

        hdulist = fits.HDUList()
        hdu = fits.PrimaryHDU(data=self.fluxes, header=self.header)
        hdulist.append(hdu)
        exthdu1 = fits.ImageHDU(data=self.errs)
        hdulist.append(exthdu1)
        if self._wvs is not None:
            exthdu2 = fits.ImageHDU(data=self.wvs)
            hdulist.append(exthdu2)
        hdulist.writeto(filepath, overwrite=True)
        hdulist.close()

class Wavecal(BasicData):
    """
    Wavelength calibration file

    Attributes:
        wvs: corresponding wavelengths if calibrated. (N_traces x N_orders x N_columns)
        method (str): method for wavecal. Can be "Star", "Telluric", "Arc"
    """
    type = "wavecal"

    def __init__(self, wvs=None, header=None, labels=None, method=None, filepath=""):
        super().__init__(wvs, header, filepath)
        self.wvs = self.data

        # grab labels for each fiber
        if labels is not None:
            if len(labels) != self.wvs.shape[0]:
                raise ValueError("There are {0} traces but {1} labels are passed in".format(self.wvs.shape[0], len(labels)))
            self.labels = labels
        else:
            num_fibs = self.wvs.shape[0]
            self.labels = []
            for i in range(num_fibs):
                try:
                    self.labels.append(self.header['FIB{0}'.format(i)])
                except KeyError:
                    print("Warning: Keyword FIB{0} not found in header. Guessing this is s{1}".format(i,i+1))
                    self.labels.append("s{0}".format(i+1))

        # fiber indices
        self.trace_index = {}
        for i, label in enumerate(self.labels):
            self.trace_index[label] = i

        # check what type of wavecal this is
        if method is not None:
            self.method = method
        else:
            try:
                self.method = self.header['WVMETHOD']
            except KeyError:
                print("Warning: wavecal {0} does not have method documented".format(filepath))
                self.method = "UNKNOWN"

        

    def save(self, filename=None, filedir=None, caldb=None):
        """
        Save file to disk with user specified filepath

        Args:
            filename (str): filename to use. Use self.filename if not specified
            filedir (str): directory to save the filename to. Use self.filepath if not specified
        """
        self.header['ISCALIB'] = True
        self.header['CALIBTYP'] = "Wavecal"

        for i, label in enumerate(self.labels):
            self.header['FIB{0}'.format(i)] = label

        self.header['WVMETHOD'] = self.method

        super().save(filename=filename, filedir=filedir)

        if caldb is not None:
            caldb.create_entry(self)
            caldb.save()