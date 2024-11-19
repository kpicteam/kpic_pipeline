import pandas as pd
import os
import configparser
import pathlib
from astropy.time import Time
import numpy as np
from kpicdrp.data import BadPixelMap, Background, TraceParams, DetectorFrame, Wavecal
from fnmatch import fnmatch


class CalDB():
    """
    Most basic database type from which everything else inherits. Defines common functionality across all databases.

    Database can be created by passing in column names or passing in a filename to load the database from disk.
    Structure can be used to create new database types - replace column names

    Args:
        col_names (str): names of the columns in the database separated by commas; "Col1,Col2,Col3" 
        filepath (str): filepath to a CSV file with an existing database
        
    Fields:
        db (pd.dataframe): the database that holds all data
        columns (list): column names of dataframe
        filename (str): file name that corresponds to the data (where it is read/written)
        filedir(str): directory where the data is located
        filepath(str): full filepath to data
    """
    def __init__(self, col_names=None, filepath=""):
        # If no filepath is specified, create an empty dataframe with column names
        if col_names is not None:
            self.columns = col_names.split(',')
            self.db = pd.DataFrame(columns = self.columns)
        # If filepath is specified, read in file
        elif len(filepath) > 0: 
            self.filepath = filepath
            self.db = pd.read_csv(filepath) 
            self.columns = list(self.db.columns.values)
        else:
            raise ValueError("Filepath and column_names cannot both be blank.")

        # split up filepath into filename and filedir
        filepath_args = filepath.split(os.path.sep)
        if len(filepath_args) == 1:
            # no directory info in filepath, so current working directory
            self.filedir = "."
            self.filename = filepath_args[0]
        else:
            self.filename = filepath_args[-1]
            self.filedir = os.path.sep.join(filepath_args[:-1])


    def create_entry(self, entry):
        """
        Add a new entry to or update an existing one in the database. Each entry has 3 values: filepath, type, time of observation
        Structure can be used to create new database types - replace column data

        Args:
            entry(BasicData obj): entry to add or update
        """
        if os.path.abspath(entry.filepath) in self.db.values:
            row_index = self.db[self.db["Filepath"]== os.path.abspath(entry.filepath)].index.values
            self.db.loc[row_index,self.columns] = [os.path.abspath(entry.filepath), entry.type, entry.time_obs.isot]
        else:
            new_entry = pd.DataFrame([[os.path.abspath(entry.filepath), entry.type, entry.time_obs.isot]], columns = self.columns)
            self.db = pd.concat([self.db, new_entry], ignore_index = True)


    def remove_entry(self, entry):
        """
        Remove an entry from the database. Removes all values associated with inputted filepath
        Function can be used for any type of database

        Args:
            entry(BasicData obj): entry to remove
        """
        if os.path.abspath(entry.filepath) in self.db.values:
            entry_index = self.db[self.db["Filepath"]==os.path.abspath(entry.filepath)].index.values
            self.db = self.db.drop(self.db.index[entry_index])
            self.db = self.db.reset_index(drop=True)
        else:
            raise ValueError("No filepath found so could not remove.")


    def save(self, filename=None, filedir=None):
        """
        Save file without numbered index to disk with user specified filepath as a CSV file 
        Function can be used for any type of database

        Args:
            filename (str): filepath to save to. Use self.filename if not specified
            filedir (str): filedir to save to. Use self.filedir if not specified
        """
        if filename is not None:
            self.filename = filename
        if filedir is not None:
            self.filedir = filedir
        
        filepath = os.path.join(self.filedir, self.filename)
        self.db.to_csv(filepath, index=False)
    

class DetectorCalDB(CalDB):
    """
    A subclass of CalDB specialized for Background and BadPixelMap frames

    Args:
        filepath (str): filepath to a CSV file with an existing DetectorCalDB database
    """
    def __init__(self, filepath=""):
        if len(filepath)==0:
            self.columns = ["Filepath", "Type","Date/Time of Obs.", "Integration Time", "Coadds", "# of Files Used", "Echelle Position", "X-Disperser Position"]
            self.db = pd.DataFrame(columns = self.columns)
        elif len(filepath) > 0: 
            self.filepath = filepath
            self.db = pd.read_csv(filepath) 
            self.columns = list(self.db.columns.values)

        if self.columns !=["Filepath", "Type","Date/Time of Obs.", "Integration Time", "Coadds", "# of Files Used", "Echelle Position", "X-Disperser Position"]:
            raise ValueError("This is not a DetectorCalDB. Please use a different type of database.")
        
        filepath_args = filepath.split(os.path.sep)
        if len(filepath_args) == 1:
            # no directory info in filepath, so current working directory
            self.filedir = "."
            self.filename = filepath_args[0]
        else:
            self.filename = filepath_args[-1]
            self.filedir = os.path.sep.join(filepath_args[:-1])

    def create_entry(self, entry):
        """
        Add or update an entry in DetectorCalDB
        Each entry has 6 values: Filepath, Type, Date/Time of Obs., Integration Time, Coadds, # of Files Used
        
        Args:
            entry (Background or BadPixelMap obj): entry to be added or updated in database
        """
        if not isinstance(entry, (Background,BadPixelMap)):
            raise ValueError("Entry needs to be instance of Background or Bad Pixel Map")
    
        if os.path.abspath(entry.filepath) in self.db.values:
            row_index= self.db[self.db["Filepath"]==os.path.abspath(entry.filepath)].index.values
            self.db.loc[row_index,self.columns] = [os.path.abspath(entry.filepath), entry.type, entry.time_obs.isot ,entry.header["TRUITIME"],entry.header["COADDS"],entry.header["DRPNFILE"],entry.header["ECHLPOS"],entry.header["DISPPOS"]]
        else:
            new_entry = pd.DataFrame([[os.path.abspath(entry.filepath), entry.type, entry.time_obs.isot ,entry.header["TRUITIME"],entry.header["COADDS"],entry.header["DRPNFILE"],entry.header["ECHLPOS"],entry.header["DISPPOS"]]], columns = self.columns)
            self.db = pd.concat([self.db, new_entry], ignore_index = True)
            
    def get_calib(self, file, type=""):
        """
        Outputs the best background or bad pixel map calibration file (same Integration Time and Coadds, >1 # of Files Used, and then searches for the most similar time) to use when a raw file is inputted

        Args:
            file (DetectorFrame object): raw data file to get calibration for
            type (str): "Background" or "BadPixelMap"
        
        Fields:
            calibdf (pd dataframe): database that holds all badpixmap or all background frames
            options (pd dataframe): database that holds all files that could be used for calibration (same Integration Time, Coadds and >1 # of Files Used)
        """
        self.type = type

        if self.type == "BadPixelMap":
            self.calibdf = self.db[self.db["Type"]=="badpixmap"]
            self.options = self.calibdf.loc[((self.calibdf["Integration Time"] == file.header["TRUITIME"]) & (self.calibdf["Coadds"] == file.header["Coadds"]) & (self.calibdf["# of Files Used"] > 1))]
            options = self.options.copy()
            options["Date/Time of Obs."]=pd.to_datetime(options["Date/Time of Obs."])
            MJD_time = Time(options["Date/Time of Obs."]).mjd
                 
            file_time = Time(file.time_obs).mjd

            result_index = np.abs(MJD_time-file_time).argmin() 
            calib_filepath = options.iloc[result_index,0]

            return BadPixelMap(filepath=calib_filepath)

        elif self.type == "Background":
            self.calibdf = self.db[self.db["Type"]=="bkgd"]
            self.options = self.calibdf.loc[((self.calibdf["Integration Time"] == file.header["TRUITIME"]) & (self.calibdf["Coadds"] == file.header["Coadds"]) & (self.calibdf["# of Files Used"] > 1) & (self.db["Echelle Position"] == file.header["ECHLPOS"]) & (self.db["X-Disperser Position"] == file.header["DISPPOS"]))]
            options = self.options.copy()
            options["Date/Time of Obs."]=pd.to_datetime(options["Date/Time of Obs."])
            MJD_time = Time(options["Date/Time of Obs."]).mjd
                 
            file_time = Time(file.time_obs).mjd

            result_index = np.abs(MJD_time-file_time).argmin() 
            calib_filepath = options.iloc[result_index,0]

            return Background(filepath=calib_filepath)

        else:
            raise ValueError("Specify type of calibration--Background or BadPixelMap")
    
    def readd_calib(self, root_dir):
        """
        Readds calibrated backgrounds and bad pixel maps to caldb from a root directory and its subdirectories
        Looks at files ending in '_coadds1.fits'
        
        Args:
            root_dir(str): filepath to main directory with calibrated files
        
        Fields:
            iscalib_list: list of files that end in '_coadds1.fits' but don't have ISCALIB keyword in header so were not added to det_caldb
        """
        list = []
        self.iscalib_list = []
        pattern = "*_coadds1.fits"

        for path, subdirs, files in os.walk(root_dir):
            for name in files:
                if fnmatch(name,pattern):
                    list.append(str(os.path.join(path,name)))

        for file in list:
            detframe = DetectorFrame(filepath=file)
            try:
                if detframe.header["ISCALIB"] == True:
                    if  detframe.header["CALIBTYP"] == "Background":
                        bkgd = Background(filepath = file)
                        det_caldb.create_entry(bkgd)
                        det_caldb.save()
                    elif detframe.header["CALIBTYP"] == "BadPixelMap":
                        bpm = BadPixelMap(filepath = file)
                        det_caldb.create_entry(bpm)
                        det_caldb.save()
            except KeyError:
                self.iscalib_list.append(str(os.path.abspath(file)))
                pass


class TraceCalDB(CalDB):
    """
    A subclass of CalDB specialized for trace parameter data

    Args:
        filepath (str): filepath to a CSV file with an existing TraceCalDB database
    """
    def __init__(self, filepath=""):
        if len(filepath)==0:
            self.columns = ["Filepath","Date/Time of Obs.", "s1", "s2","s3","s4","c0", "c1", "Echelle Position", "X-Disperser Position"]
            self.db = pd.DataFrame(columns = self.columns)
        elif len(filepath) > 0: 
            self.filepath = filepath
            self.db = pd.read_csv(filepath) 
            self.columns = list(self.db.columns.values)

        if self.columns !=["Filepath","Date/Time of Obs.", "s1", "s2","s3","s4","c0", "c1", "Echelle Position", "X-Disperser Position"]:
            raise ValueError("This is not a TraceCalDB. Please use a different type of database.")
        
        filepath_args = filepath.split(os.path.sep)
        if len(filepath_args) == 1:
            # no directory info in filepath, so current working directory
            self.filedir = "."
            self.filename = filepath_args[0]
        else:
            self.filename = filepath_args[-1]
            self.filedir = os.path.sep.join(filepath_args[:-1])

    def create_entry(self, entry):
        """    
        Add or update an entry in TraceCalDB
        Each entry has 10 values: Filepath, Date/Time of Obs., s1, s2, s3, s4, c0, c1, Echelle Position, X-Disperser Position
        If s1, s2, s3, s4, c0, or c1 is True, then the 'True' fibers were used for the frame
            
        Args:
            entry (TraceParams obj): entry to be added or updated in database
        """
        if not isinstance(entry, (TraceParams)):
            raise ValueError("Entry needs to be instance of TraceParams")
        
        if "s1" in entry.labels:
            s1_val = True
        else:
            s1_val = False
        
        if "s2" in entry.labels:
            s2_val = True
        else:
            s2_val = False

        if "s3" in entry.labels:
            s3_val = True
        else:
            s3_val = False
        
        if "s4" in entry.labels:
            s4_val = True
        else:
            s4_val = False
        
        if "c0" in entry.labels:
            c0_val = True
        else:
            c0_val = False

        if "c1" in entry.labels:
            c1_val = True
        else:
            c1_val = False

        if os.path.abspath(entry.filepath) in self.db.values:
            row_index= self.db[self.db["Filepath"]==os.path.abspath(entry.filepath)].index.values
            self.db.loc[row_index,self.columns] = [os.path.abspath(entry.filepath), entry.time_obs.isot, s1_val, s2_val, s3_val, s4_val, c0_val, c1_val, entry.header["ECHLPOS"],entry.header["DISPPOS"]]
        else:
            new_entry = pd.DataFrame([[os.path.abspath(entry.filepath), entry.time_obs.isot, s1_val, s2_val, s3_val, s4_val, c0_val, c1_val, entry.header["ECHLPOS"],entry.header["DISPPOS"]]], columns = self.columns)
            self.db = pd.concat([self.db, new_entry], ignore_index = True)

    def get_calib(self, file):
        """
        Outputs the best calibration trace file (same Echelle Position and X-Disperser Position, then searches for the most similar time) to use when a raw file is inputted

        Args:
            file (DetectorFrame object): raw data file to get calibration for
        """
        self.options = self.db.loc[((self.db["Echelle Position"] == file.header["ECHLPOS"]) & (self.db["X-Disperser Position"] == file.header["DISPPOS"]))]
        options = self.options.copy()
        options["Date/Time of Obs."]=pd.to_datetime(options["Date/Time of Obs."])
        MJD_time = Time(options["Date/Time of Obs."]).mjd
                 
        file_time = Time(file.time_obs).mjd

        result_index = np.abs(MJD_time-file_time).argmin() 
        calib_filepath = options.iloc[result_index,0]

        return TraceParams(filepath=calib_filepath)


class WaveCalDB(CalDB):
    """
    A subclass of CalDB specialized for Wavelength solutions

    Args:
        filepath (str): filepath to a CSV file with an existing WaveCalDB database
    """
    def __init__(self, filepath=""):
        if len(filepath)==0:
            self.columns = ["Filepath", "Method", "Date/Time of Obs.", "s1", "s2", "s3", "s4", "c0", "c1", "Echelle Position", "X-Disperser Position"]
            self.db = pd.DataFrame(columns = self.columns)
        elif len(filepath) > 0: 
            self.filepath = filepath
            self.db = pd.read_csv(filepath) 
            self.columns = list(self.db.columns.values)

        if self.columns != ["Filepath", "Method", "Date/Time of Obs.", "s1", "s2", "s3", "s4", "c0", "c1", "Echelle Position", "X-Disperser Position"]:
            raise ValueError("This is not a DetectorCalDB. Please use a different type of database.")
        
        filepath_args = filepath.split(os.path.sep)
        if len(filepath_args) == 1:
            # no directory info in filepath, so current working directory
            self.filedir = "."
            self.filename = filepath_args[0]
        else:
            self.filename = filepath_args[-1]
            self.filedir = os.path.sep.join(filepath_args[:-1])

    def create_entry(self, entry):
        """
        Add or update an entry in WaveCalDB
        Each entry has 11 values
        
        Args:
            entry (Background or BadPixelMap obj): entry to be added or updated in database
        """
        if not isinstance(entry, (Wavecal)):
            raise ValueError("Entry needs to be instance of Wavecal")

        if "s1" in entry.labels:
            s1_val = True
        else:
            s1_val = False
        
        if "s2" in entry.labels:
            s2_val = True
        else:
            s2_val = False

        if "s3" in entry.labels:
            s3_val = True
        else:
            s3_val = False
        
        if "s4" in entry.labels:
            s4_val = True
        else:
            s4_val = False
        
        if "c0" in entry.labels:
            c0_val = True
        else:
            c0_val = False

        if "c1" in entry.labels:
            c1_val = True
        else:
            c1_val = False

        if os.path.abspath(entry.filepath) in self.db.values:
            row_index= self.db[self.db["Filepath"]==os.path.abspath(entry.filepath)].index.values
            self.db.loc[row_index,self.columns] = [os.path.abspath(entry.filepath), entry.method, entry.time_obs.isot, s1_val, s2_val, s3_val, s4_val, c0_val, c1_val, entry.header["ECHLPOS"],entry.header["DISPPOS"]]
        else:
            new_entry = pd.DataFrame([[os.path.abspath(entry.filepath), entry.method, entry.time_obs.isot, s1_val, s2_val, s3_val, s4_val, c0_val, c1_val, entry.header["ECHLPOS"],entry.header["DISPPOS"]]], columns = self.columns)
            self.db = pd.concat([self.db, new_entry], ignore_index = True)

    def get_calib(self, file):
        """
        Outputs the best Wavecal for a given science file

        Args:
            file (DetectorFrame object): raw data file to get calibration for
        
        Fields:
            calibdf (pd dataframe): database that holds all badpixmap or all background frames
            options (pd dataframe): database that holds all files that could be used for calibration (same Integration Time, Coadds and >1 # of Files Used)
        """

        self.options = self.db.loc[((self.db["Echelle Position"] == file.header["ECHLPOS"]) & (self.db["X-Disperser Position"] == file.header["DISPPOS"]))]
        options = self.options.copy()
        options["Date/Time of Obs."]=pd.to_datetime(options["Date/Time of Obs."])
        MJD_time = Time(options["Date/Time of Obs."]).mjd
                 
        file_time = Time(file.time_obs).mjd

        result_index = np.abs(MJD_time-file_time).argmin() 
        calib_filepath = options.iloc[result_index,0]

        return Wavecal(filepath=calib_filepath)
    


# load ca
def load_caldb_fromdisk():
    """
    Reads in calibration databases from disk. Creates them if they don't exist.
    
    Returns:
        A tuple of CalDB objects in the following order:
        * `det_db`: background and badpixelmap caldb for detector frames
        * `trace_db`: trace params caldb
    """

    # load in default caldbs based on configuration file
    homedir = pathlib.Path.home()
    config_filepath = os.path.join(homedir, ".kpicdrp")
    config = configparser.ConfigParser()
    config.read(config_filepath)
    caldb_path = config.get("PATH", "caldb")

    #### create the CalDBs if they don't exist
    # backgrounds/bpmap caldb
    detdb_filepath = os.path.join(caldb_path, "caldb_detector.csv")
    if not os.path.exists(detdb_filepath):
        det_db = DetectorCalDB()
        det_db.save(filedir=caldb_path, filename="caldb_detector.csv")
    else:
        det_db = DetectorCalDB(filepath=detdb_filepath)

    # trace params caldb
    tracedb_filepath = os.path.join(caldb_path, "caldb_traces.csv")
    if not os.path.exists(tracedb_filepath):
        trace_db = TraceCalDB()
        trace_db.save(filedir=caldb_path, filename="caldb_traces.csv")
    else:
        trace_db = TraceCalDB(filepath=tracedb_filepath)

    # wavecal caldb
    wavecaldb_filepath = os.path.join(caldb_path, "caldb_wavecal.csv")
    if not os.path.exists(wavecaldb_filepath):
        wavecal_db = WaveCalDB()
        wavecal_db.save(filedir=caldb_path, filename="caldb_wavecal.csv")
    else:
        wavecal_db = WaveCalDB(filepath=wavecaldb_filepath)

    return det_db, trace_db, wavecal_db
    
det_caldb, trace_caldb, wave_caldb = load_caldb_fromdisk()
