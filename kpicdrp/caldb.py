import pandas as pd
import os
import kpicdrp.data as data

class CalDB():
    """
    Most basic database type from which everything else inherits. Defines common functionality across all databases.

    Database can be created by passing in column names or passing in a filename to load the database from disk.

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

        Args:
            entry(BasicData obj): entry to add or update
        """
        if entry.filepath in self.db.values:
            row_index = self.db[self.db["Filepath"]== entry.filepath].index.values
            self.db.loc[row_index,self.columns] = [entry.filepath, entry.type, entry.time_obs]
        else:
            self.db = self.db.append(pd.DataFrame([[entry.filepath, entry.type, entry.time_obs]], columns = self.columns), ignore_index = True)


    def remove_entry(self, entry):
        """
        Remove an entry from the database. Removes all values associated with inputted filepath

        Args:
            entry(BasicData obj): entry to remove
        """
        if entry.filepath in self.db.values:
            entry_index = self.db[self.db["Filepath"]==entry.filepath].index.values
            self.db = self.db.drop(self.db.index[entry_index])
            self.db = self.db.reset_index(drop=True)
        else:
            raise ValueError("No filepath found so could not remove.")


    def save(self, filename=None, filedir=None):
        """
        Save file without numbered index to disk with user specified filepath as a CSV file 

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
    


# subclass for Detector database
class DetectorCalDB(CalDB):
    """
    A subclass of CalDB specialized for Background and BadPixelMap frames.

    Args:
        filepath (str): filepath to a CSV file with an existing DetectorCalDB database
    """
    def __init__(self, filepath=""):
        if len(filepath)==0:
            self.columns = ["Filepath", "Type","Date/Time of Obs.", "Integration Time", "Coadds"]
            self.db = pd.DataFrame(columns = self.columns)
        elif len(filepath) > 0: 
            self.filepath = filepath
            self.db = pd.read_csv(filepath) 
            self.columns = list(self.db.columns.values)

        if self.columns !=["Filepath", "Type","Date/Time of Obs.", "Integration Time", "Coadds"]:
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
        Add or update an entry in DetectorCalDB. Each entry has 5 values: Filepath, Type, Date/Time of Obs., Integration Time, Coadds
        
        Args:
            entry (Background or BadPixelMap obj): entry to be added or updated in database
        """
        if not isinstance(entry, (data.Background,data.BadPixelMap)):
            raise ValueError("Entry needs to be instance of Background or Bad Pixel Map")
    
        if entry.filepath in self.db.values:
            row_index= self.db[self.db["Filepath"]==entry.filepath].index.values
            self.db.loc[row_index,self.columns] = [entry.filepath, entry.type, entry.time_obs,entry.header["TRUITIME"],entry.header["COADDS"]]
        else:
            self.db = self.db.append(pd.DataFrame([[entry.filepath, entry.type, entry.time_obs,entry.header["TRUITIME"],entry.header["COADDS"]]], columns = self.columns), ignore_index = True)

    def get_calib(self, file):
        """
        Outputs the best calibration file to use when a raw file is inputted.

        Args:
            file(DetectorFrame object): raw data file to be calibrated, a BasicData object
            choices (pd.dataframe): dataframe that holds selected calibration files that can be used 
        """
        self.choices = self.db[self.db["Integration Time"] == file.header["TRUITIME"]]
        
        if self.choices.empty is True:
            return "No file with same Integration Time."
        else:
            self.choices = self.choices[self.choices["Coadds"] == file.header["Coadds"]]

            if self.choices.empty is True:
                return "No file with same Coadds."