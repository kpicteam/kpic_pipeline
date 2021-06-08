import pandas as pd
import os

# CalDB - most generic class 
class CalDB():

    def __init__(self, col_names=None, filepath=""):

    # If no filepath is specified, create an empty dataframe with column names
    # input column_names as "Col1,Col2,Col3" 
        if col_names is not None:
            self.columns = col_names.split(',')
            self.db = pd.DataFrame(columns = self.columns)

        # If filepath is specified, read in file and dataframe is saved as self.db
        elif len(filepath) > 0: 
            self.filepath = filepath
            self.db = pd.read_csv(filepath) 
            self.columns = list(self.db.columns.values)

        else:
            print("Filepath and column_names cannot both be blank.") # added error message

    # split up filepath into filename and filedir, copied from data.py BasicData class
        filepath_args = filepath.split(os.path.sep)
        if len(filepath_args) == 1:
            # no directory info in filepath, so current working directory
            self.filedir = "."
            self.filename = filepath_args[0]
        else:
            self.filename = filepath_args[-1]
            self.filedir = os.path.sep.join(filepath_args[:-1])


    # Adds an entry in the dataframe or updates existing
    def create_entry(self, entry):
        # format of entry: BasicData(filepath=""), BasicData object
        # adds 3 values: filepath, type, time of observation
        if os.path.join(entry.filedir, entry.filename) in self.db.values:
            row_index = self.db[self.db["Filepath"]==os.path.join(entry.filedir, entry.filename)].index.values
            self.db.loc[row_index,self.columns] = [os.path.join(entry.filedir, entry.filename), entry.type, entry.time_obs]
            # replaced existing entry

        else:
            self.db = self.db.append(pd.DataFrame([[os.path.join(entry.filedir, entry.filename), entry.type, entry.time_obs]], columns = self.columns), ignore_index = True)
            # added entry to dataframe


    # Removes an entry in the dataframe
    # format of entry: BasicData(filepath=""), BasicData object
    def remove_entry(self, entry):
        if os.path.join(entry.filedir, entry.filename) in self.db.values:
            entry_index = self.db[self.db["Filepath"]==os.path.join(entry.filedir, entry.filename)].index.values
            self.db = self.db.drop(self.db.index[entry_index])
            self.db = self.db.reset_index(drop=True)
        else:
            return("No filepath found so could not remove.")


    # Save dataframe to existing csv file or creates new one based on filepath
    def save(self, filename=None, filedir=None):
        """
        Save file to disk with user specified filepath

        Args:
            filename (str): filepath to save to. Use self.filename if not specified
            filedir (str): filedir to save to. Use self.filedir if not specified
        """
        # copied from data.py BasicData class
        if filename is not None:
            self.filename = filename
        if filedir is not None:
            self.filedir = filedir
        
        filepath = os.path.join(self.filedir, self.filename)

        self.db.to_csv(filepath, index=False) # doesn't save index
    # throw an error if filepath is not specified?


# subclass for Detector database
class DetectorCalDB(CalDB):

    def __init__(self, col_names=None, filepath=""):
        super().__init__(col_names, filepath)
    # self.columns = define, so don't need to type in
    # col_names = "Filepath,Type,Date/Time of Obs.,Integration Time,Coadds"
        if self.columns !=["Filepath", "Type","Date/Time of Obs.", "Integration Time", "Coadds"]:
            # standard for these names, must type in exact spelling each time?
            raise ValueError("This is not a Detector Calibration Database. Please use a different type of database.")

    # Adds an entry in the dataframe or updates existing
    def create_entry(self, entry):
        # format of entry: BadPixelMap(filepath=""), BadPixelMap object
        # adds 5 values: filepath, type, time of observation, integration time, coadds
        if os.path.join(entry.filedir, entry.filename) in self.db.values:
            row_index = self.db[self.db["Filepath"]==os.path.join(entry.filedir, entry.filename)].index.values
            self.db.loc[row_index,self.columns] = [os.path.join(entry.filedir, entry.filename), entry.type, entry.time_obs,entry.header["TRUITIME"],entry.header["COADDS"]]
            # replaced existing entry

        else:
            self.db = self.db.append(pd.DataFrame([[os.path.join(entry.filedir, entry.filename), entry.type, entry.time_obs,entry.header["TRUITIME"],entry.header["COADDS"]]], columns = self.columns), ignore_index = True)
            # added entry to dataframe

# save function and remove_entry function is same as CalDB
# add a function to get best calib file


"""
Other sample test code: 
x.db to view database
"""