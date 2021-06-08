from kpicdrp.data import BasicData
import pandas as pd
import os

# CalDB - most generic class 
class CalDB():

    column_names = ["Filepath", "Type", "Date/Time of Obs."]


    def __init__(self, filepath=""):
    
     # If filepath is specified, read in file and dataframe saved as self.db
        if len(filepath) > 0: 
            self.filepath = filepath
            self.data = pd.read_csv(filepath)
            self.db = pd.DataFrame(self.data, columns = CalDB.column_names)
            # only reads in data columns, not index
        
    # If no filepath is specified, create an empty dataframe with column names
        else:
            self.db = pd.DataFrame(columns = CalDB.column_names)

    # split up filepath into filename and filedir, copied from data.py BasicData class
        filepath_args = filepath.split(os.path.sep)
        if len(filepath_args) == 1:
            # no directory info in filepath, so current working directory
            self.filedir = "."
            self.filename = filepath_args[0]
        else:
            self.filename = filepath_args[-1]
            self.filedir = os.path.sep.join(filepath_args[:-1])


    # Adds an entry to bottom of dataframe
    def add_entry(self, new_entry):
        # format of new_entry: BasicData(filepath="")
        self.db = self.db.append(pd.DataFrame([[os.path.join(new_entry.filedir, new_entry.filename), new_entry.type, new_entry.time_obs]], columns = CalDB.column_names), ignore_index = True)
        

    # Updates an entry in the dataframe if know index, maintains same index as before
    def update_entry(self, row_index, updated_entry):
        # format of updated_entry: BasicData(filepath="")
        self.db.loc[row_index,CalDB.column_names] = [os.path.join(updated_entry.filedir, updated_entry.filename), updated_entry.type, updated_entry.time_obs]


    # Removes an entry in the dataframe if know index, resets index after dropping
    def remove_entry(self, row_index):
        # format of row_index: int
        self.db = self.db.drop([self.db.index[row_index]])
        self.db = self.db.reset_index(drop=True)


    # Save dataframeto existing csv file or creates new one based on filepath
    # saves with index in csv but code opens csv without index
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
        
        self.filepath = os.path.join(self.filedir, self.filename)

        self.db = self.db.reset_index(drop=True) # reset index just in case
        self.db.to_csv(self.filepath) 


"""
Other sample test code: 
x.db.loc[index] to access a specific row if know the index as int
x.db to view database
"""