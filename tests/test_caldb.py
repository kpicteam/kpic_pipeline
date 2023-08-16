import kpicdrp.caldb as caldb
import os
from kpicdrp.data import Background, BadPixelMap

# Filepaths for a BadPixelMap file and a Background file - must change these
bp1path = "../../public_kpic_data/20200928_backgrounds/calib/20200928_persistent_badpix_nobars_tint4.42584_coadds1.fits"
bkgd1path ="../../public_kpic_data/20200928_backgrounds/calib/20200928_background_med_nobars_tint4.42584_coadds1.fits"

# Creating a BadPixelMap object and a Background Object from above filepaths
bp1 = BadPixelMap(filepath=bp1path)
bkgd1 = Background(filepath=bkgd1path)

# Filepath 1 (to save a Database)
myfiledir ="./" #change path
myfilename = "Database1"
myfilepath = os.path.join(myfiledir, myfilename)

# Filepath 2 (to save a Database)
myfiledir2 = "./" # change path
myfilename2 = "Database2"
myfilepath2 = os.path.join(myfiledir2, myfilename2)

def test_DetectorCalDB_create_entry_add():
    """
    Creates an empty dataframe, adds a BadPixelMap and Background entry while checking that # of rows increase accordingly and # of columns stay the same
    """
    y=caldb.DetectorCalDB()
    assert len(y.db.columns) == 8
    assert len(y.db.index) == 0
    y.create_entry(bp1)
    #check BPM, and check Background
    assert len(y.db.columns) == 8
    assert len(y.db.index) == 1
    y.create_entry(bkgd1)
    assert len(y.db.columns) == 8
    assert len(y.db.index) == 2

def test_DetectorCalDB_create_entry_update():
    """
    Creates an empty dataframe, adds an entry, adds same entry and checks if # of rows and # of cols stay the same
    """
    y=caldb.DetectorCalDB()
    y.create_entry(bp1)
    num_rows_initial = len(y.db.index)
    num_cols_initial = len(y.db.columns)
    y.create_entry(bp1)
    assert len(y.db.index) == num_rows_initial
    assert len(y.db.columns) == num_cols_initial

def test_DetectorCalDB_remove_entry():
    """
    Creates an empty dataframe, adds an entry, removes that same entry and checks if # of rows and # of cols is same as initial empty dataframe
    """
    y=caldb.DetectorCalDB()
    y.create_entry(bkgd1)
    y.remove_entry(bkgd1)
    assert len(y.db.columns) == 8
    assert len(y.db.index) == 0

def test_DetectorCalDB_save_newdb():
    """
    Creates an empty dataframe, saves to specified filepath (Filepath 1), reopens dataframe and checks if dimensions are same
    """
    y=caldb.DetectorCalDB()
    y.save(myfilename, myfiledir)
    z = caldb.DetectorCalDB(filepath=myfilepath)
    assert len(y.db.columns) == len(z.db.columns)
    assert len(y.db.index) == len(z.db.index)

def test_DetectorCalDB_save_newpath():
    """
    Opens existing dataframe (Filepath 1), saves to new filepath (Filepath 2), reopens from new filepath and checks if dimensions are same
    """
    x=caldb.DetectorCalDB(filepath=myfilepath)
    x.save(myfilename2,myfiledir2)
    y=caldb.DetectorCalDB(filepath=myfilepath2)
    assert len(x.db.columns) == len(y.db.columns)
    assert len(x.db.index) == len(y.db.index)

def test_DetectorCalDB_save_samepath():
    """
    Opens existing dataframe (Filepath 1), adds an entry, saves to same filepath, reopens dataframe and checks if # of rows increase by 1 and # of cols is same
    """
    x=caldb.DetectorCalDB(filepath=myfilepath)
    num_rows_initial = len(x.db.index)
    num_cols_initial = len(x.db.columns)
    x.create_entry(bp1)
    x.save()
    assert len(x.db.index) == num_rows_initial + 1
    assert len(x.db.columns) == num_cols_initial

if __name__ == "__main__":
    test_DetectorCalDB_create_entry_add()
    test_DetectorCalDB_create_entry_update()
    test_DetectorCalDB_remove_entry()
    test_DetectorCalDB_save_newdb()
    test_DetectorCalDB_save_newpath()
    test_DetectorCalDB_save_samepath()