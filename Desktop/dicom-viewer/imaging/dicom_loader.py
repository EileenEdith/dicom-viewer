import pydicom
import numpy as np

def load_dicom(path):
    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype(np.uint8)
    return img, ds