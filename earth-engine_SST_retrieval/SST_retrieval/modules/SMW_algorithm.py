import ee
from SMW_coefficients import *
import sys

ee.Initialize()

# Define the function to create a lookup between two columns in a feature collection
def get_lookup_table(fc, prop_1, prop_2):  # fc = feature collection
    reducer = ee.Reducer.toList().repeat(2)
    lookup = fc.reduceColumns(reducer, [prop_1, prop_2])
    return ee.List(lookup.get('list'))

# Define the add_band function
def add_band(landsat, image):
    
    # Select algorithm coefficients based on Landsat version
    coeff_SMW = ee.FeatureCollection(ee.Algorithms.If(landsat == 'L4', coeff_SMW_L4,
                    ee.Algorithms.If(landsat == 'L5', coeff_SMW_L5,
                    ee.Algorithms.If(landsat == 'L7', coeff_SMW_L7,
                    ee.Algorithms.If(landsat == 'L8', coeff_SMW_L8,
                    coeff_SMW_L9)))))
    
    # Create lookups for the algorithm coefficients
    A_lookup = get_lookup_table(coeff_SMW, 'TPWpos', 'A')
    B_lookup = get_lookup_table(coeff_SMW, 'TPWpos', 'B')
    C_lookup = get_lookup_table(coeff_SMW, 'TPWpos', 'C')
    
    # Map coefficients to the image using the TPW bin position
    A_img = image.remap(A_lookup.get(0), A_lookup.get(1), 0.0, 'TPWpos').resample('bilinear')
    B_img = image.remap(B_lookup.get(0), B_lookup.get(1), 0.0, 'TPWpos').resample('bilinear')
    C_img = image.remap(C_lookup.get(0), C_lookup.get(1), 0.0, 'TPWpos').resample('bilinear')
    
    # Select TIR band based on Landsat version
    tir = ee.String(ee.Algorithms.If(landsat == 'L9', 'B10',
                    ee.Algorithms.If(landsat == 'L8', 'B10',
                    ee.Algorithms.If(landsat == 'L7', 'B6_VCID_1', 'B6'))))
    
    # Get emissivity and brightness temperature from the image
    em1 = image.select('EM')
    Tb1 = image.select(tir)
    
    # Compute the LST
    lst = image.expression(
        'A * Tb1 / em1 + B / em1 + C',
        {
            'A': A_img,
            'B': B_img,
            'C': C_img,
            'em1': em1,
            'Tb1': Tb1
        }
    ).updateMask(image.select('TPW').lt(0).Not())
    
    # Add computed SST and coefficients as bands
    return image.addBands(ee.Image.cat([
        lst.rename('SST'),
        A_img.rename('A_coeff'),
        B_img.rename('B_coeff'),
        C_img.rename('C_coeff'),
        em1.rename('emissivity'),
        Tb1.rename('brightness_temp')
    ]))
