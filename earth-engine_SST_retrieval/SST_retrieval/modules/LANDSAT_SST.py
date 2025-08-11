import ee
ee.Initialize()

import cloud_mask
import NCEP_TPW
import compute_emissivity
import SMW_algorithm


# Define landsat image collections and change band names
COLLECTION = {
    'L4': {
        'TOA': ee.ImageCollection('LANDSAT/LT04/C02/T1_TOA'),
        'SR': ee.ImageCollection('LANDSAT/LT04/C02/T1_L2'),
        'TIR': ['B6', ],
        'VISW': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'QA_PIXEL'],
        'VISW_TOA': ['B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'QA_PIXEL']
    },
    'L5': {
        'TOA': ee.ImageCollection('LANDSAT/LT05/C02/T1_TOA'),
        'SR': ee.ImageCollection('LANDSAT/LT05/C02/T1_L2'),
        'TIR': ['B6', ],
        'VISW': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'QA_PIXEL'],
        'VISW_TOA': ['B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'QA_PIXEL']
    },
    'L7': {
        'TOA': ee.ImageCollection('LANDSAT/LE07/C02/T1_TOA'),
        'SR': ee.ImageCollection('LANDSAT/LE07/C02/T1_L2'),
        'TIR': ['B6_VCID_1', 'B6_VCID_2'],
        'VISW': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'QA_PIXEL'],
        'VISW_TOA': ['B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'B8', 'QA_PIXEL']
    },
    'L8': {
        'TOA': ee.ImageCollection('LANDSAT/LC08/C02/T1_TOA'),
        'SR': ee.ImageCollection('LANDSAT/LC08/C02/T1_L2'),
        'TIR': ['B10', 'B11'],
        'VISW': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL'],
        'VISW_TOA': ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'QA_PIXEL']
    },
    'L9': {
        'TOA': ee.ImageCollection('LANDSAT/LC09/C02/T1_TOA'),
        'SR': ee.ImageCollection('LANDSAT/LC09/C02/T1_L2'),
        'TIR': ['B10', 'B11'],
        'VISW': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL'],
        'VISW_TOA': ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'QA_PIXEL']

    }
}

def filter_panchromatic_band(image):
    icemask = image.select('B8').lt(0.21) # iceberg reflectance threshold from Scheick et al., 2019 (UPPER BOUND)
    return image.updateMask(icemask)

def ice_mask(image):
    ## filtering based on NIR and SWIR ratio to get rid of sea ice
    NIR_SWIR_ratio = image.normalizedDifference(['B4', 'B5']) 
    ratio_mask = NIR_SWIR_ratio.lt(0.85)  
    return image.updateMask(ratio_mask)

def kelvin2celcius(image):
    image_celcius = image.select('SST').subtract(273.15)#.copyProperties(image, ['system:time_start', 'CLOUD_COVER', 'LANDSAT_PRODUCT_ID'])
    return image.addBands(image_celcius.rename('SST'), overwrite=True)

def mask_water(image, landsat):
    """Function to mask land pixels, keeping only water for SST"""
    # Select QA_PIXEL band (for Landsat 4-9)
    qa = image.select('QA_PIXEL')
    
    # Create water mask (bit 7 is water flag in Landsat QA_PIXEL)
    water_mask = qa.bitwiseAnd(1 << 7).eq(0)  # 0=water, 1=land
    
    # Apply additional cloud/ice masking based on sensor
    if landsat in ['L7', 'L8', 'L9']:
        image = filter_panchromatic_band(image)
    elif landsat in ['L5', 'L4']:
        image = ice_mask(image)
    
    # Mask SST band with water mask and add as new band
    sst_masked = image.select('SST').updateMask(water_mask.Not())
    
    # Add masked SST as new band while preserving original bands
    return image.addBands(sst_masked.rename('SST_masked'), overwrite=True)


# Collection function
def collection(landsat, date_start, date_end, validation=None):
    # Load TOA Radiance/Reflectance
    collection_dict = COLLECTION[landsat] # landsat is 'L4' or 'L5' etc...        

    landsatTOA = ee.ImageCollection(collection_dict['TOA'])\
                  .filter(ee.Filter.date(date_start, date_end))
    
    if validation==True: 
        if landsat in ['L7', 'L8', 'L9']:
            landsatTOA = landsatTOA.map(lambda image: cloud_mask.toa_cloud_mask(image)) # cloudmask filters
        elif landsat in ['L5', 'L4']:
            landsatTOA = landsatTOA.map(lambda image: cloud_mask.toa_cloud_mask(image))
        #Load Surface Reflectance collection for NDVI
        landsatSR = ee.ImageCollection(collection_dict['SR'])\
                    .filter(ee.Filter.date(date_start, date_end))\
                    .map(lambda image: cloud_mask.sr_cloud_mask(image))\
                    .map(lambda image: NCEP_TPW.NCEP_TPW(image))\
                    .map(lambda image: compute_emissivity.add_band(image))
    else:
        landsatSR = ee.ImageCollection(collection_dict['SR'])\
                    .filter(ee.Filter.date(date_start, date_end))\
                    .map(lambda image: NCEP_TPW.NCEP_TPW(image))\
                    .map(lambda image: compute_emissivity.add_band(image))

    # Combine collections
    # All channels from surface reflectance collection
    # except TIR channels: from TOA collection
    # Select TIR bands
    tir = collection_dict['TIR']
    visw = collection_dict['VISW'] + ['TPW', 'TPWpos', 'EM']
    visw_toa = collection_dict['VISW_TOA'] 
    landsatALL = landsatSR.select(visw).combine(landsatTOA.select(tir), True)

    # Compute the LST
    landsatLST = landsatALL.map(lambda image: SMW_algorithm.add_band(landsat, image))
    landsatLST_celsius = landsatLST.map(lambda image: kelvin2celcius(image))

    landsat_collection = landsatTOA.select(visw_toa).combine(landsatLST_celsius.select('SST', 'A_coeff', 'B_coeff', 'C_coeff', 'emissivity', 'brightness_temp'), True)

    landsat_collection = landsat_collection.map(lambda image: mask_water(image, landsat)) # Mask water pixels

    return landsat_collection









    # if validation==True: 
    #     if landsat in ['L7', 'L8', 'L9']:
    #         landsatTOA = landsatTOA.map(lambda image: filter_panchromatic_band(image)).map(lambda image: cloud_mask.toa_cloud_mask(image))
    #     elif landsat in ['L5', 'L4']:
    #         landsatTOA = landsatTOA.map(lambda image: ice_mask(image)).map(lambda image: cloud_mask.toa_cloud_mask(image))

    #     #Load Surface Reflectance collection for NDVI
    #     landsatSR = ee.ImageCollection(collection_dict['SR'])\
    #                 .filter(ee.Filter.date(date_start, date_end))\
    #                 .map(lambda image: cloud_mask.sr_cloud_mask(image))\
    #                 .map(lambda image: NCEP_TPW.NCEP_TPW(image))\
    #                 .map(lambda image: compute_emissivity.add_band(image))
    # else:
    #     landsatSR = ee.ImageCollection(collection_dict['SR'])\
    #                 .filter(ee.Filter.date(date_start, date_end))\
    #                 .map(lambda image: NCEP_TPW.NCEP_TPW(image))\
    #                 .map(lambda image: compute_emissivity.add_band(image))