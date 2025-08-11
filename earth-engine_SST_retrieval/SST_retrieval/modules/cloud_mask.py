import ee
ee.Initialize()

# Define the TOA cloud mask function
# def toa_cloud_mask(image):
#     qa = image.select('QA_PIXEL')
#     mask = qa.bitwiseAnd(1 << 1).Or(qa.bitwiseAnd(1 << 3)).Or(qa.bitwiseAnd(1 << 4)) #https://gis.stackexchange.com/questions/407458/google-earth-engine-understanding-landsat-7-collection-2-qa-pixel-bitwiseand
#     return image.updateMask(mask.Not())

# Define the SR cloud mask function
def sr_cloud_mask(image):
    qa = image.select('QA_PIXEL')
    mask = qa.bitwiseAnd(1 << 3).Or(qa.bitwiseAnd(1 << 5))
    return image.updateMask(mask.Not())


# Define the function for bitwise extraction
def bitwise_extract(input, from_bit, to_bit):
    mask_size = ee.Number(1).add(to_bit).subtract(from_bit)
    mask = ee.Number(1).leftShift(mask_size).subtract(1)
    return input.rightShift(from_bit).bitwiseAnd(mask)

# # Define the function to mask Landsat images based on QA bands
# def toa_cloud_mask(landsat_image):
#     # Select the QA band
#     qa = landsat_image.select('QA_PIXEL')

#     # Perform bitwise extraction for different QA bands
#     dilated_cloud = bitwise_extract(qa, 1, 1).eq(0)
#     cloud = bitwise_extract(qa, 3, 3).eq(0)
#     cloud_shadow = bitwise_extract(qa, 4, 4).eq(0)
#     cloud_confidence = bitwise_extract(qa, 8, 9).lt(2)
#     cloud_shadow_confidence = bitwise_extract(qa, 10, 11).lt(2)
#     cirrus_confidence = bitwise_extract(qa, 14, 15).lt(2)

#     # Create masks based on extracted QA bands
#     mask = dilated_cloud.And(cloud).And(cloud_shadow).And(cloud_confidence).And(cloud_shadow_confidence).And(cirrus_confidence)
#     saturation_mask = landsat_image.select('QA_RADSAT').eq(0)

#     # Update the Landsat image mask
#     return landsat_image.updateMask(mask).updateMask(saturation_mask)

def toa_cloud_mask(image):
    # Select the QA_PIXEL band
    qa = image.select('QA_PIXEL')
    
    # Apply bitwise operations to filter out clouds and shadows
    cloud = qa.bitwiseAnd(1 << 5) \
             .And(qa.bitwiseAnd(1 << 7)) \
             .Or(qa.bitwiseAnd(1 << 3))
    
    # Reduce the mask to minimize edge effects
    mask2 = image.mask().reduce(ee.Reducer.min())
    
    # Update the image mask to exclude clouds and shadows
    return image.updateMask(cloud.Not()).updateMask(mask2)