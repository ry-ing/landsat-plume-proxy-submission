import ee
ee.Initialize()

# Define the addBand function
def add_band(image):

    # Create an image for emissivity with a constant value of 0.99
    EM = ee.Image.constant(0.967)
    
    # Prescribe emissivity of water bodies
    qa = image.select('QA_PIXEL')
    EM = EM.where(qa.bitwiseAnd(1 << 7), 0.967)
    
    # Prescribe emissivity of snow/ice bodies
    EM = EM.where(qa.bitwiseAnd(1 << 5), 0.967)
    
    # Add the emissivity band to the image
    image_with_EM = image.addBands(EM.rename('EM'))
    
    return image_with_EM



