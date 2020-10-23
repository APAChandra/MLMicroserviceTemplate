import asyncio
from PIL import Image
from SodaNet.sodanet_model import SodaModel
from matplotlib.image import imread
import numpy
import os
import shutil
model = None
async def init():
    """
    This method will be run once on startup. You should check if the supporting files your
    model needs have been created, and if not then you should create/fetch them.
    """
    await asyncio.sleep(2)
    print('aaa')
    global model

    # Loading the sodanet module
    print('Loading SodaNet model')
    model = SodaModel()


def predict(image_file):
    """
    Interface method between model and server. This signature must not be
    changed and your model must be able to predict given a file-like object
    with the image as an input.
    """
    global model
    # image = Image.open(image_file.name)
    image = Image.open(image_file.name)
    numpydata = numpy.asarray(image)
    if model == None:
        raise RuntimeError("SodaNet model is not loaded properly")
    print(image_file.name)
    print(image.size)
    model.load_image(numpydata)
    predicted, im_ret = model.evaluate()
    #predicted_value_converted_to_yn = "Yes" if str(predicted_value) == 1 else "No"
    print(predicted)
    return {
        "Contains Coke (Can)": str(predicted[0]),
    }
