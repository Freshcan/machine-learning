import uvicorn
import tensorflow as tf
import numpy as np

from typing import Union
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

from PIL import Image

from io import BytesIO
from tensorflow.keras.utils import load_img, img_to_array
from fastapi import FastAPI, File, UploadFile

app = FastAPI(title="Freshcan Classifier API")

class Img(BaseModel):
    img_url: str


def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image


def predict(gambar):
    loaded_model = tf.keras.models.load_model('model.h5')

    # Preprocess
    gambar = gambar.resize((150, 150))
    gambar = img_to_array(gambar)
    

    gambar = np.expand_dims(gambar, 0)
    gambar = gambar/255

    gambar = np.vstack([gambar])

    # Predict
    all_res = loaded_model(gambar)[0]
    label = tf.argmax(all_res).numpy()
    precentage = round(float(all_res.numpy()[label])*100, 2)
    #print(all_res.numpy())

    chosen_class = pick_fruits_or_vegetables(label)
    final_res = chosen_class + " " + str(precentage) + "%"

    return final_res

def pick_fruits_or_vegetables(label):
    # 20 remaining labels will be added after optimizing the model
    # There will be some changes for the label
    if label == 0:
        return "Rotten Banana"
    elif label == 1:
        return "Fresh Cucumber"
    elif label == 2:
        return "Rotten Cucumber"
    elif label == 3:
        return "Rotten Tomato"
    elif label == 4:
        return "Fresh Orange"
    elif label == 5:
        return "Rotten Tomato"
    elif label == 6:
        return "Rotten Apple"
    elif label == 7:
        return "Rotten Orange"
    elif label == 8:
        return "Fresh Banana"
    else:
        return "Fresh Apple"

@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    prediction = predict(image)
    #print(prediction)
    return prediction

# Test in local
if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0',port=8001)