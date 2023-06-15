import uvicorn
import tensorflow as tf
import numpy as np

from typing import Union
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

from PIL import Image

from io import BytesIO
import keras.utils as image
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
    gambar = gambar.convert("RGB")
    gambar = gambar.resize((150, 150))
    gambar = image.img_to_array(gambar)
    

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
    # print(label)
    return final_res

def pick_fruits_or_vegetables(label):
    # Final Labels
    if label == 0:
        return "Fresh Tomato"
    elif label == 1:
        return "Rotten Tomato"
    elif label == 2:
        return "Rotten Banana"
    elif label == 3:
        return "Rotten Carrot"
    elif label == 4:
        return "Fresh Carrot"
    elif label == 5:
        return "Rotten Cucumber"
    elif label == 6:
        return "Fresh Mango"
    elif label == 7:
        return "Rotten Mango"
    elif label == 8:
        return "Rotten Capsicum"
    elif label == 9:
        return "Fresh Capsicum"
    elif label == 10:
        return "Fresh Banana"
    elif label == 11:
        return "Rotten Apple"
    elif label == 12:
        return "Fresh Strawberry"
    elif label == 13:
        return "Rotten Strawberry"
    elif label == 14:
        return "Rotten Orange"
    elif label == 15:
        return "Fresh Cucumber"
    elif label == 16:
        return "Fresh Apple"
    else:
        return "Fresh Orange"

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
    uvicorn.run(app, host='0.0.0.0',port=8080)
