# Freshcan Machine Learning Model
We have built an image classification model to detect fruit or vegetable freshness. Our model will give output of the type of fruit or vegetable and its percentage of how fresh or rotten it is. For this project. we only train several types of fruits and vegetables.
## Tech Stack
- [Tensorflow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [FastAPI](https://fastapi.tiangolo.com/id/)
## Datasets
There are 2 datasets from kaggle that we use. Here are the links for those datasets.
-   https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification
-   https://www.kaggle.com/datasets/raghavrpotdar/fresh-and-stale-images-of-fruits-and-vegetables
We pick 9 types of fruits and vegetables which consist of apple, banana, orange, tomato, cucumber, carrot, mango, strawberry, and capsicum. Each fruit and vegetable will be classified as rotten or fresh, so in total we have 18 classes.
## ML Model Detail
We train our model using Convolutional Neural Networks and transfer learning InceptionV3. InceptionV3 is chosen based on PoC amongst the other 2 models that use pre-trained model VGG16 and customized model.
## Prequisites
If you want to try our model on your local or modify it, you can follow the steps below.
- Clone the repo 
```sh
git clone https://github.com/Freshcan/machine-learning.git
```
- Install all requirements using the command below. Make sure that you've installed python 3.10 or above.
```sh
pip install -r requirements.txt
```
- You can use jupyter notebook or Google Colab to run the code. You can simply just put the model into your Colab account then run the model or follow the steps below to use jupyter notebook.
```sh
pip install jupyterlab
jupyter-lab
pip install notebook
jupyter notebook
```
- Run main.py if you want to try the model's API locally. Open http://127.0.0.1:8001/docs#/ to try uploading the image and see the prediction result.
## Authors
- Atifah Nabilla Al Qibtiyah
- Indra Ma'dika
- Kartika Dian Pratiwi
