import flask
import tensorflow as tf
import keras
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler


# app.graph = load_graph(modelpath)
app = flask.Flask(__name__)
with open("scaler.pkl","rb") as file:
    scale = pickle.load(file)

# make sure the model on the same graph thread
g = tf.Graph()
with g.as_default():
    print("Loading model")
    modelpath = './weather.h5'
    model = keras.models.load_model(modelpath)

@app.route('/predict', methods=['GET', 'POST'])
def demo():
    json = flask.request.get_json()
    #print(str(json))
    day1 = np.asarray(json['day1']).reshape((12,10))
    day2 = np.asarray(json['day2']).reshape((12,10))
    day3 = np.asarray(json['day3']).reshape((12,10))
    input = np.stack((day1,day2,day3))
    for index, item in enumerate(input):
        input[index] = scale.transform(item)

    with g.as_default():
        pred = model.predict(input)
    return str([x[0] for x in pred])


if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)