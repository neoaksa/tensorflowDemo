## what is this model used for？
This model is used to predict probability of snow day according to previous hours weather conditions. 
The data is download from NOAA where you can find local weather data. I used the data of Grand Rapids in period of 2014-2018.
The training process is deployed in Colaboratory by Google, the script is "**weather_prediction.ipynb**", keras model is "**weather.h5**",tensorflow model is "**weather.pd**", scale file is "**scale.pkl**", source file is "**input.csv**".

## what is the structure of this model?
1. input: 10 scaled columns
2. RNN: 2 GRU layer 
3. Output: 2 nodes with softmax

## How to deploy this model
Using flask running on python is pretty easy way to open this  model to public. You can find code in **server.py**.

## How to call server on android
Using Json to transfer array to flask server, and get feedback by string. You can find code in **Predict.java**.

## issue to be waiting to fix:
I dont think Json is a good way to transfer multidimension array, there must be another way, maybe [flatbuffers](https://google.github.io/flatbuffers/) is a good method.
