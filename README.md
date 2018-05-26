### 1.mnist Demo.py
#### This is a tensorflow demo using the famous mnist data
MLP structure is dynamic
```python
structure = [784,500,500,10]
x = tf.placeholder(dtype='float',shape=[None,structure[0]])
y = tf.placeholder(dtype='float')
batch_size = 100
epoch_max = 10
```
#### The model inculdes two part: feedforward and traning
```python
# feedforward Model
def netural_network_model(x):
#training
def train_model(x):
```

### 2.word sentiment model
#### put the training and test dataset into folder “sentiment140” which you can download from [here](http://help.sentiment140.com/for-students/)
```python
# word2vec.py is used for word to vector using one-hot coding
# ANN.py is ANN structure and training , testing parts

```