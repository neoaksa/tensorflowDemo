"""
this snippet transfers words into one-hot vector which can be handled
in MLP
"""
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import pandas as pd

lemmatizer = WordNetLemmatizer()

def init_process(fin, fout):
    outfile = open(fout, 'a')
    with open(fin, buffering=200000, encoding='latin-1') as f:
        try:
            # replace special characters
            for line in f:
                line = line.replace('"', '')
                # the first column is label
                initial_polarity = line.split(',')[0]
                # 0=negative 4=positive
                if initial_polarity == '0':
                    initial_polarity = [1, 0]
                elif initial_polarity == '4':
                    initial_polarity = [0, 1]
                # the last column is input
                tweet = line.split(',')[-1]
                outline = str(initial_polarity) + ':::' + tweet
                outfile.write(outline)
        except Exception as e:
            print(str(e))
    outfile.close()

# save for training and testing dataset
init_process('sentiment140/training.1600000.processed.noemoticon.csv', 'train_set.csv')
init_process('sentiment140/testdata.manual.2009.06.14.csv', 'test_set.csv')

# create lexicon
def create_lexicon(fin):
    lexicon = []
    with open(fin, 'r', buffering=100000, encoding='latin-1') as f:
        try:
            counter = 1
            content = ''
            for line in f:
                counter += 1
                # randomly pick up line for sampling to lexicon
                if (counter / 2500.0).is_integer():
                    tweet = line.split(':::')[1]
                    content += ' ' + tweet
                    words = word_tokenize(content)
                    words = [lemmatizer.lemmatize(i) for i in words]
                    lexicon = list(set(lexicon + words))
                    print(counter, len(lexicon))

        except Exception as e:
            print(str(e))

    with open('lexicon.pickle', 'wb') as f:
        pickle.dump(lexicon, f)

# create lexicon and save to pickle
create_lexicon('train_set.csv')

# convert dataset input to vector
def convert_to_vec(fin, fout, lexicon_pickle):
    # open lexicon
    with open(lexicon_pickle, 'rb') as f:
        lexicon = pickle.load(f)
    outfile = open(fout, 'a')
    with open(fin, buffering=20000, encoding='latin-1') as f:
        counter = 0
        for line in f:
            counter += 1
            label = line.split(':::')[0]
            tweet = line.split(':::')[1]
            # tokenize to array
            current_words = word_tokenize(tweet.lower())
            # lemmatize for each element
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            # one-hot coding
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    # OR DO +=1, test both
                    features[index_value] += 1

            features = list(features)
            outline = str(features) + '::' + str(label) + '\n'
            outfile.write(outline)

        print(counter)


convert_to_vec('test_set.csv', 'processed-test-set.csv', 'lexicon.pickle')


def shuffle_data(fin):
    df = pd.read_csv(fin, error_bad_lines=False)
    df = df.iloc[np.random.permutation(len(df))]
    print(df.head())
    df.to_csv('train_set_shuffled.csv', index=False)


shuffle_data('train_set.csv')

# split the csv into x, y dataset
def create_test_data_pickle(fin):
    feature_sets = []
    labels = []
    counter = 0
    with open(fin, buffering=20000) as f:
        for line in f:
            try:
                features = list(eval(line.split('::')[0]))
                label = list(eval(line.split('::')[1]))

                feature_sets.append(features)
                labels.append(label)
                counter += 1
            except:
                pass
    print(counter)
    feature_sets = np.array(feature_sets)
    labels = np.array(labels)

create_test_data_pickle('processed-test-set.csv')