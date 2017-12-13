import numpy as np
import tensorflow as tf
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from IPython import get_ipython
import re
from random import randint
import datetime

wordsList = np.load('wordsList.npy')
# print('Loaded the word list!')
wordsList = wordsList.tolist() #Originally loaded as numpy array
wordsList = [word.decode('UTF-8') for word in wordsList] #Encode words as UTF-8
wordVectors = np.load('wordVectors.npy')
# print ('Loaded the word vectors!')
numDimensions = 300 #Dimensions for each word vector


positiveFiles = ['positiveReviews/' + f for f in listdir('positiveReviews/') if isfile(join('positiveReviews/', f))]
negativeFiles = ['negativeReviews/' + f for f in listdir('negativeReviews/') if isfile(join('negativeReviews/', f))]
numWords = []
for pf in positiveFiles:
    with open(pf, "r", encoding='utf-8') as f:
        line=f.readline()
        counter = len(line.split())
        numWords.append(counter)
# print('Positive files finished')

for nf in negativeFiles:
    with open(nf, "r", encoding='utf-8') as f:
        line=f.readline()
        counter = len(line.split())
        numWords.append(counter)
# print('Negative files finished')

# numFiles = len(numWords)
# print('The total number of files is', numFiles)
# print('The total number of words in the files is', sum(numWords))
# print('The average number of words in the files is', sum(numWords)/len(numWords))

# run_line_magic('matplotlib', 'inline')
# plt.hist(numWords, 50)
# plt.xlabel('Sequence Length')
# plt.ylabel('Frequency')
# plt.axis([0, 1200, 0, 8000])
# plt.show()

maxSeqLength = 250

strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())

# ids = np.zeros((numFiles, maxSeqLength), dtype='int32')
# fileCounter = 0
# for pf in positiveFiles:
#    with open(pf, "r", encoding="utf8") as f:
#        indexCounter = 0
#        line=f.readline()
#        cleanedLine = cleanSentences(line)
#        split = cleanedLine.split()
#        for word in split:
#            try:
#                ids[fileCounter][indexCounter] = wordsList.index(word)
#            except ValueError:
#                ids[fileCounter][indexCounter] = 399999 #Vector for unkown words
#            indexCounter = indexCounter + 1
#            if indexCounter >= maxSeqLength:
#                break
#        fileCounter = fileCounter + 1
#
# for nf in negativeFiles:
#    with open(nf, "r", encoding="utf8") as f:
#        indexCounter = 0
#        line=f.readline()
#        cleanedLine = cleanSentences(line)
#        split = cleanedLine.split()
#        for word in split:
#            try:
#                ids[fileCounter][indexCounter] = wordsList.index(word)
#            except ValueError:
#                ids[fileCounter][indexCounter] = 399999 #Vector for unkown words
#            indexCounter = indexCounter + 1
#            if indexCounter >= maxSeqLength:
#                break
#        fileCounter = fileCounter + 1
#Pass into embedding function and see if it evaluates.

# np.save('idsMatrix', ids)
ids = np.load('idsMatrix.npy')

def getTrainBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        if (i % 2 == 0):
            num = randint(1,11499)
            labels.append([1,0])
        else:
            num = randint(13499,24999)
            labels.append([0,1])
        arr[i] = ids[num-1:num]
    return arr, labels

def getTestBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        num = randint(11499,13499)
        if (num <= 12499):
            labels.append([1,0])
        else:
            labels.append([0,1])
        arr[i] = ids[num-1:num]
    return arr, labels

batchSize = 24
lstmUnits = 64
numClasses = 2
iterations = 201#100000

tf.reset_default_graph()
labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
data = tf.nn.embedding_lookup(wordVectors,input_data)

lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)

correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(loss)

# with tf.Session() as sess:
#     tf.summary.scalar('Loss', loss)
#     tf.summary.scalar('Accuracy', accuracy)
#     merged = tf.summary.merge_all()
#     logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
#     writer = tf.summary.FileWriter(logdir, sess.graph)

sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

### CODE TO TRAIN DATA ###
# for i in range(iterations):
#    #Next Batch of reviews
#    nextBatch, nextBatchLabels = getTrainBatch();
#    sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})
#
#    #Write summary to Tensorboard
#    if (i % 50 == 0):
#        summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
#        writer.add_summary(summary, i)
#
#    #Save the network every 10,000 training iterations
#    if (i % 200 == 0 and i != 0):
#        save_path = saver.save(sess, "models/pretrained_smalllstm.ckpt", global_step=i)
#        print("saved to %s" % save_path)
# writer.close()

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint('models'))

saver.restore(sess, "models/pretrained_lstm.ckpt-80000")
# saver.restore(sess, "models/pretrained_smalllstm.ckpt-200")

# iterations = 30
# tot = 0
# for i in range(iterations):
#     nextBatch, nextBatchLabels = getTestBatch();
#     num = (sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})) * 100
#     print("Accuracy for this batch:", num)
    # tot += num
# print(tot/iterations)

def getMatrix(s):
    split = s.split(' ')
    firstSentence = np.zeros((24,250), dtype='int32')
    indexCounter = 0
    for word in split:
        try:
            firstSentence[0][indexCounter] = wordsList.index(word)
        except ValueError:
            firstSentence[0][indexCounter] = 399999 #Vector for unkown words
        indexCounter = indexCounter + 1
    return firstSentence

if __name__ == "__main__":
  while(1):
    review = input("Please submit a movie review!!!\n")
    if review == "quit":
        exit(0)
    inp = getMatrix(review)
    pred = sess.run(prediction,{input_data: inp})[0]
    if pred[0] > pred[1]:
        print("Probability positive: "+str(pred[0])+'\n')
        print("Probability negative: "+str(pred[1])+'\n')
        print("Positive Sentiment\n\n")
    elif pred[0] < pred[1]:
        print("Probability positive: "+str(pred[0])+'\n')
        print("Probability negative: "+str(pred[1])+'\n')
        print("Negative Sentiment\n\n")
    else:
        print("Neutral Sentiment\n\n")
