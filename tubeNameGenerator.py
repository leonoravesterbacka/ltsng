from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
import os
import time
import optparse
import urllib

from include import processInputTarget
from ml import buildModel
from ml import loss
from ml import generateText

parser = optparse.OptionParser(usage="usage: %prog [opts]", version="%prog 1.0")
parser.add_option('-n', '--nepochs',  action='store', type=int, dest='nepochs',  default=10, help='specify the number of epochs')
(opts, args) = parser.parse_args()
EPOCHS       = opts.nepochs

##load data, in this case name of London tube stations
fileStations = tf.keras.utils.get_file('tubestations3.txt', 'http://mvesterb.web.cern.ch/mvesterb/tubestations3.txt')
listStations = open(fileStations, 'rb').read().decode(encoding='utf-8')
print ('First couple of station names: {}'.format(listStations[:25]))
print(listStations[:250])
sort = sorted(set(listStations))
print ('{} unique characters'.format(len(sort)))

##cast the chars as integers 
char2idx = {u:i for i, u in enumerate(sort)}
idx2char = np.array(sort)

listAsInt = np.array([char2idx[c] for c in listStations])

# The maximum length sentence we want for a single input in characters
seqLen = 20
examples = len(listStations)//(seqLen+1)

# Create training examples / targets
charDataset = tf.data.Dataset.from_tensor_slices(listAsInt)

for i in charDataset.take(100):
    print(idx2char[i.numpy()])

##The `batch` method converts these individual characters to sequences of the desired size.
sequences = charDataset.batch(seqLen+1, drop_remainder=True)

for item in sequences.take(5):
    print(repr(''.join(idx2char[item.numpy()])))

##For each sequence, duplicate and shift it to form the input and target text by using the `map` method to apply a simple function to each batch:
dataset = sequences.map(processInputTarget)

#Print the first examples input and target values:
for inputExample, targetExample in  dataset.take(1):
    print ('Input data: ', repr(''.join(idx2char[inputExample.numpy()])))
    print ('Target data:', repr(''.join(idx2char[targetExample.numpy()])))

for i, (inputIdx, targetIdx) in enumerate(zip(inputExample[:10], targetExample[:10])):
    print("Step {:4d}".format(i))
    print("  input: {} ({:s})".format(inputIdx, repr(idx2char[inputIdx])))
    print("  expected output: {} ({:s})".format(targetIdx, repr(idx2char[targetIdx])))

#shuffle data nad pack into batches
BATCH_SIZE = 64
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# Length of the sorted list in chars
size = len(sort)

# The embedding dimension
embeddingDim = 256

# Number of RNN units
rnnUnits = 1024

model = buildModel(
  size = len(sort),
  embeddingDim=embeddingDim,
  rnnUnits=rnnUnits,
  batchSize=BATCH_SIZE)

#try the model
print(dataset.take(1))
for inputExampleBatch, targetExampleBatch in dataset.take(5):
    exampleBatchPredictions = model(inputExampleBatch)
model.summary()

sampledIndices = tf.random.categorical(exampleBatchPredictions[0], num_samples=1)
sampledIndices = tf.squeeze(sampledIndices,axis=-1).numpy()

##start the training
exampleBatchLoss  = loss(targetExampleBatch, exampleBatchPredictions)
print("prediction shape: ", exampleBatchPredictions.shape, " # (batch size, sequence length, size)")
print("loss:      ", exampleBatchLoss.numpy().mean())

model.compile(optimizer='adam', loss=loss)

# save checkpoints in following directory
checkpointPath = './training'
checkpointPrefix = os.path.join(checkpointPath, "ckpt_{epoch}")

checkpointCallback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpointPrefix,
    save_weights_only=True)


history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpointCallback])


model = buildModel(size, embeddingDim, rnnUnits, batchSize=1)
model.load_weights(tf.train.latest_checkpoint(checkpointPath))
model.build(tf.TensorShape([1, None]))

model.summary()

output = generateText(model, char2idx, idx2char, startString=u"E")
print("output:  ", output)
with open('output/result_'+str(EPOCHS)+'.txt', 'w') as file:
   file.write(output)

