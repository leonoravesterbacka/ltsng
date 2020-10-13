from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
import os
import time
import optparse

from include import processInputTarget
from ml import buildModel
from ml import loss

parser = optparse.OptionParser(usage="usage: %prog [opts]", version="%prog 1.0")
parser.add_option('-n', '--nepochs',  action='store', type=int, dest='nepochs',  default=10, help='specify the number of epochs')
(opts, args) = parser.parse_args()
EPOCHS       = opts.nepochs

##load data, in this case name of London tube stations
file_stations = tf.keras.utils.get_file('tubeStations.txt', 'https://github.com/leonoravesterbacka/ltsng/blob/main/data/tubeStations.txt')
list_stations = open(file_stations, 'rb').read().decode(encoding='utf-8')
print ('First couple of station names: {}'.format(list_stations[:25]))
print(list_stations[:250])
sort = sorted(set(list_stations))
print ('{} unique characters'.format(len(sort)))

##cast the chars as integers 
char2idx = {u:i for i, u in enumerate(sort)}
idx2char = np.array(sort)

list_as_int = np.array([char2idx[c] for c in list_stations])
print('{')
for char,_ in zip(char2idx, range(20)):
    print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
print('  ...\n}')
print ('{} ---- characters mapped to int ---- > {}'.format(repr(list_stations[:13]), list_as_int[:13]))

# The maximum length sentence we want for a single input in characters
seq_length = 20
examples_per_epoch = len(list_stations)//(seq_length+1)

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(list_as_int)

for i in char_dataset.take(100):
    print(idx2char[i.numpy()])

##The `batch` method converts these individual characters to sequences of the desired size.
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

for item in sequences.take(5):
    print(repr(''.join(idx2char[item.numpy()])))

##For each sequence, duplicate and shift it to form the input and target text by using the `map` method to apply a simple function to each batch:
dataset = sequences.map(processInputTarget)

#Print the first examples input and target values:
for input_example, target_example in  dataset.take(1):
    print ('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
    print ('Target data:', repr(''.join(idx2char[target_example.numpy()])))

for i, (input_idx, target_idx) in enumerate(zip(input_example[:10], target_example[:10])):
    print("Step {:4d}".format(i))
    print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
    print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))

#shuffle data nad pack into batches
BATCH_SIZE = 64
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# Length of the sorted list in chars
size = len(sort)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024

model = buildModel(
  size = len(sort),
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=BATCH_SIZE)

#try the model
print(dataset.take(1))
for input_example_batch, target_example_batch in dataset.take(5):
    example_batch_predictions = model(input_example_batch)
model.summary()

sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()

##start the training
example_batch_loss  = loss(target_example_batch, example_batch_predictions)
print("prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, size)")
print("scalar_loss:      ", example_batch_loss.numpy().mean())

model.compile(optimizer='adam', loss=loss)

# save checkpoints in following directory
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)


history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
