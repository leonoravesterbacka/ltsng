import tensorflow as tf



def buildModel(size, embeddingDim, rnnUnits, batchSize):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(size, embeddingDim,
                              batch_input_shape=[batchSize, None]),
    tf.keras.layers.GRU(rnnUnits,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(size)
  ])
  return model


def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

def generateText(model, char2idx, idx2char, startString):
  # Evaluation step 
  print(startString)
  # Number of characters to generate
  nGenerate = 1000

  # Converting our start string to numbers (vectorizing)
  inputEval = [char2idx[s] for s in startString]
  inputEval = tf.expand_dims(inputEval, 0)

  # Empty string to store our results
  textGenerated = []

  temperature = 1.0

  model.reset_states()
  for i in range(nGenerate):
      predictions = model(inputEval)
      predictions = tf.squeeze(predictions, 0)

      # using a categorical distribution to predict the character returned by the model
      predictions = predictions / temperature
      predictedId = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # We pass the predicted character as the next input to the model
      inputEval = tf.expand_dims([predictedId], 0)

      textGenerated.append(idx2char[predictedId])
  return (startString + ''.join(textGenerated))
