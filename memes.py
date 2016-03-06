from __future__ import absolute_import
from __future__ import print_function

import collections
import math
import os
import random
import zipfile

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


from read import words
import read

print("Number of words: " + str(len(words)))






# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 5000

def build_dataset(words):
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)
del words  # Hint to reduce memory.
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10])

data_index = 0


# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1 # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [ skip_window ]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
  print(batch[i], '->', labels[i, 0])
  print(reverse_dictionary[batch[i]], '->', reverse_dictionary[labels[i, 0]])

# Step 4: Build and train a skip-gram model.

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 2       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default():

  # Input data.
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  loss = tf.reduce_mean(
      tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels,
                     num_sampled, vocabulary_size))

  # Construct the SGD optimizer using a learning rate of 1.0.
  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

  # Compute the cosine similarity between minibatch examples and all embeddings.
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)







def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')









# Step 5: Begin training.
num_steps = 10001

with tf.Session(graph=graph) as session:
  # We must initialize all variables before we use them.
  tf.initialize_all_variables().run()
  print("Initialized")

  average_loss = 0
  for step in xrange(num_steps):
    batch_inputs, batch_labels = generate_batch(
        batch_size, num_skips, skip_window)
    feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}

    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val

    if step % 2000 == 0:
      if step > 0:
        average_loss /= 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print("Average loss at step ", step, ": ", average_loss)
      average_loss = 0

    # Note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 10000 == 0:
      sim = similarity.eval()
      for i in xrange(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8 # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k+1]
        log_str = "Nearest to %s:" % valid_word
        for k in xrange(top_k):
          try:
            close_word = reverse_dictionary[nearest[k]]
            log_str = "%s %s," % (log_str, close_word)
          except KeyError:
            log_str = "%s %s," % (log_str, 'err')
            

        print(log_str)

  final_embeddings = normalized_embeddings.eval()


  saver = tf.train.Saver()
  save_path = saver.save(session,"/tmp/model.ckpt")
  print("Model saved in file: %s" % save_path)




  x = tf.placeholder(tf.float32, shape=[None, 16384])
  y_ = tf.placeholder(tf.float32, shape=[None, read.m])
  #W = tf.Variable(tf.zeros([19200,read.m]))
  #b = tf.Variable(tf.zeros([read.m]))
  #session.run(tf.initialize_all_variables())
  #y = tf.nn.softmax(tf.matmul(x,W) + b)
  #cross_entropy = -tf.reduce_sum(y_*tf.log(y))


  W_conv1 = weight_variable([5, 5, 1, 32])
  b_conv1 = bias_variable([32])
  x_image = tf.reshape(x, [-1,128,128,1])
  
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  h_pool1 = max_pool_2x2(h_conv1)

  W_conv2 = weight_variable([5, 5, 32, 64])
  b_conv2 = bias_variable([64])

  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  h_pool2 = max_pool_2x2(h_conv2)

  W_conv3 = weight_variable([5, 5, 64, 128])
  b_conv3 = bias_variable([128])

  h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
  h_pool3 = max_pool_2x2(h_conv3)

  W_fc1 = weight_variable([16 * 16 * 128, 1024])
  b_fc1 = bias_variable([1024])

  h_pool3_flat = tf.reshape(h_pool3, [-1, 16*16*128])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
  
  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  W_fc2 = weight_variable([1024, read.m])
  b_fc2 = bias_variable([read.m])

  y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

  cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  session.run(tf.initialize_all_variables())


  #train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

  lines = []
  for line in read.lines:
    l = []

   
    for i in range(0, len(line)):
      try:      
        word = line[i]
        l = np.concatenate((l, (final_embeddings[dictionary[word]])))

      except KeyError:
        a = 1 + 1
        #print(word)
      
    l = np.concatenate((l, (np.full(128 * 128-len(l),0))))#* (150 - len(l)))))
    lines.append(np.array(l))



  for i in range(read.m/25):
    batch = [lines[25*i : 25*i + 50], read.onehots[25*i: 25*i + 50]]
    #batch = mnist.train.next_batch(50)
    if i%100 == 0:
      train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
      print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

  print("test accuracy %g"%accuracy.eval(feed_dict={x: lines, y_: read.onehots, keep_prob: 1.0}))


#  for i in range(500):

    #batch = [lines[7*i : 7*i + 6], read.onehots[7*i: 7*i + 6]]

 #   train_step.run(feed_dict={x: lines[7*i : 7*i + 7], y_: read.onehots[7*i: 7*i + 7]})


 # correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
 # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
 # print(accuracy.eval(feed_dict={x: lines, y_: read.onehots}))




def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
  plt.figure(figsize=(18, 18))  #in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i,:]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)

try:
  from sklearn.manifold import TSNE
  import matplotlib as mlp
  mlp.use('Agg')

  import matplotlib.pyplot as plt

  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
  plot_only = 500
  low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only,:])
  labels = [reverse_dictionary[i] for i in xrange(plot_only)]
  plot_with_labels(low_dim_embs, labels)

except ImportError:
  print("Please install sklearn and matplotlib to visualize embeddings.")


labels = [reverse_dictionary[i] for i in reverse_dictionary]


print(len(labels))
print (len(final_embeddings))

f = open('labels.txt','w')
for a in labels:
  f.write(a + '\n')
f.close













#f2 = open('embeddings.txt' , 'w')
#for a in final_embeddings:
#  f2.write(str(a) + '\n')
#f2.close
