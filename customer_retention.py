import numpy as np
from sklearn import preprocessing

# Import the data
raw_csv_data = np.loadtxt("Audiobooks-data.csv", delimiter=',')
raw_train_data = raw_csv_data[:, 1:-1]
raw_target_data = raw_csv_data[:, -1]


# Shuffle the dataset prior to balancing as we want to utilise as much data as possible for 0's
shuffled_indices_for_raw_data = np.arange(raw_target_data.shape[0])
np.random.shuffle(shuffled_indices_for_raw_data)

raw_train_data = raw_train_data[shuffled_indices_for_raw_data]
raw_target_data = raw_target_data[shuffled_indices_for_raw_data]

# Balancing the dataset
total_num_of_ones = int(np.sum(raw_target_data))
zero_counter = 0
indices_to_remove = []

for i in range(raw_target_data.shape[0]):
    if raw_target_data[i]==0:
        if zero_counter < total_num_of_ones:
            zero_counter += 1
        else:
            indices_to_remove.append(i)

balanced_training_data = np.delete(raw_train_data, indices_to_remove, axis=0)
balanced_target_data = np.delete(raw_target_data, indices_to_remove, axis=0)


# Scale the input data TODO::Check wtf this function does
scaled_training_data = preprocessing.scale(balanced_training_data)

indices_array = np.arange(scaled_training_data.shape[0])
np.random.shuffle(indices_array)
# Rearranges the array directly according to the array passed
shuffled_inputs = scaled_training_data[indices_array]
shuffled_targets = balanced_target_data[indices_array]

# Split the dataset into training, testing, and validation

samples_count = shuffled_inputs.shape[0]

train_count = int(0.8*samples_count)
test_count = int(0.1*samples_count)
validation_count = samples_count - train_count - test_count

input_train_data = shuffled_inputs[:train_count]
target_train_data = shuffled_targets[:train_count]

input_test_data = shuffled_inputs[train_count:train_count + test_count]
target_test_data = shuffled_targets[train_count:train_count + test_count]

input_validation_data = shuffled_inputs[train_count + test_count:]
target_validation_data = shuffled_targets[train_count + test_count:]

# print(input_train_data.shape[0] + input_validation_data.shape[0] + input_test_data.shape[0])
# print(samples_count)

np.savez("Audiobooks_training_data", inputs = input_train_data, targets = target_train_data)
np.savez("Audiobooks_test_data", inputs = input_test_data, targets = target_test_data)
np.savez("Audiobooks_validation_data", inputs = input_validation_data, targets = target_validation_data)


# Class for batching

import numpy as np

class AudioBooks_Data_Reader():

    # Load npz data into inputs and targets variables. Init the total number of batches (count) based on the batch size.
    def __init__(self, data, batch_size=None):

        npz = np.load('Audiobooks_{0}_data.npz'.format(data))

        self.inputs, self.targets = npz['inputs'].astype(np.float), npz['targets'].astype(int)

        # counts number of batches based on given size
        if batch_size is None:
            self.batch_size = self.inputs.shape[0]
        else:
            self.batch_size = batch_size

        self.current_batch = 0
        self.batch_count = self.inputs.shape[0] // self.batch_size



    # function that loads the next batch.
    # Check whether the batch count (current batch you are on) is > number of batches. If yes, stop iteration.
    def next(self):
        if self.current_batch >= self.batch_count:
            self.current_batch = 0
            raise StopIteration()
        # Get next batch slice by returning the input and targets for the given batch (ex. from 1-10, or 11-20 etc)

        batch_slice = slice(self.current_batch * self.batch_size, (self.current_batch + 1) * self.batch_size)

        input_batch = self.inputs[batch_slice]
        target_batch = self.targets[batch_slice]
        self.current_batch += 1
        # One hot encode the target_batch
        classes_num = 2
        targets_one_hot = np.zeros((target_batch.shape[0], classes_num)) # returns an array of 0's of size (input_size * classes)
        targets_one_hot[range(target_batch.shape[0]), target_batch] = 1 #TODO : What is happening here?

        # The function will return the inputs batch and the one-hot encoded targets
        return input_batch, targets_one_hot

    def __iter__(self):
        return self



# Building the Machine Learning Model.

import tensorflow as tf

input_size = 10
hiddenlayer_size = 25
output_size = 2
batch_size = 100
max_epochs = 100

number_hidden_layer = 1

tf.reset_default_graph()

inputs = tf.placeholder(tf.float32, [None, input_size])
targets = tf.placeholder(tf.int32, [None, output_size])

start_weights = tf.get_variable("start_weights", [input_size, hiddenlayer_size])
start_bias = tf.get_variable("start_bias", [hiddenlayer_size])

# print inputs, weights0
Input_next = tf.nn.relu(tf.matmul(inputs, start_weights) + start_bias)

for i in range(number_hidden_layer - 1): #-1 since out first hidden layer is created outside the for loop.
    weights = 'weights{}'.format(str(i))
    bias = 'bias{}'.format(str(i))
    Weights = tf.get_variable(weights, [hiddenlayer_size, hiddenlayer_size])
    Bias = tf.get_variable(bias, [hiddenlayer_size])
    Input_next = tf.nn.relu(tf.matmul(Input_next, Weights) + Bias)

final_weight = tf.get_variable("final_weight", [hiddenlayer_size, output_size])
final_bias = tf.get_variable("final_bias", [output_size])
# print hidden_layer
# Final layer matmul of weights and biases which is later used to calcualte the activation values.
outputs = tf.matmul(Input_next, final_weight) + final_bias

print("final output var: " + str(outputs))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = targets, logits = outputs))

optimizer = tf.train.AdamOptimizer(.001).minimize(loss)

# TODO: How does this generalize to models with more than 2 classes?
equal_vector = tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1))
accuracy = tf.reduce_mean(tf.cast(equal_vector, tf.float32))

sess = tf.InteractiveSession()
initializer = tf.global_variables_initializer()
# for var in tf.global_variables():
#     print var

sess.run(initializer)

prev_validation_loss = 999999999.

# Loads data using the Audiobooks_Data_Reader class.
train_data = AudioBooks_Data_Reader('training', batch_size=10)
validation_data = AudioBooks_Data_Reader('validation')
test_data = AudioBooks_Data_Reader('test')

#Training the model.
for epoch in range(max_epochs):
    current_epoch_loss = 0

    for input_batch, target_batch in train_data:
        _, batch_loss = sess.run([optimizer, loss], feed_dict = {
            inputs: input_batch, targets: target_batch
        })

        current_epoch_loss += batch_loss
    current_epoch_loss /= train_data.batch_count # train_data is the class instance of Audiobook class.

    # Validating the model with validation data.
    validation_loss = 0.
    validation_accuracy = 0.

    for input_batch, target_batch in validation_data:
        validation_loss, validation_accuracy = sess.run([loss, accuracy], feed_dict = {
            inputs: input_batch, targets: target_batch
        })

    print "Epoch: {}".format(str(epoch + 1))
    print("Epoch loss for training set: {} \nValidation loss: {}\nValidation Accuracy: {}%\n".format(
        current_epoch_loss,
        validation_loss,
        validation_accuracy * 100
    ))

    if prev_validation_loss < validation_loss:
        break

    prev_validation_loss = validation_loss

for input_batch, target_batch in test_data:
    test_loss, test_accuracy = sess.run([loss, accuracy], feed_dict={
        inputs: input_batch, targets: target_batch
    })

print "Test accuracy: {}%".format(test_accuracy * 100)