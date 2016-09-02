
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import time

# In[2]:

VALIDATION_SIZE = 0.2

INPUT_SIZE = 192
OUTPUT_SIZE = 99
FIRST_LAYER_HIDDEN = 4096
SECOND_LAYER_HIDDEN = 4096
THIRD_LAYER_HIDDEN = 1024
FOURTH_LAYER_HIDDEN = 521

LEARNING_RATE = 0.5
TRAINING_ITERATIONS = 25001
BATCH_SIZE = 16
EARLY_STOPPING_ROUND = 50
DROP_OUT = 0.5

BATCH_NORM_EPISILON = 1e-3


# In[3]:

train = pd.read_csv("train.csv")
train_x = train.drop(["id","species"], axis=1)
le = LabelEncoder().fit(train["species"])
train_y = le.transform(train["species"])


# In[4]:

train_x = train_x.iloc[:,:].values


# In[5]:

class_num = np.unique(train_y).shape[0]
label_num = len(train_y)


# In[6]:

def dense_to_one_hot(dense_label, class_num):
    label_num = len(dense_label)
    index_offset = np.arange(label_num) * class_num
    labels_one_hot = np.zeros((label_num, class_num))
    labels_one_hot.flat[index_offset + dense_label] = 1
    return labels_one_hot


# In[7]:

train_y = dense_to_one_hot(train_y, class_num)


# In[8]:

VALIDATION_SIZE = int(train_x.shape[0] * VALIDATION_SIZE)
valid_x = train_x[:VALIDATION_SIZE]
valid_y = train_y[:VALIDATION_SIZE]
train_x = train_x[VALIDATION_SIZE:]
train_y = train_y[VALIDATION_SIZE:]


# In[9]:

def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# In[20]:

x = tf.placeholder(tf.float32, shape=[None, INPUT_SIZE], name="ph_x")
y_ = tf.placeholder(tf.float32, shape=[None, OUTPUT_SIZE], name="ph_y")

keep_prob = tf.placeholder('float32', name="ph_dropout")

weight_1 = weight_variable([INPUT_SIZE, FIRST_LAYER_HIDDEN],"w_1")
bias_1 = bias_variable([FIRST_LAYER_HIDDEN],"b_1")

weight_2 = weight_variable([FIRST_LAYER_HIDDEN, SECOND_LAYER_HIDDEN],"w_2")
bias_2 = bias_variable([SECOND_LAYER_HIDDEN],"b_2")

weight_3 = weight_variable([SECOND_LAYER_HIDDEN, THIRD_LAYER_HIDDEN],"w_3")
bias_3 = bias_variable([THIRD_LAYER_HIDDEN],"b_3")

weight_4 = weight_variable([THIRD_LAYER_HIDDEN, FOURTH_LAYER_HIDDEN],"w_4")
bias_4 = bias_variable([FOURTH_LAYER_HIDDEN],"b_4")

weight_output = weight_variable([FOURTH_LAYER_HIDDEN, OUTPUT_SIZE],"w_o")
bias_output = bias_variable([OUTPUT_SIZE],"b_o")

'''
weight_output = weight_variable([SECOND_LAYER_HIDDEN, OUTPUT_SIZE],"w_o")
bias_output = bias_variable([OUTPUT_SIZE],"b_o")
'''
#Fully connected network
'''h_fc1 = tf.nn.relu(tf.matmul(x, weight_1) + bias_1)
'''
'''h_fc2 = tf.nn.relu(tf.matmul(h_fc1, weight_2) + bias_2)

h_fc3 = tf.nn.relu(tf.matmul(h_fc2, weight_3) + bias_3)
h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)

y = tf.nn.softmax(tf.matmul(h_fc3_drop, weight_output) + bias_output)'''

#with batch normalization
#layer 1

z_1 = tf.matmul(x, weight_1) + bias_1
batch_mean_1,batch_var_1 = tf.nn.moments(z_1,[0])
beta_1 = tf.Variable(tf.zeros([FIRST_LAYER_HIDDEN]))
scale_1 = tf.Variable(tf.ones([FIRST_LAYER_HIDDEN]))
BN_1 = tf.nn.batch_normalization(z_1, batch_mean_1, batch_var_1, beta_1, scale_1, BATCH_NORM_EPISILON)
h_fc1 = tf.nn.relu(z_1)

#layer 2
z_2 = tf.matmul(h_fc1, weight_2) + bias_2
batch_mean_2, batch_var_2 = tf.nn.moments(z_2, [0])
beta_2 = tf.Variable(tf.zeros([SECOND_LAYER_HIDDEN]))
scale_2 = tf.Variable(tf.ones([SECOND_LAYER_HIDDEN])) 
BN_2 = tf.nn.batch_normalization(z_2, batch_mean_2, batch_var_2, beta_2, scale_2, BATCH_NORM_EPISILON)
h_fc2 = tf.nn.relu(BN_2)

#h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
#y = tf.nn.softmax(tf.matmul(h_fc2_drop, weight_output) + bias_output)

#layer 3
z_3 = tf.matmul(h_fc2, weight_3) + bias_3
batch_mean_3, batch_var_3 = tf.nn.moments(z_3, [0])
beta_3 = tf.Variable(tf.zeros([THIRD_LAYER_HIDDEN]))
scale_3 = tf.Variable(tf.ones([THIRD_LAYER_HIDDEN])) 
BN_3 = tf.nn.batch_normalization(z_3, batch_mean_3, batch_var_3, beta_3, scale_3, BATCH_NORM_EPISILON)
h_fc3 = tf.nn.relu(BN_3)

#layer 4
z_4 = tf.matmul(h_fc3, weight_4) + bias_4
batch_mean_4, batch_var_4 = tf.nn.moments(z_4, [0])
beta_4 = tf.Variable(tf.zeros([FOURTH_LAYER_HIDDEN]))
scale_4 = tf.Variable(tf.ones([FOURTH_LAYER_HIDDEN])) 
BN_4 = tf.nn.batch_normalization(z_4, batch_mean_4, batch_var_4, beta_4, scale_4, BATCH_NORM_EPISILON)
h_fc4 = tf.nn.relu(BN_4)


h_fc4_drop = tf.nn.dropout(h_fc4, keep_prob)

y = tf.nn.softmax(tf.matmul(h_fc4_drop, weight_output) + bias_output)


cross_entropy = -tf.reduce_mean(y_*tf.log(y))
train_step = tf.train.AdagradOptimizer(LEARNING_RATE).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

predict = tf.argmax(y, 1)


# In[21]:

index_in_epoch = 0
epochs_completed =0
num_examples = train_x.shape[0]
def next_batch(batch_size):
    global train_x
    global train_y
    global index_in_epoch
    global epochs_completed
    
    start = index_in_epoch
    index_in_epoch += batch_size
    
    if index_in_epoch >= num_examples:
        epochs_completed += 1
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_x = train_x[perm]
        train_y = train_y[perm]
        
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_x[start:end], train_y[start:end]

init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)

DISPLAY_STEP = 500
loss_list = list()
valid_accu_list = list()
EARLY_STOPPING_FLAG = 0
#for i in range(1):
for i in range(TRAINING_ITERATIONS):
    batch_x, batch_y = next_batch(16)
    #print(i)
    if i == 0 or i % DISPLAY_STEP == 0:
        loss = cross_entropy.eval(feed_dict = {x:valid_x, y_:valid_y, keep_prob:1.0})
        valid_accu = accuracy.eval(feed_dict = {x:valid_x, y_:valid_y, keep_prob:1.0})
        loss_list.append(loss)
        valid_accu_list.append(valid_accu)
        '''if loss - loss_list[-1] == 0:
            EARLY_STOPPING_FLAG += 1
            if EARLY_STOPPING_FLAG == EARLY_STOPPING_ROUND:
                EARLY_STOPPING_FLAG = 0
                break'''
        print("At step {0}, loss=> {1:.4f}, validation accuracy => {2:.4f}".format(i, loss, valid_accu))
    else:
        sess.run(train_step,feed_dict = {x:batch_x, y_:batch_y, keep_prob:DROP_OUT})


# In[ ]:

test_data = pd.read_csv("test.csv")
test_x = test_data.iloc[:,1:].values

prediction = np.zeros([test_x.shape[0],OUTPUT_SIZE])
prediction.shape

for i in range(0, test_x.shape[0]//BATCH_SIZE + 1):
    prediction[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = y.eval(feed_dict={x:test_x[i*BATCH_SIZE:(i+1)*BATCH_SIZE],
                                                                 keep_prob:1.0})


submission = pd.read_csv("sample_submission.csv")
submission.shape

output_columns = submission.columns.values[1:]

output_data = pd.DataFrame(prediction, columns=list(output_columns))
output_data = pd.concat([submission.loc[:,['id']],output_data], axis=1)
# In[ ]:
train_time = time.strftime('%Y%m%d%H%M%S',time.localtime())
output_data.to_csv("nn_submission"+train_time+".csv", index=False,columns=output_data.columns.values)

