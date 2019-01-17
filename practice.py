# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 23:45:02 2018

@author: user
"""


# static_rnn : 정해진 사이즈로 입장 :padding, dynaminc_rnn
# output 과 state를 만듬
# state.h(출력해서 나가는 것), c(옆으로 가는 status)
outputs, state = tf.nn.dynamic_rnn(cell = lstm_cell, dtype = tf.float32, inputs = values)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    output_run, state_run = sess.run([outputs, state])
    print(output_run.shape)
    print(output_run)


import numpy as np
import pandas as pd
X_pca
pd.DataFrame(X_pca)
pd.DataFrame(X_scaled)

def train_classifier(self, argcands):
        # Extract the necessary features from the argument candidates
        train_argcands_feats = []
        train_argcands_target = []

        for argcand in argcands:
            train_argcands_feats.append(self.extract_features(argcand))
            train_argcands_target.append(argcand["info"]["label"])

        # Transform the features to the format required by the classifier
        self.feat_vectorizer = DictVectorizer()
        train_argcands_feats = self.feat_vectorizer.fit_transform(train_argcands_feats)

        # Transform the target labels to the format required by the classifier
        self.target_names = list(set(train_argcands_target))
        train_argcands_target = [self.target_names.index(target) for target in train_argcands_target]

        # Train the appropriate supervised model
        self.classifier = LinearSVC()
        #self.classifier = SVC(kernel="poly", degree=2)

        self.classifier.fit(train_argcands_feats,train_argcands_target)

        return

def execute(self, argcands_test):
        # Extract features
        test_argcands_feats = [self.extract_features(argcand) for argcand in argcands_test]

        # Transform the features to the format required by the classifier
        test_argcands_feats = self.feat_vectorizer.transform(test_argcands_feats)

        # Classify the candidate arguments
        test_argcands_targets = self.classifier.predict(test_argcands_feats)

        # Get the correct label names
        test_argcands_labels = [self.target_names[int(label_index)] for label_index in test_argcands_targets]

        return zip(argcands_test, test_argcands_labels)





import sys
import numpy as np
import tensorflow as tf
from datetime import datetime

device_name = "/gpu:0"
shape=[3,2]
with tf.device(device_name):
    random_matrix = tf.random_uniform(shape=shape, minval=0, maxval=1)
    dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix))
    sum_operation = tf.reduce_sum(dot_operation)


startTime = datetime.now()
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
        result = session.run(sum_operation)
        print(result)


print("\n" * 5)
print("Shape:", shape, "Device:", device_name)
print("Time taken:", datetime.now() - startTime)
print("\n" * 5)

tf.test.is_gpu_available()




########
RNN

import tensorflow as tf
import numpy as np

unique = ['h', 'e', 'l', 'o']
ydata = [1, 2, 2, 3]  # ello
xdata = [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]]

xdata = np.array(xdata, dtype = np.float32)

cells = tf.nn.rnn_cell.BasicRNNCell(4)
state = tf.zeros([1, cells.state_size])
xdata = tf.split(0, 4, xdata)

print(*xdata, sep = '\n') # [[1, 0, 0, 0]]

output, state = tf.nn.rnn(cells, xdata, state)
print(*output, sep = '\n')



###

import sys
import os
from keras.layers import *
from keras.optimizers import *
from keras.applications import *
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as k

# fix seed for reproducible results (only works on CPU, not GPU)
seed = 9
np.random.seed(seed=seed)
tf.set_random_seed(seed=seed)

# hyper parameters for model
nb_classes = 2  # number of classes
based_model_last_block_layer_number = 126  # value is based on based model selected.
img_width, img_height = 299, 299  # change based on the shape/structure of your images
batch_size = 32  # try 4, 8, 16, 32, 64, 128, 256 dependent on CPU/GPU memory capacity (powers of 2 values).
nb_epoch = 50  # number of iteration the algorithm gets trained.
learn_rate = 1e-4  # sgd learning rate
momentum = .9  # sgd momentum to avoid local minimum
transformation_ratio = .05  # how aggressive will be the data augmentation/transformation


def train(train_data_dir, validation_data_dir, model_path):
    # Pre-Trained CNN Model using imagenet dataset for pre-trained weights
    base_model = Xception(input_shape=(img_width, img_height, 3), weights='imagenet', include_top=False)

    # Top Model Block
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(nb_classes, activation='softmax')(x)

    # add your top layer block to your base model
    model = Model(base_model.input, predictions)
    print(model.summary())

    # # let's visualize layer names and layer indices to see how many layers/blocks to re-train
    # # uncomment when choosing based_model_last_block_layer
    # for i, layer in enumerate(model.layers):
    #     print(i, layer.name)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all layers of the based model that is already pre-trained.
    for layer in base_model.layers:
        layer.trainable = False

    # Read Data and Augment it: Make sure to select augmentations that are appropriate to your images.
    # To save augmentations un-comment save lines and add to your flow parameters.
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=transformation_ratio, #회전
                                       shear_range=transformation_ratio, # 절단
                                       zoom_range=transformation_ratio, #줌
                                       cval=transformation_ratio,
                                       horizontal_flip=True,
                                       vertical_flip=True)

    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    os.makedirs(os.path.join(os.path.abspath(train_data_dir), '../preview'), exist_ok=True)
    train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                        target_size=(img_width, img_height),
                                                        batch_size=batch_size,
                                                        class_mode='categorical')
    # save_to_dir=os.path.join(os.path.abspath(train_data_dir), '../preview')
    # save_prefix='aug',
    # save_format='jpeg')
    # use the above 3 commented lines if you want to save and look at how the data augmentations look like

    validation_generator = validation_datagen.flow_from_directory(validation_data_dir,
                                                                  target_size=(img_width, img_height),
                                                                  batch_size=batch_size,
                                                                  class_mode='categorical')

    model.compile(optimizer='nadam',
                  loss='categorical_crossentropy',  # categorical_crossentropy if multi-class classifier
                  metrics=['accuracy'])

    # save weights of best training epoch: monitor either val_loss or val_acc

    top_weights_path = os.path.join(os.path.abspath(model_path), 'top_model_weights.h5')
    callbacks_list = [
        ModelCheckpoint(top_weights_path, monitor='val_acc', verbose=1, save_best_only=True),
        EarlyStopping(monitor='val_acc', patience=5, verbose=0)
    ]

    # Train Simple CNN
    model.fit_generator(train_generator,
                        samples_per_epoch=12500/32,
                        nb_epoch=nb_epoch / 5,
                        validation_data=validation_generator,
                        nb_val_samples=(16*63+8)/32,
                        callbacks=callbacks_list)


    print("\nStarting to Fine Tune Model\n")


    model.load_weights(top_weights_path)

    for layer in model.layers[:based_model_last_block_layer_number]:
        layer.trainable = False #Xception의 훈련에서 제외되어야 할 layer

    for layer in model.layers[based_model_last_block_layer_number:]:
        layer.trainable = True


    model.compile(optimizer='nadam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])


    final_weights_path = os.path.join(os.path.abspath(model_path), 'model_weights.h5')
    callbacks_list = [
        ModelCheckpoint(final_weights_path, monitor='val_acc', verbose=1, save_best_only=True),
        EarlyStopping(monitor='val_loss', patience=5, verbose=0)
    ]


    model.fit_generator(train_generator,
                        samples_per_epoch=12500/32,
                        nb_epoch=nb_epoch,
                        validation_data=validation_generator,
                        nb_val_samples=(16*63+8)/32,
                        callbacks=callbacks_list)



    model_json = model.to_json()
    with open(os.path.join(os.path.abspath(model_path), 'model.json'), 'w') as json_file:
        json_file.write(model_json)


data_dir = os.path.abspath("./data/")
train_dir = os.path.join(os.path.abspath(data_dir), 'train')
# Inside, each class should have it's own folder
validation_dir = os.path.join(os.path.abspath(data_dir), 'valid')
# each class should have it's own folder
model_dir = os.path.abspath("./dogscats/model")

os.makedirs(os.path.join(os.path.abspath(data_dir), 'preview'), exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

train(train_dir, validation_dir, model_dir)  # train model


k.clear_session()





과제 1 ~ 3 중 하나 또는 여러개를 선택해서 수행 계획표를 작성
    선택을 할때는 예시 데이터를 참조

수행 계획표를 통과해야 실제 데이터를 받고 분석 참여가능
    예상 경쟁률 2:1, 3:1

통과를 하게 될시 중간에 추가 과제가 있음
이 추가 과제가 메인 포인트

마지막 발표는 울산에서 하게 되며
총 12팀이 상을 받게됨
    최우수 2팀
    우수 2팀
    장려 4팀
    입상 4팀

과제 1
  녹스 속스는 운전자 예측 상태와 연관
  탄소 4개를 섞어서 예측 값
  최적의 혼탄 조합을 찾는 것

  정상적으로 운영되는 데이터
  즉, 이상치가 포함되어있다는 뜻

  가용 석탄의 최적 조합 찾기 시뮬레이션 알고리즘 도출

 과제 2


from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import statsmodels.api as sm

data = load_boston()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

def stepwise_selection(X, y,
                       initial_list=[],
                       threshold_in=0.01,
                       threshold_out = 0.05,
                       verbose=True):
    """ Perform a forward-backward feature selection
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.argmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included

result = stepwise_selection(X, y)

print('resulting features:')
print(result)


#########################

sb = read_csv("subject.csv", engine='python')

# change name of column
soot.rename(columns={'수연비\nFW NCIC32B ZQ03  %': 'y'}, inplace=True)

# setting x, y
X = soot.drop(['y'], axis = 1)
y = pd.DataFrame(soot['y'])



#######################################
2019.01.15
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import statsmodels.api as sm

def stepwise_selection(X, y,
                       initial_list=[],
                       threshold_in=0.01,
                       threshold_out = 0.05,
                       verbose=True):
    """ Perform a forward-backward feature selection
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.argmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included, print("model:", model, "model.pvalues", model.pvalues.iloc[1:])


# loading data
sb = pd.read_csv("subject.csv", engine='python')

# changing name of column Y
sb.rename(columns={'수연비\nFW NCIC32B ZQ03  %': 'y'}, inplace=True)

# setting X, y
X = sb.drop(['date-time', 'y'], axis = 1)
y = pd.DataFrame(sb['y'])

# setting parallel working
from dask.distributed import Client
# before you start this, you need to put "dask-scheduler" in prompt
client = Client('192.168.1.23:8786')
client.restart()
client.start()

total = client.submit(stepwise_selection, X, y)

total.result()
total.gather()

















##
