from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import coremltools
from scipy import stats

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing
from keras.layers import Bidirectional
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, Conv1D, LSTM, MaxPooling1D, GlobalAveragePooling1D
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
import numpy
from keras.datasets import imdb
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

# Set some standard parameters upfront
pd.options.display.float_format = '{:.1f}'.format
sns.set() # Default seaborn look and feel
plt.style.use('ggplot')
print('keras version ', keras.__version__)
# Same labels will be reused throughout the program

# LABELS = ['1.0', '1.1', '1.2', '1.3', '1.4', '1.5'] # private  dataset 내꺼
# LABELS = ['Downstair', 'Upstair', 'Left', 'Right', 'Standing', 'Walking', 'Upelve', 'Downelve','U', 'Sitting'] # private  dataset 내꺼
LABELS = ['Downstair', 'Left', 'Right', 'Upstair', 'Standing', 'Walking'] # private  dataset 내꺼


# The number of steps within one time segment
TIME_PERIODS = 25
# The steps to take from one segment to the next; if this value is equal to
# TIME_PERIODS, then there is no overlap between the segments
STEP_DISTANCE = 10


def read_data(file_path):

    column_names = ['user-id', 'activity',
                    'x-axis-accel',
                    'y-axis-accel',
                    'z-axis-accel',
                    'x-axis-gyro',
                    'y-axis-gyro',
                    'z-axis-gyro',
                    'timestamp'
                    ]
    df = pd.read_csv(file_path,
                     header=None,
                     names=column_names)
    # Last column has a ";" character which must be removed ...
    df['timestamp'].replace(regex=True,
                         inplace=True,
                         to_replace=r';',
                         value=r'')
    # ... and then this column must be transformed to float explicitly
    df['timestamp'] = df['timestamp'].apply(convert_to_float)
    # This is very important otherwise the model will not fit and loss
    # will show up as NAN
    df.dropna(axis=0, how='any', inplace=True)

    return df


def convert_to_float(x):
    try:
        return np.float(x)
    except:
        return np.nan


def show_basic_dataframe_info(dataframe):
    # Shape and how many rows and columns
    print('Number of columns in the dataframe: %i' % (dataframe.shape[1]))
    print('Number of rows in the dataframe: %i\n' % (dataframe.shape[0]))


# Load data set containing all the data from csv
df = read_data('activity_6_axis.csv')

df['activity'].value_counts().plot(kind='bar', color=[plt.cm.Paired(np.arange(len(df)))],title='Training Examples by activity Type')
# plt.show()
df['user-id'].value_counts().plot(kind='bar', color=[plt.cm.Paired(np.arange(len(df)))],title='Training Examples by User')
# plt.show()

# Show how many training examples exist for each of the six activities
# df['activity'].value_counts().plot(kind='bar',title='Training Examples by Activity Type')
# plt.show()
# Better understand how the recordings are spread across the different
# users who participated in the study
# df['user-id'].value_counts().plot(kind='bar', title='Training Examples by User')
# plt.show()

def plot_activity(activity, data):

    fig, (ax0, ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=3,
         figsize=(15, 10),
         sharex=True)
    plot_axis(ax0, data['timestamp'], data['x-axis-accel'], 'X-Axis')
    plot_axis(ax1, data['timestamp'], data['y-axis-accel'], 'Y-Axis')
    plot_axis(ax2, data['timestamp'], data['z-axis-accel'], 'Z-Axis')
    plot_axis(ax0, data['timestamp'], data['x-axis-gyro'], 'X-Axis-gyro')
    plot_axis(ax1, data['timestamp'], data['y-axis-gyro'], 'Y-Axis-gyro')
    plot_axis(ax2, data['timestamp'], data['z-axis-gyro'], 'Z-Axis-gyro')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.90)
    # plt.show()

def plot_axis(ax, x, y, title):

    ax.plot(x, y, 'r')
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)

for activity in np.unique(df['activity']):
    subset = df[df['activity'] == activity][:40]
    plot_activity(activity, subset)



# Define column name of the label vector
LABEL = 'ActivityEncoded'
# Transform the labels from String to Integer via LabelEncoder
le = preprocessing.LabelEncoder()
# Add a new column to the existing DataFrame with the encoded values
df[LABEL] = le.fit_transform(df['activity'].values.ravel())

# Better understand how the recordings are spread across the different
# users who participated in the study

df_test = df[df['user-id'] > 1]
df_train = df[df['user-id'] <= 1]

# Normalize features for training data set (values between 0 and 1)
# Surpress warning for next 3 operation
pd.options.mode.chained_assignment = None  # default='warn'
df_train['x-axis-accel'] = df_train['x-axis-accel'] / df_train['x-axis-accel'].max()
df_train['y-axis-accel'] = df_train['y-axis-accel'] / df_train['y-axis-accel'].max()
df_train['z-axis-accel'] = df_train['z-axis-accel'] / df_train['z-axis-accel'].max()
df_train['x-axis-gyro'] = df_train['x-axis-gyro'] / df_train['x-axis-gyro'].max()
df_train['y-axis-gyro'] = df_train['y-axis-gyro'] / df_train['y-axis-gyro'].max()
df_train['z-axis-gyro'] = df_train['z-axis-gyro'] / df_train['z-axis-gyro'].max()

df_test['x-axis-accel'] = df_test['x-axis-accel'] / df_test['x-axis-accel'].max()
df_test['y-axis-accel'] = df_test['y-axis-accel'] / df_test['y-axis-accel'].max()
df_test['z-axis-accel'] = df_test['z-axis-accel'] / df_test['z-axis-accel'].max()
df_test['x-axis-gyro'] = df_test['x-axis-gyro'] / df_test['x-axis-gyro'].max()
df_test['y-axis-gyro'] = df_test['y-axis-gyro'] / df_test['y-axis-gyro'].max()
df_test['z-axis-gyro'] = df_test['z-axis-gyro'] / df_test['z-axis-gyro'].max()

#데이터 값이 소수점으로 인하여 일정치않아 소수점 4번째까지 끊어줌 6번까지 하면 더 좋긴한데 데이터가 더 많아야되서 못하였습니다.
data_test = df_test.round(
    # {'x-axis-accel': 4, 'y-axis-accel': 4, 'z-axis-accel': 4, 'total-axis-accel':4,'x-axis-gyro': 4, 'y-axis-gyro': 4, 'z-axis-gyro': 4})
    {'x-axis-accel': 4, 'y-axis-accel': 4, 'z-axis-accel': 4,'x-axis-gyro': 4, 'y-axis-gyro': 4, 'z-axis-gyro': 4})
# Round numbers
data_train = df_train.round(
    # {'x-axis-accel': 4, 'y-axis-accel': 4, 'z-axis-accel': 4, 'total-axis-accel':4,'x-axis-gyro': 4, 'y-axis-gyro': 4, 'z-axis-gyro': 4})
    {'x-axis-accel': 4, 'y-axis-accel': 4, 'z-axis-accel': 4,'x-axis-gyro': 4, 'y-axis-gyro': 4, 'z-axis-gyro': 4})

def create_segments_and_labels(df, time_steps, step, label_name):

    # x, y, z acceleration as features
    N_FEATURES = 6 # 뽑을 픽쳐수
    # Number of steps to advance in each iteration (for me, it should always
    # be equal to the time_steps in order to have no overlap between segments)
    # step = time_steps
    segments = []
    labels = []
    for i in range(0, len(df) - time_steps, step):
        x = df['x-axis-accel'].values[i: i + time_steps]
        # x = butter_lowpass_filter(x, 5 , 2) #x = 데이터, 5는 cutoff, 2는 주파수(데이터 센싱 0.5초 주기로 센싱)
        y = df['y-axis-accel'].values[i: i + time_steps]
        # y = butter_lowpass_filter(y, 5, 2)
        z = df['z-axis-accel'].values[i: i + time_steps]
        # total = df['total-axis-accel'].values[i: i + time_steps]
        x2 = df['x-axis-gyro'].values[i: i + time_steps]
        y2 = df['y-axis-gyro'].values[i: i + time_steps]
        z2 = df['z-axis-gyro'].values[i: i + time_steps]
        # Retrieve the most often used label in this segment
        label = stats.mode(df[label_name][i: i + time_steps])[0][0]
        segments.append([x, y, z, x2, y2, z2])
        labels.append(label)

    # Bring the segments into a better shape
    reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, time_steps, N_FEATURES)
    labels = np.asarray(labels)

    return reshaped_segments, labels

x_train, y_train = create_segments_and_labels(df_train,TIME_PERIODS,STEP_DISTANCE,LABEL)
x_test, y_test = create_segments_and_labels(df_test,TIME_PERIODS,STEP_DISTANCE,LABEL)

print('x_train shape: ', x_train.shape)
print(x_train.shape[0], 'training samples')
print('y_train shape: ', y_train.shape)

# Set input & output dimensions
num_time_periods, num_sensors = x_train.shape[1], x_train.shape[2]
num_time_periods_test, num_sensors_test = x_test.shape[1], x_test.shape[2]
num_classes = le.classes_.size
print(list(le.classes_))

input_shape = (num_time_periods*num_sensors)
input_shape_test = (num_time_periods_test*num_sensors_test)
x_train = x_train.reshape(x_train.shape[0], input_shape)
x_test= x_test.reshape(x_test.shape[0], input_shape_test)
print('x_train shape:', x_train.shape)
print('input_shape:', input_shape)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

y_train_hot = np_utils.to_categorical(y_train, num_classes)
y_test_hot = np_utils.to_categorical(y_test, num_classes)
print('New y_train shape: ', y_train_hot.shape)


model_m = Sequential()
# Remark: since coreml cannot accept vector shapes of complex shape like
# [80,3] this workaround is used in order to reshape the vector internally
# prior feeding it into the network
model_m.add(Reshape((TIME_PERIODS, 6), input_shape=(input_shape,)))
model_m.add(Conv1D(48, 6, activation='relu', input_shape=(TIME_PERIODS, num_sensors)))
model_m.add(MaxPooling1D(pool_size=3))
model_m.add(Conv1D(24, 2, activation='relu'))
model_m.add(MaxPooling1D(pool_size=3))
model_m.add(Bidirectional(LSTM(256, return_sequences=True)))
model_m.add(Bidirectional(LSTM(128, return_sequences=True)))
model_m.add(Dense(100, activation='relu'))
model_m.add(Flatten())
model_m.add(Dense(num_classes, activation='softmax'))
print(model_m.summary())


callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss', save_best_only=True),
    keras.callbacks.EarlyStopping(monitor='acc', patience=1)
]

model_m.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])

# Hyper-parameters
BATCH_SIZE = 100
EPOCHS = 70

# y_pred_train = model_m.predict(x_train)
# # Take the class with the highest probability from the train predictions
# max_y_pred_train = np.argmax(y_pred_train, axis=1)
# max_y_train = np.argmax(y_train)


# Enable validation to use ModelCheckpoint and EarlyStopping callbacks.
history = model_m.fit(x_train,
                      y_train_hot,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      # callbacks=callbacks_list,
                      validation_split=0.2,
                      verbose=1)


plt.figure(figsize=(6, 4))
plt.plot(history.history['acc'], 'r', label='Accuracy of training data')
plt.plot(history.history['val_acc'], 'b', label='Accuracy of validation data')
plt.plot(history.history['loss'], 'r--', label='Loss of training data')
plt.plot(history.history['val_loss'], 'b--', label='Loss of validation data')
plt.title('Model Accuracy and Loss')
plt.ylabel('Accuracy and Loss')
plt.xlabel('Training Epoch')
plt.ylim(0)
plt.legend()
# plt.show()
def show_confusion_matrix(validations, predictions):

    matrix = metrics.confusion_matrix(validations, predictions)
    plt.figure(figsize=(6, 4))
    print("train matrix", matrix)
    sns.heatmap(matrix,
                cmap='coolwarm',
                xticklabels=LABELS,
                yticklabels=LABELS,
                annot=True,
                fmt='d')
    plt.title('Train Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

y_pred_train = model_m.predict(x_train)
# Take the class with the highest probability from the test predictions
max_y_pred_train = np.argmax(y_pred_train, axis=1)
max_y_train = np.argmax(y_train_hot, axis=1)

show_confusion_matrix(max_y_train, max_y_pred_train)
print(classification_report(max_y_train, max_y_pred_train))

scores1 = model_m.evaluate(x_train, y_train_hot, verbose=0)
print("train Accuracy : ", scores1[1]*100)


def show_confusion_matrix2(validations, predictions):

    matrix = metrics.confusion_matrix(validations, predictions)
    print("Test matrix", matrix)
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix,
                cmap='coolwarm',
                xticklabels=LABELS,
                yticklabels=LABELS,
                annot=True,
                fmt='d')
    plt.title('Test Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


y_pred_test = model_m.predict(x_test)
# Take the class with the highest probability from the test predictions
max_y_pred_test = np.argmax(y_pred_test, axis=1)
max_y_test = np.argmax(y_test_hot, axis=1)

show_confusion_matrix2(max_y_test, max_y_pred_test)

print(classification_report(max_y_test, max_y_pred_test))
scores = model_m.evaluate(x_test, y_test_hot, verbose=0)
print("Accuracy : ", scores[1]*100)


# keras.clear_session ()