import glob
import numpy as np
import pandas as pd


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2


file_list = glob.glob('expression_data/**/*.npy')

emotions = {'1': 'amusing', '2': 'amusing', '3': 'boring', '4': 'boring', '5': 'relaxed', '7': 'scary', '8': 'scary'}

def get_expression(file_path):
    return emotions[file_path.split('/')[-1].split('.')[0].split('_')[-1]]
def get_user_name(file_path):
    return file_path.split('/')[1]

df = pd.DataFrame(columns=['user', 'data', 'class'])


for file_path in file_list:
    expression, user = get_expression(file_path), get_user_name(file_path)
    if user == 'avijit' or user == 'amrta':
        continue
    data = np.load(file_path)[..., 24:82]
    counter = 0
    for _ in range(int(data.shape[0]/3)):
        row = {'user': user, 'class': expression, 'data': data[counter]}
        df.loc[len(df)] = row
        counter = counter + 1
        df['data'].apply(lambda x: np.abs(x)) 
    print(expression, user, file_path.split('/')[-1])

df['data'].apply(lambda x: np.abs(x))
print('Saving dataset')
df.to_pickle('processed.pkl')
# df = pd.read_pickle('processed.pkl')
print('datset saved')


def preprocess(df):
    X = df['data'].apply(lambda x: np.abs(x))
    y = df['class'].map(dict(zip(['amusing', 'boring', 'relaxed', 'scary', 'scray'],[0,1,2,3,3])))
    scaler = MinMaxScaler()
    X_scaled = pd.Series(X.apply(lambda x: scaler.fit_transform(x.reshape(-1, 1)).reshape(x.shape)))
    X_scaled = np.stack(X_scaled.values)
    X_scaled = X_scaled.transpose(0,1,3,2)
    
    return X_scaled, y


def get_model():
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(5, 3), activation='relu', padding='same', kernel_initializer='he_uniform', input_shape=(128,58,1)))
    model.add(MaxPooling2D(pool_size=(4, 1)))
    model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D(pool_size=(3, 2)))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=L2(l2=0.05)))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=L2(l2=0.05)))
    model.add(Dropout(0.1))
    model.add(Dense(4, activation='softmax'))
    return model


def show_results(model, history, X_test, y_test):
    score = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
    
    plt.plot(history.history['loss'], label='training')
    plt.plot(history.history['val_loss'], label='validation')
    plt.ylabel('Loss value')
    plt.xlabel('No. epoch')
    plt.legend()
    plt.savefig('loss.png')
    
    plt.plot(history.history['accuracy'], label='Accuracy (training data)')
    plt.plot(history.history['val_accuracy'], label='Accuracy (validation data)')
    plt.ylabel('Acuracy value')
    plt.xlabel('No. epoch')
    plt.legend()
    plt.savefig('accuracy.png')


def get_confusion_matrix(X_test, y_test, model):
    pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, np.argmax(pred, axis=1))
    class_report = classification_report(y_test, np.argmax(pred, axis=1))
    f1 = f1_score(y_test, np.argmax(pred, axis=1), average="weighted")
    result = "confusion matrix\n" + repr(conf_matrix) + "\n" + "report\n" + class_report + "\nf1_score(weighted)\n" + repr(f1)
    print(result)
    
    total = conf_matrix / conf_matrix.sum(axis=1).reshape(-1, 1)
    total = np.round(total, 2)
    labels = ['amusing', 'boring', 'relaxed', 'scary']
    df_cm = pd.DataFrame(total, index=[i for i in labels], columns=[i for i in labels])
    sns.heatmap(df_cm, vmin=0, vmax=1, annot=True, cmap="Blues")
    plt.savefig('confusion_matrix.png')


X_train, y_train = preprocess(df)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.33, random_state=42)

model = get_model()

best_save=tf.keras.callbacks.ModelCheckpoint(filepath='all_user_model.h5',save_weights_only=True, monitor='val_accuracy',mode='max',save_best_only=True)
model.compile(loss='sparse_categorical_crossentropy',
             optimizer=Adam(lr=0.001),
             metrics=['accuracy'])

history = model.fit(X_train,y_train,batch_size=256,epochs=100, validation_split=0.2, callbacks=[best_save])

show_results(model, history, X_test, y_test)
# model.load_weights('all_user_model.h5')
get_confusion_matrix(X_test, y_test, model)

print('Model Saved')
model.save_weights("all_user_model.h5")
