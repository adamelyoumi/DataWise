import numpy as np

from keras.models import Model
from keras.models import Sequential
from keras.layers import LSTM, Reshape, Dense, Conv2D, MaxPooling2D, Flatten, Normalization, Input
from keras.layers.merge import concatenate
from keras.optimizers import Adam

import matplotlib.pyplot as plt
from pandas import read_csv
from pandasql import sqldf
from sklearn.model_selection import train_test_split

from weather_gen import w,h

import warnings
warnings.filterwarnings("ignore")

## Final, ideal model, which uses weather data

look_back = 30

def mod(n_feat, look_back = look_back):

    # weather encoder
    inp_w = Input(shape=(h,w,1))
    conv1 = Conv2D(filters = 16, kernel_size = (3,3), strides = 1)(inp_w)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(filters = 32, kernel_size = (3,3), strides = 1)(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    #conv3 = Conv2D(filters = 32, kernel_size = (3,3), strides = 1)(pool2)
    #pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    flat = Flatten()(pool2)

    # trajectory encoder
    inp_traj = Input(shape=(look_back,n_feat)) # (nb_sequence, nb_timestep, nb_feature)
    lstm1 = LSTM(100, return_sequences = True)(inp_traj)
    lstm2 = LSTM(50, return_sequences = True)(lstm1)
    lstm3 = LSTM(20, return_sequences = True)(lstm2)
    lstm4 = LSTM(20)(lstm3)

    #merge
    merge = concatenate([flat, lstm4])

    #decoder
    dense1 = Dense(20)(merge)
    dense2 = Dense(10)(dense1)
    dense3 = Dense(10)(dense2)
    dense4 = Dense(n_feat)(dense3)

    model = Model(inputs = [inp_w,inp_traj], outputs = dense4)

    return(model)


def rnn(n_feat):
    model = Sequential()
    model.add(LSTM(100, input_shape=(look_back,n_feat), return_sequences = True)) # (nb_sequence, nb_timestep, nb_feature)
    model.add(LSTM(75, return_sequences = True))
    model.add(LSTM(50))
    model.add(Dense(50))
    model.add(Dense(20))
    model.add(Dense(4))
    return(model)


def summ(model, s = True):
    opt = Adam(learning_rate=0.05, beta_1=0.9, beta_2=0.999, epsilon=1e-03)
    model.compile(loss='mean_squared_error', optimizer=opt)

    if s:
        model.summary()

    return

"""
def f(x):
    return(x + np.sin(x))

def data_f(n,n_t):
    dataset = []
    for i in range(n):
        x0 = rnd.uniform(-3,3)
        l = [f(x0)]
        for p in range(1,n_t):
            q = p/10
            l.append(f(p/10 + x0))
        dataset.append(l)
    return(np.array(dataset))
"""

def create_dataset(dataset, look_back = look_back):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        dataX.append(dataset[i : (i + look_back), :])
        dataY.append(dataset[i + look_back, :])
    return (np.array(dataX), np.array(dataY))


def load_track_data(trajec_list):

    X, y = [], []
    for p in range(trajec_list.shape[0]):
        dx,dy = create_dataset(trajec_list[p])
        for r in dx:
            X.append(r)
        for r in dy:
            y.append(r)

    n_feat = trajec_list[0].shape[1]
    X = np.array(X)
    y = np.array(y).reshape((-1,1,n_feat))

    mu = np.mean(X, axis = (0,1))
    sigma = np.sqrt(np.var(X, axis = (0,1)))
    X = (X-mu)/sigma
    y = (y-mu)/sigma
    
    X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size = 0.1)
    X_train, X_val, y_train, y_val = train_test_split(np.array(X_train), np.array(y_train), test_size = 0.1)

    y_train = y_train.reshape(-1,n_feat)
    y_val = y_val.reshape(-1,n_feat)
    y_test = y_test.reshape(-1,n_feat)

    return(X_train, y_train, X_val, y_val, X_test, y_test, mu, sigma)

def train_model(X_train, y_train, X_val, y_val, X_test, y_test, index_list, epochs = 40, batch_size = 50, plots = False):

    n_feat = X_train.shape[2]
    model = mod(n_feat)

    summ(model)
    
    # Chercher données adaptées ie données récentes
    
    hist = mod.fit(X_train, y_train, epochs = epochs, validation_data=(X_val,y_val), batch_size = batch_size)

    if plots:
        plt.plot(range(epochs), hist.history['loss'], label = "train_loss")
        plt.plot(range(epochs), hist.history['val_loss'], label = "val_loss")
        plt.legend()
        plt.show()

        u = mod.predict([wind_v_test, X_test])
        for p in range(4):
            plt.scatter(X_test[p][:,0],X_test[p][:,1], label = "trajec "+str(p))
            plt.scatter(u[p][0],u[p][1], label = "pred " + str(p))
            plt.scatter(y_test[p][0], y_test[p][1], label = "obs " + str(p))

    plt.legend()
    plt.show()

    return(mod)

    #plt.plot(range(n_t),X[-1])
    #plt.show()

    #pred = a.predict(X_test)

    #print(pred, y_test)

"""
for p in range(10):
    plt.scatter(trajec_list[p][:,1], trajec_list[p][:,0], trajec_list[p][:,2])

plt.show()


for p in range(5):
    for q in range(5):
        plt.scatter(X_train[10*p+q][:,1],X_train[10*p+q][:,0], label = str(q))
        plt.scatter(y_train[10*p+q][:,1], y_train[10*p+q][:,0], label = "obs " + str(q))
    plt.legend()
    plt.show()

"""
def col(q):
    a = ["","","red","blue","green","yellow","orange","pink"]
    return(a[q//10])

if __name__ == "__main__":
    tracks = read_csv("C:/Users/adame/Documents/DeepTPmaster/src/DATA/flight_tracks.csv")

    IDs = list(sqldf("SELECT DISTINCT FID FROM tracks WHERE FID IS NOT NULL")['FID'])
    n_ids = len(IDs)

    trajec_list = []
    for ID in IDs:
        trajec_list.append(np.array(sqldf("SELECT FID AS ID, Lon, Lat, Alt, Speed FROM tracks WHERE FID = " + str(ID) )))

    trajec_list = np.array(trajec_list)

    n_feat = trajec_list[0].shape[1]-1
    
    X_train, y_train, X_val, y_val, X_test, y_test, mu, sigma = load_track_data(trajec_list)

    plot_split = False
    
    if plot_split:
        for q in [20,30,40,50,60,70]:
        
            for i in range(50):
        
                traj = trajec_list[i]
                plt.scatter(traj[q,0], traj[q,1], marker = "+", color = col(q))
        
        plt.legend()
        plt.show()

    #train_model(X_train, y_train, X_val, y_val, X_test, y_test, plots=True)
