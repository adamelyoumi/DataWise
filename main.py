import os
import numpy as np
import matplotlib.pyplot as plt
from nn import load_track_data, train_model, look_back, mod, summ
from weather_gen import generate_weather, h, w
from pandasql import sqldf
from pandas import read_csv

os.chdir("C:/Users/adame/OneDrive/Bureau/IRP doc")

############# "Preprocessing" phase #############
###### (doesn't correspond to real application) #######

def dist(lon1,lon2,lat1,lat2,alt1,alt2):
    # Function to calculate the distance in km based on coordinates and altitudes
    
    d_flat = 6378 * np.arccos((np.sin(lat1) * np.sin(lat2)) + np.cos(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)) # km
    d_h = alt1-alt2
    return(np.sqrt(d_flat * d_flat + d_h * d_h)) # Pythagoras for the final distance

def indexes(trajec_list):
    # Similarity index gathering similar trajectories
    # The differentiation is made on whether the latitude is within certain ranges
    
    index_list = []

    for traj in trajec_list:
        if traj[40][2] > 36.8:
            index_list.append(0)
        elif traj[40][2] > 34.5:
            index_list.append(1)
        else:
            index_list.append(2)

    return index_list

tracks = read_csv("C:/Users/adame/Documents/DeepTPmaster/src/DATA/flight_tracks.csv") #Real flight tracks (BOS to XXX)

# Get all distinct flight IDs
IDs = list(sqldf("SELECT DISTINCT FID FROM tracks WHERE FID IS NOT NULL")['FID'])
n_ids = len(IDs)

# List distinct trajectories
trajec_list = []
for ID in IDs:
    trajec_list.append(np.array(sqldf("SELECT FID AS ID, Lon, Lat, Alt, Speed FROM tracks WHERE FID = " + str(ID) )))

trajec_list = np.array(trajec_list)

# Number of columns without the ID
n_feat = trajec_list[0].shape[1]-1

# Generate training data from trajectory list
# trajectories are cut into segments of length "look_back", normalized (mu, sigma are mean and stddev) and shuffled
X_train, y_train, X_val, y_val, X_test, y_test, mu, sigma = load_track_data(trajec_list)

index_list = indexes(trajec_list)

wcc, wind_v, wind_h = [], [], []

# Generate as many different datasets as there are different indexes
for _ in range(3):
    generate_weather(h,w)
    wcc.append(np.load("data_own/wcc.npy"))
    wind_v.append(np.load("data_own/wind_vert.npy"))
    wind_h.append(np.load("data_own/wind_horiz.npy"))
    os.remove("data_own/wcc.npy")
    os.remove("data_own/wind_vert.npy")
    os.remove("data_own/wind_horiz.npy")

# Create weather datasets
wcc_train, wcc_val, wcc_test = [],[],[]
wind_v_train, wind_v_val, wind_v_test = [],[],[]
wind_h_train, wind_h_val, wind_h_test = [],[],[]

for u in X_train:
    idd = (u[0]*sigma+mu)[0]
    for p in range(len(IDs)):
        if IDs[p]==idd:
            wcc_train.append(wcc[index_list[p]])
            wind_v_train.append(wind_v[index_list[p]])
            wind_h_train.append(wind_h[index_list[p]])

for u in X_val:
    idd = (u[0]*sigma+mu)[0]
    for p in range(len(IDs)):
        if IDs[p]==idd:
            wcc_val.append(wcc[index_list[p]])
            wind_v_val.append(wind_v[index_list[p]])
            wind_h_val.append(wind_h[index_list[p]])

for u in X_test:
    idd = (u[0]*sigma+mu)[0]
    for p in range(len(IDs)):
        if IDs[p]==idd:
            wcc_test.append(wcc[index_list[p]])
            wind_v_test.append(wind_v[index_list[p]])
            wind_h_test.append(wind_h[index_list[p]])

wcc_train, wcc_val, wcc_test = np.array(wcc_train).reshape((-1,h,w,1)), np.array(wcc_val).reshape((-1,h,w,1)), np.array(wcc_test).reshape((-1,h,w,1))
wind_v_train, wind_v_val, wind_v_test = np.array(wind_v_train).reshape((-1,h,w,1)), np.array(wind_v_val).reshape((-1,h,w,1)), np.array(wind_v_test).reshape((-1,h,w,1))
wind_h_train, wind_h_val, wind_h_test = np.array(wind_h_train).reshape((-1,h,w,1)), np.array(wind_h_val).reshape((-1,h,w,1)), np.array(wind_h_test).reshape((-1,h,w,1)) 

# Remove IDs from features and constants
X_train = X_train[:,:,1:5]
X_val = X_val[:,:,1:5]
X_test = X_test[:,:,1:5]

mu = mu[1:5]
sigma = sigma[1:5]

# Initialize the model
epochs = 50
batch_size = 1000
model = mod(n_feat)

# Compile
summ(model)

hist = model.fit(
	x = [wind_v_train, X_train], y = y_train,
	validation_data=([wind_v_val, X_val], y_val),
	epochs = epochs, batch_size = batch_size)

model.save("models/model", save_format='h5')

# Plot a couple trajectories, observation and prediction
plot = True
if plot:
    u = mod.predict([wind_v_test, X_test])
    for p in range(4):
        plt.scatter(X_test[p][:,0],X_test[p][:,1], label = "traj "+str(p))
        plt.scatter(u[p][0],u[p][1], label = "pred " + str(p))
        plt.scatter(y_test[p][0], y_test[p][1], label = "obsv " + str(p))

    plt.legend()
    plt.show()

"""



# Create FL framework
## Partition data among aircrafts

############# Training phase #############

# Training using FL

############# Operational phase #############

penalties_n, penalties_s = [], []

traj_sample = trajec_list[0:10]

def is_above_threshold(d_coord, d_speed, th_coord = 0.5, th_speed = 0.05):
    return(d_coord > th_coord or d_speed > th_speed)

for traj in traj_sample:
    n = traj.shape[0]
    pen_normal = 0
    pen_syst = 0
    coords_actual, coords_pred = [], []

    for i in range(n):
        pen_normal += 1

        if i < look_back:
            pen_syst += 1
            coords_actual.append(traj[i])
            coords_pred.append(traj[i])

        else:
            last_n = coords_pred[i-look_back : i]
            pred = mod.predict(np.array(last_n).reshape((1,look_back,4)))
            pred = pred.reshape((4)) * sigma + mu
            d_coord = dist(pred[0], traj[i][0], pred[1], traj[i][1], pred[2], traj[i][2])
            d_speed = np.absolute(pred[3] - traj[i][3])

            if is_above_threshold(d_coord, d_speed, th_coord = 0.5, th_speed = 0.05):
                pen_syst += 1
                coords_actual.append(traj[i])
                coords_pred.append(traj[i])

            else:
                coords_actual.append(traj[i])
                coords_pred.append(pred)

    penalties_n.append(pen_normal)
    penalties_s.append(pen_syst)

"""
"""
Get first n data points

Predict next data point
"Wait dt"
Compare prediction to actual DP
If outside threshold, add penalty corresponding to data transferred & use actual DP for next pred
If within threshold, use prediction for next prediction

Repeat until end of flight

"""

# Compare final penalty to (nb of DP times individual penalty)

# Various plots
