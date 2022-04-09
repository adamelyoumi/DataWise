### Generate Weather map ###

# Weather convective cells

import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import os

os.chdir("C:/Users/adame/OneDrive/Bureau/IRP doc")

h, w = 25,25

def generate_weather(h,w, plots = False):
    
    n_wcc = int((h+w)/5)
    cell_size = 0.85

    wcc = np.zeros((h,w))

    for _ in range(n_wcc):
        x, y = rnd.randint(h), rnd.randint(w)
        wcc[x][y] = 1
    for _ in range(10):
        for i in range(1,h-1):
            for j in range(1,w-1):
                if wcc[i][j] == 0 and (wcc[i+1][j] == 1 or wcc[i][j+1] == 1 or wcc[i-1][j] == 1 or wcc[i][j-1] == 1):
                    u = rnd.random()
                    if u > cell_size:
                        wcc[i][j] = 1
        
    # Wind map

    wind_v, wind_h = np.zeros((h,w)), np.zeros((h,w))
    wind_v[0][0], wind_h[0][0] = rnd.uniform(-1,1), rnd.uniform(-1,1)

    for i in range(1,h):
        wind_h[i][0] = wind_h[i-1][0] + rnd.uniform(-0.1,0.1)
        wind_v[i][0] = wind_v[i-1][0] + rnd.uniform(-0.1,0.1)

    for j in range(1,w):
        wind_h[0][j] = wind_h[0][j-1] + rnd.uniform(-0.1,0.1)
        wind_v[0][j] = wind_v[0][j-1] + rnd.uniform(-0.1,0.1)

    for i in range(1,h):
        for j in range(1,w):
            wind_h[i][j] = (wind_h[i][j-1] + wind_h[i-1][j])/2 + rnd.uniform(-0.1,0.1)
            wind_v[i][j] = (wind_v[i][j-1] + wind_v[i-1][j])/2 + rnd.uniform(-0.1,0.1)

    np.save("data_own/wcc", wcc)
    np.save("data_own/wind_horiz", wind_h)
    np.save("data_own/wind_vert", wind_v)

    # Plots

    if plots:

        plt.imshow(wcc)
        plt.show()

        plt.imshow(wind_h)
        plt.show()


        x, y = np.linspace(0,100,20), np.linspace(0,100,20)
        u, v = np.zeros((20,20)), np.zeros((20,20))

        for i in range(20):
            for j in range(20):
                u[i][j] = wind_h[i*5][j*5]
                v[i][j] = wind_v[i*5][j*5]

        plt.quiver(x,y,u,v)
        plt.show()
    


if __name__ == "__main__":
    generate_weather(h,w)
