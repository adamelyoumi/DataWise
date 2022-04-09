a = np.load("feature_cubes.npz")

items = list(a.items())

for p in items:
	print(p[0])
	print(p[1].shape)

feature_cubes
(8387, 20, 20, 4)
feature_grid
(8387, 400, 2)
feature_grid_qidx
(8387, 400)

for i in range(140,150):
    grid = items[0][1][i]
    layer = np.array([grid[:,:,k] for k in range(4)])
    #layer.shape # (20, 20)


    for k in range(4):
        plt.subplot(1,4,k+1)
        plt.imshow(layer[k], cmap='hot', interpolation='nearest')
        plt.title("i = "+str(i)+" ; k = "+str(k))
    plt.show()

for i in range(140,150):
    grid = items[0][1][i]
    plt.imshow(grid, cmap='hot', interpolation='nearest')
    plt.show()
