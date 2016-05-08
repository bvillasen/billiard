import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5

readCuda = False

for option in sys.argv:
  if option == "cuda": readCuda = True

if readCuda: dataFile = h5.File("data_billard_cuda.h5", 'r')
else: dataFile = h5.File("data_billard_julia.h5", 'r')

posData = dataFile['pos_snap'][...]
dim, nParticles, nSteps = posData.shape
dataFile.close()

pos_x_all = posData[0,:,:].T
pos_y_all = posData[1,:,:].T
pos_x_all = pos_x_all[::-1].reshape( nParticles * nSteps )
pos_y_all = pos_y_all[::-1].reshape( nParticles * nSteps )

colors = np.array([ i*np.ones(nParticles) for i in range(nSteps)])
colors.reshape( nParticles * nSteps )

if readCuda: n = 0
else: n = 1

fig = plt.figure(n)
fig.clf()
ax = fig.add_subplot(111)
ax.scatter( pos_x_all, pos_y_all, marker=',', s=3, c=colors, edgecolors='none', cmap='gnuplot_r' )
# ax.colorbar()
ax.set_aspect('equal', 'datalim')
fig.show()




# plt.clf()
# for pId in range(nParticles):
#   pos_all = posData[:,pId,:]
#   plt.plot(pos_all[0], pos_all[1])
# plt.axes().set_aspect('equal', 'datalim')
# # plt.axes().set_xlim(5, 5)
# # plt.axes().set_ylim(-5, 5)
# plt.show()
