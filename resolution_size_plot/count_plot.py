import csv
import numpy as np
import matplotlib as plt
import scipy.stats as stats
import pylab as pl

# file = 'E:/vid/vidtimit_mtcnn_crop.csv'
file = 'E:/ff++/face2face_original.csv'
f = csv.reader(open(file, 'rb'))
height = []
width = []
for row in f:
    if row[0] == 'name':
        continue
    height.append(int(row[1]))
    width.append(int(row[2]))
heigth = np.array(height)
wifth = np.array(width)

fit = stats.norm.pdf(heigth, np.mean(heigth), np.std(heigth))  # this is a fitting indeed
# pl.plot(heigth, fit, linewidth=0.1, color='blue')
pl.hist(heigth, normed=True)  # use this to draw histogram of your data

nfit = stats.norm.pdf(wifth, np.mean(wifth), np.std(wifth))
# pl.plot(wifth, nfit, linewidth=0.1, color='red')
pl.hist(wifth, normed=True)

leg = []
leg.append('height')
leg.append('width')
pl.legend(leg)

# pl.xlim([-1, 1])
# pl.ylim([0.0, 1])
pl.xlabel('size', fontsize=15)
pl.ylabel('numbers', fontsize=15)
pl.title('ff++_mtcnn_crop', fontsize=20) #and impostor   genuine
pl.grid()
pl.show()


# plt.plot(height)