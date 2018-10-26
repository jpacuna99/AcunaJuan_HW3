import Image
import numpy as np
from scipy.fftpack import fft2, ifft2
from scipy import fftpack, ndimage
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

image = plt.imread("arbol(1).png") 



fourier2=fft2(image)
#freq = np.fft.fftfreq(len(arbol[0]))
plt.figure()
plt.imshow(np.log10(abs(fourier2)),plt.cm.gray)
plt.colorbar()

for i in range(np.shape(fourier2)[0]):
	for j in range(np.shape(fourier2)[1]):	
		if np.log10(abs(fourier2[i,j]))>3.613:
			fourier2[i,j]=0



plt.figure()
plt.imshow(np.log10(abs(fourier2)),plt.cm.gray)
plt.colorbar()
plt.figure()
arbol=ifft2(fourier2)

plt.imshow(abs(arbol),plt.cm.gray)




plt.show()
print arbol[0,0]












