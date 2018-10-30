import Image
import numpy as np
from scipy.fftpack import fft2, ifft2
from scipy import fftpack, ndimage
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

image = plt.imread("arbol(1).png") 



fourier2=fft2(image)
freq = np.fft.fftfreq(len(fourier2[0]))
freq1 = np.fft.fftfreq(len(fourier2[1]))




plt.figure()
plt.imshow(np.log10(abs(np.fft.fftshift(fourier2))),plt.cm.gray)
plt.colorbar()


filtro=np.copy(fourier2) 
for i in range(np.shape(fourier2)[0]):
	for j in range(np.shape(fourier2)[1]):	
		if abs(filtro[i,j])>4100 and abs(fourier2[i,j])<4600 :
			filtro[i,j]=10**-10



plt.figure()
plt.imshow(np.log10(abs(np.fft.fftshift(filtro))),plt.cm.gray)
plt.colorbar()

plt.figure()
arbol=ifft2(filtro)

plt.imshow(abs(arbol),plt.cm.gray)




plt.show()












