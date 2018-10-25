import Image
import numpy as np
from scipy.fftpack import fft2, ifft2
from scipy import fftpack, ndimage
import matplotlib.pyplot as plt

image = ndimage.imread("arbol(1).png") 


arbol = np.array(image)
fourier2=np.fft.fft2(arbol)
freq = np.fft.fftfreq(len(arbol[0]))

plt.imshow(np.log10(abs(fourier2)))
plt.figure()
plt.plot(freq,np.log10(abs(fourier2[:,0])))
plt.show()
print arbol[0,0]
