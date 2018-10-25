import Image
import numpy as np
from scipy.fftpack import fft2, ifft2
import matplotlib.pyplot as plt
pilimg = Image.open("arbol(1).png")


arbol = np.array(pilimg)
print np.shape(arbol)
