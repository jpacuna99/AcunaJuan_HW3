import numpy as np
import matplotlib.pylab as plt
from scipy.fftpack import fft, ifft, fftfreq
from scipy.interpolate import interp1d



signal=np.transpose(np.genfromtxt("signal.dat",delimiter=","))

plt.plot(signal[0],signal[1])

print np.shape(signal)



N = 512 # number of point in the whole interval
freq =np.array(fftfreq(N,signal[0,1]-signal[0,0])) #  frequency in Hz





# SU implementacion de la transformada de fourier
def fourier(f):
	b=[]
	
	for k in range(0,len(f)):
		a=0
		for i in range(0,len(f)):
			a=a+f[i]*np.exp(2*-1j*np.pi*i*(float(k)/float(N)))
		b.append(a)
	b=np.array(b)
	return b


enes=[]
print np.shape(freq), np.shape(abs(fourier(signal[1])))
for i in range(1,N):
	enes.append(float(i)/float(N))


#plt.figure()
#plt.plot(freq,abs(fourier(signal[1])))


filtro=np.copy(abs(fourier(signal[1])))

for i in range(len(fourier(signal[1]))):
		
	if abs(freq[i])>1000 :
		filtro[i]=0


plt.plot(signal[0],np.real(ifft(filtro)))

plt.figure()
#plt.plot(freq, filtro)
plt.show()













