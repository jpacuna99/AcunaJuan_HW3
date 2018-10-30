import numpy as np
import matplotlib.pylab as plt
from scipy.fftpack import fft, ifft, fftfreq
from scipy.interpolate import interp1d



signal=np.transpose(np.genfromtxt("signal.dat",delimiter=","))
N = 512 # number of point in the whole interval
freq =np.array(fftfreq(N,signal[0,1]-signal[0,0])) #  frequency in Hz

plt.figure()
plt.plot(signal[0],signal[1],label="Datos originales")
plt.legend()
plt.savefig("AcunaJuan_signal.pdf")




# Implementacion de la transformada de fourier
def fourier(f):
	b=[]
	
	for k in range(0,len(f)):
		a=0
		for i in range(0,len(f)):
			a=a+f[i]*np.exp(2*-1j*np.pi*i*(float(k)/float(N)))
		b.append(a)
	b=np.array(b)
	return b


plt.figure()
plt.plot(freq,abs(fourier(signal[1])),label="Transformanda")
plt.legend()
plt.savefig("AcunaJuan_TF.pdf")



filtro=np.copy(fourier(signal[1]))

for i in range(len(fourier(signal[1]))):
		
	if abs(freq[i])>1000 :
		filtro[i]=0




plt.figure()
plt.plot(signal[0], np.real(ifft(filtro)),label="Senal filtrada")
plt.legend()
plt.savefig("AcunaJuan_Filtrada.pdf")



#Interpolaciones


def interpolacionCua(a):
	f1 = interp1d(a[:,0], a[:,1],kind='quadratic')
	return f1

def interpolacionCub(a):
	f2= interp1d(a[:,0], a[:,1],kind='cubic')	
	return f2

#Plots de cada interpolacion
xnew=np.linspace(0.000390625 ,0.028515625,512)

incompletos=np.genfromtxt("incompletos.dat",delimiter=",")



intcua=interpolacionCua(incompletos)(xnew)
intcub=interpolacionCub(incompletos)(xnew)
fouriercua=fourier(intcua)
fouriercub=fourier(intcub)
freqcua =np.array(fftfreq(N,intcua[1]-intcua[0]))
freqcub =np.array(fftfreq(N,intcub[1]-intcub[0]))


plt.figure()
f, (ax1, ax2,ax3) = plt.subplots(3, 1, sharey=True)
ax1.plot(freq,abs(fourier(signal[1])))
ax1.set_title('Datos completos')
ax2.plot(freqcua,abs(fouriercua))
ax2.set_title('InterCua')
ax3.plot(freqcub,abs(fouriercub))
ax3.set_title('Inter Cub')
plt.savefig("AcunaJuan_TF_Interpola.pdf")





filtrocua100=np.copy(fouriercua)

for i in range(len(fouriercua)):
		
	if abs(freqcua[i])>1000 :
		filtrocua100[i]=0

filtrocub100=np.copy(fouriercub)

for i in range(len(fouriercua)):
		
	if abs(freqcua[i])>1000 :
		filtrocua100[i]=0

filtrocua500=np.copy(fouriercua)

for i in range(len(fouriercua)):
		
	if abs(freqcua[i])>500 :
		filtrocua100[i]=0

filtrocub500=np.copy(fouriercub)

for i in range(len(fouriercub)):
		
	if abs(freqcua[i])>500 :
		filtrocua100[i]=0





filtro500=np.copy(fft(signal[1]))

for i in range(len(fft(signal[1]))):
		
	if abs(freq[i])>500 :
		filtro500[i]=0




plt.figure()
f, (ax1, ax2) = plt.subplots(2, 1, sharey=True)
ax1.plot(signal[0], np.real(ifft(filtro)),label="Datos originales")
ax1.plot(xnew,np.real(ifft(filtrocua100)),label="Int cua")
ax1.plot(xnew,np.real(ifft(filtrocub100)),label="Int cub")
ax1.legend()
ax1.set_title('Filtro 1000')



ax2.plot(signal[0], np.real(ifft(filtro500)),label="Datos originales")
ax2.plot(xnew,np.real(ifft(filtrocua500)),label="Int cua")
ax2.plot(xnew,np.real(ifft(filtrocub500)),label="Int cub")
ax2.legend()


ax2.set_title('Filtro 500')
plt.savefig("AcunaJuan_2Filtros.pdf")


plt.show()
 












	








