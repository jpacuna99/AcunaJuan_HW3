import numpy as np
import matplotlib.pyplot as plt


datos=np.genfromtxt("WDBC.dat",delimiter=",")
datos1=np.transpose((np.array(datos)))[2:,:]

for i in range(len(datos1)):
	datos1[i] = datos1[i]-np.mean(datos1[i])
	datos1[i]=datos1[i]/(np.sqrt(np.var(datos1[i])))










def matrizcov(data):
	cov=np.zeros((np.shape(data)[0],np.shape(data)[0]))
	
	

	for i in range(1,np.shape(data)[0]+1):
		for j in range(1,np.shape(data)[0]+1):
			cov[i-1,j-1]=np.sum((data[i-1,:]-np.mean(data[i-1,:]))*(data[j-1,:]-
np.mean(data[j-1,:])))/(np.shape(data)[1]-1)

	return cov

print np.shape(matrizcov(datos1))


valoresP=np.linalg.eig(matrizcov(datos1))[0]
eigenvectors=np.linalg.eig(matrizcov(datos1))[1]

print valoresP
"""
for i in range (30):
	print "Autovalor: ", valoresP[i], "Autovector: ", eigenvectors[:,i]

print "Los parametros mas importantes son "

vector1=eigenvectors[:,0]
vector2=eigenvectors[:,1]
print "Se usan los dos primeros autovectores porque sus valores propios son los mas significativos"
"""

v1=eigenvectors[:,0]
v2=eigenvectors[:,1]



pc1=np.dot(np.transpose(datos1),v1)
pc2=np.dot(np.transpose(datos1),v2)

tumor=np.genfromtxt("WDBC.dat", delimiter=",",dtype="unicode")[:,1]


print len(tumor)
buenos1=[]
malos1=[]
buenos2=[]
malos2=[]
for i in range(len(tumor)):
	if tumor[i]=="M":
		malos1.append(pc1[i])
		malos2.append(pc2[i])
	if tumor[i]=="B":
		buenos1.append(pc1[i])
		buenos2.append(pc2[i])



plt.scatter(buenos1,buenos2,c="green",label="Benignos")

plt.scatter(malos1,malos2,c="red",label="Daninos")
plt.legend()
plt.show()




