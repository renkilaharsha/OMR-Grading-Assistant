import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
df = pd.read_csv("voting_a-27.csv")
print(df.shape)
k = df.to_numpy()
#print(k)
data = image.imread('test-images/a-27.jpg')

array = np.zeros((k.shape))
ind = np.unravel_index(np.argsort(k, axis=None), k.shape)
array_k = np.zeros((k.shape))
print(len(array))
i=0
while i <len(array):
    start = i
    index = start
    flag = 1
    while flag:
        if(k[index,0] >10):
            index+=1
        else:
            flag=0
    if(index-start >10):
        array_k[start,0] =1
        array_k[index,0] =1
        print("0",start,index)
    i += index-start+1
    flag =1

print(len(array))
i=0
while i <len(array):
    start = i
    index = start
    flag = 0
    while flag==0:
        if(k[index,90] >10):
            index+=1
        else:
            flag=1
    if (index - start > 10):
        array_k[start, 90] = 1
        array_k[index, 90] = 1
        print(start,index)
    flag=0
    i = i+ index-start+1


'''print(max(k[:,0]))
p = np.argsort(k[:, 0])
for j in range(100):
    array[p[-1-j],0]= 1
print(len(ind[0]),len(ind[1]))'''
#print(ind)
'''sum = 0
for thres in range(1000):
    i,j = ind[0][-1-thres],ind[1][-1-thres]
    if(j==0):
        print(ind[0][-1-thres],ind[1][-1-thres],k[i,j],sum)
        array[i,j] = 1
        sum +=1
        if(sum>250):
            break
p =0
for thres in range(1000):
    i,j = ind[0][-1-thres],ind[1][-1-thres]
    if(j==90):
        print(ind[0][-1-thres],ind[1][-1-thres],k[i,j],"sum_h",sum)
        array[i,j] = 1
        p+=1
        if (p > 250):
            break'''

'''for i in range(len(k)-1):
    for j in range(len(k[0])-1):
        if(array_1[i,j]==1):
            if(k[i+1,j] > k[i,j]  and k[i-1,j] > k[i,j]):
                array_1[i,j] = 0
            if (k[i , j-1] > k[i, j] and k[i, j+1] > k[i, j]):
                array[i, j] = 0'''
'''sum =0
for thres in range(len(ind[0])):
    i,j = ind[0][-1-thres],ind[1][-1-thres]
    print(ind[0][-1-thres],ind[1][-1-thres],k[i,j])
    if(i==90 or i==270  and  sum<100):
        array[i,j] = 1
        sum+=1'''


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
x = np.linspace(0,len(data),1000)
'''ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')'''
print(max(k[:,0]))
count  = 0
for i in range(len(array)):
    for j in range(len(array[0])):
        #x = range(0, 2500)
        if(array_k[i][j]==1 ):
            if j==0:
                plt.axhline(y=i, color='b')
            elif(j==90):
                plt.axvline(x=i, color='k')
            count+=1
print(count)
'''for i in range(len(array)):
    for j in range(len(array[0])):
        x_p = [0]*1000
        if(array[i][j]==1 and (j<=5 or(j>=175))):
            y_p = [i]*len(x_p)
            plt.plot(x_p,y_p)

        if (array[i][j] == 1 and (j >= 85 or j<=95)):
            plt.axvline(x=i, color='k')

            x = [0] * len(array)
            y = [i] * len(array)
            print(x,y)
            plt.plot(x, y)'''
plt.legend(loc='upper left')
#plt.axes("off")
plt.imshow(data)

plt.show()
