import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
import math
import os
import copy

# detectie fete folosind LBP

# LBP calculat per fereastra
def LBP_2(fer):

    c = fer[1,1]
    i7 = fer[0,0]>c
    i6 = fer[0,1]>c
    i5 = fer[0,2]>c
    i4 = fer[1,2]>c
    i3 = fer[2,2]>c
    i2 = fer[2,1]>c
    i1 = fer[2,0]>c
    i0 = fer[1,0]>c
    val = i7*128+i6*64+i5*32+i4*16+i3*8+i2*4+i1*2+i0*1
    return val

# LBP calculat direct la nivel de imagine
def LBP(im):
    line = im.shape[0]
    col = im.shape[1]
    c = im[1:line-1, 1:col-1]
    i7 = im[0:line-2,0:col-2]>c
    i6 = im[0:line-2,1:col-1]>c
    i5 = im[0:line-2,2:col]>c
    i4 = im[1:line-1,2:col]>c
    i3 = im[2:line,2:col]>c
    i2 = im[2:line,1:col-1]>c
    i1 = im[2:line,0:col-2]>c
    i0 = im[1:line-1,0:col-2]>c
    val = i7*128+i6*64+i5*32+i4*16+i3*8+i2*4+i1*2+i0*1
    return val

plt.close("all")  
'''
img = io.imread("./Mari/BioID_F1_1.jpg")
img_g = color.rgb2gray(img)

img_lbp = copy.deepcopy(img_g)
plt.figure()
plt.imshow(img_g, cmap="gray")
'''

#a = [[86, 91, 68], [87,80,79], [76,88,74]]
#a = np.asarray(a)
#print(type(a), type(img_g))
#b = LBP(a)
#print(b)
#print(len(img_g[0]), len(img_g))

'''
for i in range(1,img_g.shape[0]-1):
    for j in range(1, img_g.shape[1]-1):
        fer = img_g[i-1:i+2,j-1:j+2]
        img_lbp[i,j] = LBP_2(fer)
'''

poze = os.listdir('./Mari')
mat_descriptor = np.zeros([12,12544]) #12544 = 49 ferestre * 255 bini
# masca de ponderare (din platforma)
mask = np.ones([7,7])
mask[3:7,0] = 0
mask[3:7,6] = 0
mask[2:4,3] = 0
mask[0:2,0] = 2
mask[0:2,6] = 2
mask[5,3] = 2
mask[1,1:3] = 4
mask[1,4:6] = 4

for k,nume in enumerate(poze):
    l = []
    # citire imagini
    img = io.imread('./Mari/'+nume)
    gray = color.rgb2gray(img)
    # val = dimens img /7 (pt ca impart img in 49 patrate)
    val = math.ceil(img.shape[0]/7)
    for i in range(7):
        for j in range(7):
            # pt fiecare fereastra aplic LBP
            fer = gray[i*val:(i+1)*val, j*val:(j+1)*val]
            lbp = LBP(fer)
            # calcul histograma pt rezultatul lbp
            rez = np.histogram(lbp,bins = 256)[0] * mask[i,j]
            l.append(rez)
    l = np.asarray(l)
    l = l.flatten()
    # descriptorul pt img k salvat in matrice de descriptori
    mat_descriptor[k,:] = l
    
# calcul distante intre descriptori
dist = np.zeros([mat_descriptor.shape[0],mat_descriptor.shape[0]]) 
arrange = np.zeros([mat_descriptor.shape[0],mat_descriptor.shape[0]])
for i in range(mat_descriptor.shape[0]):
    for j in range(mat_descriptor.shape[0]):
        dist[i,j] = math.sqrt(sum((mat_descriptor[i]-mat_descriptor[j])**2))
    # sortarea argumentelor in functie de valori (a index ului imaginii in functie de distanta)
    arrange[i] = np.argsort(dist[i,:])
# afisare pt primele 3 cele mai apropiate valori    
print(arrange[:,0:3])

wrong_class = 0
for i in range(len(arrange)):

    s = sum(arrange[i,0:3])
    if(i//3==0):
        if(s != 3):
            wrong_class = wrong_class+1
    else:
        if(i//3==1):
            if(s!=12):
                wrong_class = wrong_class+1
        else:
            if(i//3==2):
                if(s!=21):
                    wrong_class = wrong_class+1
            else:
                if(i//3==3):
                    if(s!=30):
                        wrong_class = wrong_class+1
    
print(wrong_class)
correct = (arrange.shape[0]-wrong_class)/arrange.shape[0]*100
print(correct, "%")

    

     

    