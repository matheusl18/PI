from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

img = Image.open('imagens/Red_Wine_Glass.jpg')
matriz = np.asarray(img)
print(matriz)
plt.imshow(img)
plt.show()


img_gray = img.convert('L')
matrix_gray = np.asarray(img_gray)
print(matrix_gray)
plt.imshow(img_gray, cmap='gray')
plt.show()

histograma = np.zeros(256).astype(int)
print(histograma)

linhas = matrix_gray.shape[0]
colunas = matrix_gray.shape[1]

for i in range(linhas):
    for j in range(colunas):
        cor = matrix_gray[i,j]
        cor = int(cor)
        histograma[cor] = histograma[cor] + 1


print(histograma)
plt.plot(range(256), histograma)
plt.show()



ret, thresh = cv.threshold(matrix_gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
plt.plot(range(256), histograma)
plt.plot([ret, ret], [0, 600])
plt.show()

plt.imshow(thresh, cmap='gray')
plt.show()


mimg_blur = matrix_gray.copy()

lins = matrix_gray.shape[0]
cols = matrix_gray.shape[1]

m = matrix_gray
for i in range(1, lins-1):
    for j in range(1, cols-1):
        mimg_blur[i][j] = (
                                    1*m[i-1][j-1] + 1*m[i-1][j  ] + 1*m[i-1][j+1] + 
                                    1*m[i  ][j-1] + 1*m[i  ][j  ] + 1*m[i  ][j+1] + 
                                    1*m[i+1][j-1] + 1*m[i+1][j  ] + 1*m[i+1][j+1]
                                    )/9
        
plt.imshow(mimg_blur, cmap='gray')
plt.show()


sobelx = cv.Sobel(matrix_gray, cv.CV_64F,1,0,ksize=5)  
sobely = cv.Sobel(matrix_gray, cv.CV_64F,0,1,ksize=5)
plt.imshow(sobelx, cmap = 'gray')
plt.show()
plt.imshow(sobely, cmap = 'gray')
plt.show()


matrix_sobel = sobelx + sobely
for i in range(1, lins-1):
    for j in range(1, cols-1):
        matrix_sobel[i][j] = max(0, matrix_sobel[i][j])
        matrix_sobel[i][j] = min(255, matrix_sobel[i][j])

plt.imshow(matrix_sobel, cmap='gray')
plt.show()

unsharp = np.subtract(matrix_gray, mimg_blur)
plt.imshow(unsharp, cmap='gray')
plt.show()

sharpening = np.array(unsharp) + np.array(matrix_gray)
plt.imshow(sharpening, cmap='gray')
plt.show()