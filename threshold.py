import cv2 
import numpy as np
import matplotlib.pyplot as plt
import math
class Segmentation(object):
    def __init__(self,kernel_size=3,sigma=1):
        self.image = None
        self.N = None
        self.M = None
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.kernel = self.gaussian_kernel()
        self.T = None
        self.R1 = None
        self.R2 = None
        self.mean1 = None
        self.mean2 = None
        self.iter = 0
    def threshold(self,image):
        self.image = image
        smooth = self.gaussian_blur(image)
        self.N = len(smooth)
        self.M = len(smooth[0])
        self.calculate_T(smooth)
        threshold = [[0 for k in range(self.M)] for i in range(self.N)]
        for i in range(self.N):
            for k in range(self.M):
                if(smooth[i][k]>self.T):
                    threshold[i][k] = 255
                else:
                    threshold[i][k] = 0
        return threshold                
    def calculate_T(self,smooth):
        self.T = int(sum(list(map(sum, smooth)))/(self.N*self.M))
        while(True):
            if (self.iter>10):
                break
            else:
                R1 = []
                R2 = []
                for i in range(self.N):
                    for k in range(self.M):
                        if(smooth[i][k]>self.T):
                            R1.append(smooth[i][k])
                        else:
                            R2.append(smooth[i][k])
                self.R1 = R1
                self.R2 = R2
                self.mean1 = int(sum(self.R1)/len(self.R1))   
                self.mean2 = int(sum(self.R2)/len(self.R2))    
                T = 0.5*(self.mean1+self.mean2)
                print(T,self.T)
                if(abs(T-self.T)<2):
                    self.iter +=1        
                self.T = T    
    def gaussian_blur(self,image):    
        smooth = self.convolve(image,self.kernel)
        return smooth
    def convolve(self,img,kernel):
        size = len(kernel)
        smallWidth = len(img) - (size-1)
        smallHeight = len(img[0]) - (size-1)
        output = [([0]*smallHeight) for i in range(smallWidth)]
        for i in range(smallWidth):
            for j in range (smallHeight):
                value = 0
                for n in range(size):
                    for m in range(size):
                        value = value + img[i+n][j+m]*kernel[n][m]
                output[i][j] = int(value)      
        return output     
    def gaussian_kernel(self):
        x = [[-1+k for i in range(self.kernel_size)] for k in range(self.kernel_size)]
        y = [[-1,0,1] for k in range(self.kernel_size)]
        g = [[0,0,0] for k in range(self.kernel_size)]
        normal = 1 / (2.0 * math.pi * self.sigma**2)
        for i in range(self.kernel_size):
            for k in range(self.kernel_size):
                g[i][k] = math.exp(-((x[i][k]**2 + y[i][k]**2) / (2.0*self.sigma**2))) * normal
        return g
image = cv2.imread('Inter-means_thresholding/apple.jpg',0)

seg =  Segmentation()
t = np.array(seg.threshold(image),dtype=np.uint8)
cv2.imshow("image",t)
cv2.waitKey(0)