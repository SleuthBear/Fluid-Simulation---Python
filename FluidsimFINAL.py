# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 19:59:33 2020

@author: alpha

Using matrix calculations from Phillip Zucker
http://www.philipzucker.com/annihilating-my-friend-will-with-a-python-fluid-simulation-like-the-cur-he-is/
"""


import numpy as np
import cv2
from scipy import interpolate
from scipy import ndimage
from scipy import sparse
import scipy
import scipy.sparse.linalg as linalg
import itertools
import matplotlib.pyplot as plt

#The time step
dt = 0.01


#Adds in the viscosity and diffusion rate
visc = 0.002
diff = 0.01

#The image used as a texture map for the visualisation
img = cv2.imread("checks.jpg")
img = cv2.pyrDown(img)
img = cv2.pyrDown(img)


#The dimensions of the array of vector points
Nx = img.shape[0]
Ny = img.shape[1]

#Creates a 2 dimensional vector field
v = np.zeros((Nx,Ny,2))
v0 = np.zeros((Nx,Ny,2))
#The force array act on the fluid
F = np.zeros((Nx,Ny,2))

#Creates arrays X and Y to store the X and Y coordinates
x = np.linspace(0,1,Nx, endpoint=False)
y = np.linspace(0,1,Ny, endpoint=False)
X, Y = np.meshgrid(x, y, indexing = "ij")





# A function to create the intitial vortex
def build_vortex(v, x, y, size):
    i=1
    x = x+1
    direction = itertools.cycle([[1, 0], [0, 1], [-1, 0], [0, -1]])
    magnitude = itertools.cycle([[3,0], [0,3], [-3,0], [0,-3]])
    while i < size:  
        direc = next(direction)
        mag = next(magnitude)
        for j in range(i):
            v[y,x] = mag
            y += direc[0]
            x += direc[1]
        i+=1
    return v 
v = build_vortex(v, int(Nx/2), int(Ny/2), 50)              
            
        
        
#########Build the matrices for interpoaltion and derivatives#################        
#Creates N x N finite difference matrix
#Following the forward gradient (first order taylor)   
def Build_grad(N):
    data = np.array([-np.ones(N),np.ones(N)])
    return sparse.diags(data, np.array([0,1]), shape = (N, N))

#Create the gradient operators. creates a kronecker product of an identity
#matrix and the previously defined derivative matrix.
gradx = sparse.kron(Build_grad(Nx), sparse.identity(Ny))
grady = sparse.kron(sparse.identity(Nx), Build_grad(Ny))

#Creates an N-1xN-1 central difference matrix to calculate 
#a second derivative
def Build_K(N):
    data = np.array([-np.ones(N-1), 2*np.ones(N), -np.ones(N-1)])
    diags = np.array([-1,0,1])
    return sparse.diags(data,diags)

# Create a laplacian operator apparently the directions are reversed, which is
# why Ny comes first
K = sparse.kronsum(Build_K(Ny), Build_K(Nx))
D = sparse.kronsum(Build_K(Ny), Build_K(Nx))
#projection solver
Ksolve = linalg.factorized(K)
#velocity diffusion solver
Vsolve = linalg.factorized(sparse.identity((Nx)*(Ny)) - D*visc)
#scalar diffusion solver
Dsolve = linalg.factorized(sparse.identity((Nx)*(Ny)) - D*diff)

##############################################################################
# Projects the velocity field onto the divergence free field to conserve mass
def Project(v):
    vx = v[:,:,0]
    vy = v[:,:,1]
    dvx, dvy = ProjectPass(vx,vy)
    v[:,:,0] -= dvx
    v[:,:,1] -= dvy
    return v

# finds the divergent part of the vector field so it can be subtracted
def ProjectPass(vx, vy):
    #sets the boundary conditions
    vx[0,:]/=2
    vx[-1,:]/=2    
    vy[:,0]/=2
    vy[:,-1]/=2
    #Calculates the divergence
    div = gradx.dot(vx.flatten()) + grady.dot(vy.flatten())
    #Calculates the divergence component of the vector field
    w = Ksolve(div.flatten())
    return gradx.T.dot(w).reshape(Nx,Ny), grady.T.dot(w).reshape(Nx,Ny)
    
# adds the effects of a force to the vector field
def addForce(v, F):
    v = v + F
    return v

# moves the vector field by itself    
def advect(v, v0):
    #Create the particle trace coordinates using euler method, and assuming
    #the whole array has size 1.
    coords = np.stack([(X - v[:,:,0]*dt)*Nx, (Y - v[:,:,1]*dt)*Ny], axis=0)
    #Uses a scipy built in interpolation function
    v[:,:,0] = ndimage.map_coordinates(v[:,:,0], coords, order = 5, mode = \
                                       "wrap")
    v[:,:,1] = ndimage.map_coordinates(v[:,:,1], coords, order = 5, mode = \
                                       "wrap")
    return v

# moves the image coordinates by the vector field
def advectS(v, img):
    coords = np.stack([(X - v[:,:,0]*dt)*Nx, (Y-v[:,:,1]*dt)*Ny])
    for j in range(3):
        img[:,:,j] = ndimage.map_coordinates(img[:,:,j], coords, order=5, \
                                             mode='wrap')
    return img

# spreads out the velocity to adjacent gridpoints over time
def DiffuseV(v, v0, visc): 
    v[:,:,0] = Vsolve(v[:,:,0].flatten()).reshape(Nx,Ny)
    v[:,:,1] = Vsolve(v[:,:,1].flatten()).reshape(Nx,Ny)
    return v

# spreads out the values of the colours in the image over time to mimic 
# diffusion of a dye (is disabled by default)
def DiffuseS(img):
    img0 = img[:,:,0].copy()
    img1 = (img[:,:,1].copy())
    img2 = img[:,:,2].copy()
    img[:,:,0] = Dsolve(img0.flatten()).reshape(Nx,Ny)
    img[:,:,1] = Dsolve(img1.flatten()).reshape(Nx,Ny)
    img[:,:,2] = Dsolve(img2.flatten()).reshape(Nx,Ny)
    return img
    
# moves the velocity field forward 1 timestep
def VStep(v, v0, img, F):     
    v = addForce(v, F) 
    v = advect(v, v0)
    v = DiffuseV(v, v0, visc)
    v = Project(v)
    return v, v0, img
  
# moves the image forward by 1 timestep  
def SStep(v, img):
    img = advectS(v, img)
    #img = DiffuseS(img)
    return img
    
# calls both the velocity and image steps
def TimeStep(v0, v, img, F):
    v, v0, img = VStep(v, v0, img, F) 
    img = SStep(v, img)
      
    return v0, v, img

    
    
    
#frames = []    
#xvels = [] 
#frame = 0
#vals_above = 0
#vals_list = []

# runs the simulation for set number of timesteps. code to produce images
# have been commented out.
for i in range(400):
    print("running")
    v0, v, img = TimeStep(v0, v, img, F)
    cv2.imshow('image', img)   
    #cv2.imwrite(f"C:/FILEPATH/fluid{i}.jpg", img)
    #xvels.append(v[int(Ny/2+20),int(Nx/2)])
    #frame+=1
    #frames.append(frame)
    #for row in range(Ny):
    #    for col in range(Nx):
    #        if(np.sqrt(v[row,col,0]**2 + v[row,col,1]**2)>1):
    #            vals_above+=1
    #vals_list.append(vals_above)
                
                
                
    #cv2.imwrite(f'arghh/{i:06}.jpg',img)
    k = cv2.waitKey(15) & 0xFF
    if k==ord(' '):
        break
cv2.destroyAllWindows()
 
#plt.title("Velocity 20 pixels above vortex")
#plt.xlabel("yellow = horizontal velocity blue = vertical velocity")  
#plt.plot(frames, xvels)   
#plt.savefig("scalar diffusion graph 2.jpg")
#plt.plot(frames, vals_list)
#plt.title("number of gridpoints with velocity above 1")
#plt.xlabel("timesteps")  








