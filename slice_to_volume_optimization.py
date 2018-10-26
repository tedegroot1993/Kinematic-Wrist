#! /usr/bin/env python2.7

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.ndimage.filters import convolve, gaussian_filter
#import cv2 as cv2
from scipy.ndimage.measurements import label
from scipy import ndimage
import pickle

mat_file = sio.loadmat('Volume_segmented.mat')
bones = mat_file['volume_I_dynamic']

def CannyEdgeDetector(im, blur, highThreshold, lowThreshold,mins,maxs):
    # Install a package to eliminate writing this function
    mins = np.int64(mins)
    maxs = np.int64(maxs)
    
    im = np.array(im, dtype=float) #Convert to float to prevent clipping values
 
    #Gaussian blur to reduce noise
    im2 = gaussian_filter(im, blur)
 
    #Use sobel filters to get horizontal and vertical gradients
    im3h = convolve(im2,[[-1,0,1],[-2,0,2],[-1,0,1]]) 
    im3v = convolve(im2,[[1,2,1],[0,0,0],[-1,-2,-1]])

 
    #Get gradient and direction
    grad = np.power(np.power(im3h, 2.0) + np.power(im3v, 2.0), 0.5)
    theta = np.arctan2(im3v, im3h)
    thetaQ = (np.round(theta * (5.0 / np.pi)) + 5) % 5 #Quantize direction
 
    #Non-maximum suppression
    gradSup = grad.copy()
#     for r in range(im.shape[0]):
    for r in range(mins[0]-3,maxs[0]+3):
#         for c in range(im.shape[1]):
        for c in range(mins[1]-3,maxs[1]+3):
            #Suppress pixels at the image edge
            if r == 0 or r == im.shape[0]-1 or c == 0 or c == im.shape[1] - 1:
                gradSup[r, c] = 0
                continue
            tq = thetaQ[r, c] % 4
 
            if tq == 0: #0 is E-W (horizontal)
                if grad[r, c] <= grad[r, c-1] or grad[r, c] <= grad[r, c+1]:
                    gradSup[r, c] = 0
            if tq == 1: #1 is NE-SW
                if grad[r, c] <= grad[r-1, c+1] or grad[r, c] <= grad[r+1, c-1]:
                    gradSup[r, c] = 0
            if tq == 2: #2 is N-S (vertical)
                if grad[r, c] <= grad[r-1, c] or grad[r, c] <= grad[r+1, c]:
                    gradSup[r, c] = 0
            if tq == 3: #3 is NW-SE
                if grad[r, c] <= grad[r-1, c-1] or grad[r, c] <= grad[r+1, c+1]:
                    gradSup[r, c] = 0
 
    #Double threshold
    strongEdges = (gradSup > highThreshold)
 
    #Strong has value 2, weak has value 1
    thresholdedEdges = np.array(strongEdges, dtype=np.uint8) + (gradSup > lowThreshold)
 
    #Tracing edges with hysteresis    
    #Find weak edge pixels near strong edge pixels
    finalEdges = strongEdges.copy()
    currentPixels = []
    for r in range(1, im.shape[0]-1):
        for c in range(1, im.shape[1]-1):    
            if thresholdedEdges[r, c] != 1:
                continue #Not a weak pixel
 
            #Get 3x3 patch    
            localPatch = thresholdedEdges[r-1:r+2,c-1:c+2]
            patchMax = localPatch.max()
            if patchMax == 2:
                currentPixels.append((r, c))
                finalEdges[r, c] = 1
                
    #Extend strong edges based on current pixels
    while len(currentPixels) > 0:
        newPix = []
        for r, c in currentPixels:
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    if dr == 0 and dc == 0: continue
                    r2 = r+dr
                    c2 = c+dc
                    if thresholdedEdges[r2, c2] == 1 and finalEdges[r2, c2] == 0:
                        #Copy this weak pixel to final result
                        newPix.append((r2, c2))
                        finalEdges[r2, c2] = 1
        currentPixels = newPix
    return finalEdges

def COM_f(I):
    # Calculatees center of mass of a 2D array
    S = np.shape(I)
    sum_x = float(0)
    sum_y = float(0)
    total = float(0)
    for i in range(S[0]):
        for j in range(S[1]):
            if I[i][j] > 0:
                sum_x = sum_x + i
                sum_y = sum_y + j
                total = total + 1
    x = ((sum_x)/total) - float(S[0]/2)
    y = ((sum_y)/total) - float(S[1]/2)
    return x,y

def combine_slices(A,slice_range,mins,maxs):  
    S = np.shape(A)
    output_slice = np.zeros((S[0],S[1]))
    z = slice_range[0]
    mins = np.int64(mins)
    maxs = np.int64(maxs)
    
    for x in range(mins[0]-1,maxs[0]+1):
        for y in range(mins[1]-1,maxs[1]+1):
            for z in range(slice_range[0],slice_range[1]):
                output_slice[x][y] = output_slice[x][y] + A[x][y][z]
    
    output_slice = ndimage.gaussian_filter(output_slice, 3)
               
    return output_slice              

def Rz(theta):
    # Calculates transformation matrix
    # about the z axis
    # input in degrees
    
    theta = np.deg2rad(theta)
    C = np.cos(theta)
    S = np.sin(theta)
    
    Rz = np.array([[C, -S, 0, 0],[S, C, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]])
    return Rz

def Rx(theta):
    # Calculates transformation matrix
    # about the x axis
    # input in degrees
    
    theta = np.deg2rad(theta)
    C = np.cos(theta)
    S = np.sin(theta)
    
    Rx = np.array([[1, 0, 0, 0],[0, C, -S, 0],[0, S, C, 0],[0, 0, 0, 1]])
    return Rx

def Ry(theta):
    # Calculates transformation matrix
    # about the y axis
    # input in degrees
    
    theta = np.deg2rad(theta)
    C = np.cos(theta)
    S = np.sin(theta)
    
    Ry = np.array([[C, 0, S, 0],[0, 1, 0, 0],[-S, 0, C, 0],[0, 0, 0, 1]])
    return Ry

def Trans(Translations):
    # Calculates transformation matrix
    # for x,y,and z translation
    
    x = Translations[0]
    y = Translations[1]
    z = Translations[2]
    
    Trans = np.array([[1, 0, 0, x],[0, 1, 0, y],[0, 0, 1, z],[0, 0, 0, 1]])
    return Trans

def point_transformation(bone):
    # Calculates the x,y,z coordinates for each point in the matrix,
    # and shifts the origin to the center of the matrix
    S = np.shape(bone)
    
    l = S[0]*S[1]*S[2]
    
    P = np.zeros((4,l))
    
    j = 0
    
    for z in range(S[2]):
        my_slice = bone[:,:,z]
        [x_loc,y_loc] = np.nonzero(my_slice>0)
        x_shape = np.shape(x_loc)
        if x_shape[0] > 0:
            for i in range(len(x_loc)):
                P[0][j] = x_loc[i]-.5*S[0]
                P[1][j] = y_loc[i]-.5*S[1]
                P[2][j] = z-.5*S[2]
                P[3][j] = 1
                j = j + 1
    my_points = np.ones((4,j))
    k = 0
  
    while k < j:
        my_points[0][k] = P[0][k]
        my_points[1][k] = P[1][k]
        my_points[2][k] = P[2][k]
        k = k+1
    return my_points

def transform(Translations,theta,bone,P):
    # Transforms the volume based off of the 6 DOF (rotation and translation
    
    S_b = np.shape(bone)

    
    output_V = np.zeros((S_b[0],S_b[1],S_b[2]))

    
    S_p = np.shape(P)
    
    T = np.mat(Trans(Translations))*np.mat(Rx(theta[0]))*np.mat(Ry(theta[1]))*np.mat(Rz(theta[2]))
    
    T_P = np.mat(T)*np.mat(P)
    
    X = T_P[0][:]
    Y = T_P[1][:]
    Z = T_P[2][:]
    
    h_x = S_b[0]/2.
    h_y = S_b[1]/2.
    h_z = S_b[2]/2.
    
    minimums = [np.round(X.min()+h_x),np.round(Y.min()+h_y),np.round(Z.min()+h_z)]
    
    maximums = [np.round(X.max()+h_x),np.round(Y.max()+h_y),np.round(Z.max()+h_z)]
    
    for i in range(S_p[1]):
        x = np.round(T_P[0,i]+h_x)
        y = np.round(T_P[1,i]+h_y)
        z = np.round(T_P[2,i]+h_z)
        if (x>=0and x<255):
            if (y>=0 and y<255):
                if(1==1):
                    if (z>0 and z<100):
                        output_V[x][y][z] = 1        
    return output_V,minimums,maximums
       

def find_min_max(I):
    # find the maximum and minimum column and row locations of the rigid volume 
    I = np.round(I)
    S = np.shape(I)
    x_min = 50000
    x_max = 0
    y_min = 50000
    y_max = 0
    for i in range(S[0]):
        for j in range(S[1]):
            if I[i][j] > 0:
                if i<x_min:
                    x_min = i
                if i>x_max:
                    x_max = i 
                if j<y_min:
                    y_min = j
                if j>y_max:
                    y_max = j 
    
    return [x_min,y_min],[x_max,y_max]
                    

def isolate_I(I,mins,maxs):
    # passes through only the bones volume, removes artifacts caused by canny filter
    mins = np.int64(mins)
    maxs = np.int64(maxs)
    output_I = np.zeros(np.shape(I))
    for i in range(mins[0]+1,maxs[0]-1):
        for j in range(mins[1]+1,maxs[1]-1):
            output_I[i][j] = I[i][j]
    return output_I
    


def fitness_level(bone,locs,translations,theta,dynamic_slice):
    # Determines the fitness of a member of the population

    rotated_volume,minimums,maximums = transform(translations,theta,bone,locs)
    
    slice_I = combine_slices(rotated_volume,(48-3,48+3),minimums,maximums)
    
    slice_I = ndimage.zoom(slice_I,2)
    minimums = minimums*2
    maximums = maximums*2
    
    minimums,maximums = find_min_max(slice)
    
    edge_image = CannyEdgeDetector(slice,1,.2,.1,minimums,maximums)
    edge_image = isolate_I(edge_image,minimums,maximums)
    
    struct = ndimage.generate_binary_structure(2, 2)
    dilated_I = ndimage.binary_dilation(edge_image, structure=struct).astype(edge_image.dtype)
    
    fitness = np.sum(dilated_I*dynamic_slice)/np.sum(dilated_I)
 
    return fitness,slice_I

def initial_pop(COM_S,COM_D,pop_size,min_theta,max_theta):
    # creates an initial population, the two members with highest fitness score will be the output
    
    population = np.zeros((pop_size,6))
    
    A = np.linspace(min_theta,max_theta,pop_size)
    
    for i in range(pop_size):
        
        theta = A[i]
        theta_r = np.deg2rad(theta)
        
        C = np.cos(theta_r)
        S = np.sin(theta_r)
         
        population[i][0] = COM_D[0]-COM_S[0]*C+COM_S[1]*S
        population[i][1] = COM_D[1]-COM_S[0]*S-COM_S[1]*C
        
        population[i][5] = theta
        
    return population

def test_slice(bone,locs,translations,theta):
    # Creates a slice that the volume will be navigated to, for validation only
    test1,minimums,maximums = transform(translations,theta,bone,locs)
    I = combine_slices(test1,(48-3,48+3),(0,0),(255,255))  
    canny = CannyEdgeDetector(I,1,.2,.1,minimums,maximums)
    return I,canny

def cal_pop_fitness(equation_inputs, pop):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function caulcuates the sum of products between each input and its corresponding weight.
    fitness = np.sum(pop*equation_inputs, axis=1)
    return fitness

def select_mating_pool(pop, fitness, num_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    parents = np.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999
    return parents

def crossover(parent1,parent2,offspring_size):
    
    offspring = np.zeros((offspring_size,6))
    
    for i in range(offspring_size):
        offspring[i][0] = (parent1[0] + parent2[0])/2
        offspring[i][1] = (parent1[1] + parent2[1])/2
        offspring[i][2] = (parent1[2] + parent2[2])/2
        offspring[i][3] = (parent1[3] + parent2[3])/2
        offspring[i][4] = (parent1[4] + parent2[4])/2
        offspring[i][5] = (parent1[5] + parent2[5])/2
        
    return offspring

def mutation(offspring_crossover,mutation_rate,COM):
    # Mutation changes a single gene in each offspring randomly.
    S = np.shape(offspring_crossover)
    mutation_rate_rand = np.random.uniform(0,1,(S[0],S[1]))
    for i in range(offspring_crossover.shape[0]):
        if 1.5*mutation_rate_rand[i][0] > mutation_rate[0][0]:
            
            theta = offspring_crossover[i][5]+np.random.uniform(-15,15,1)
            theta_r = np.deg2rad(theta)
            
            C = np.cos(theta_r)
            S = np.sin(theta_r)
            
            offspring_crossover[i][5] = theta
            
            offspring_crossover[i][0] =  (C*COM[0] + S*COM[1] - COM[0]) + offspring_crossover[i][0] + np.random.uniform(-1.5,1.5,1)
            offspring_crossover[i][1] = (-S*COM[0] + C*COM[1] - COM[1]) + offspring_crossover[i][1] + np.random.uniform(-1.5,1.5,1)
            
        if mutation_rate_rand[i][2] > mutation_rate[0][2]:
            offspring_crossover[i][2] = offspring_crossover[i][2] + np.random.uniform(-4,4,1)
        if mutation_rate_rand[i][3] > mutation_rate[0][3]:
            offspring_crossover[i][3] = offspring_crossover[i][3] + np.random.uniform(-5,5,1)
        if mutation_rate_rand[i][4] > mutation_rate[0][4]:
            offspring_crossover[i][4] = offspring_crossover[i][4] + np.random.uniform(-5,5,1)
            
    return offspring_crossover


def two_parents(bone,locs,DOF,dynamic_slice,parent1,parent1_fitness,COM1,parent2,parent2_fitness,COM2):
    # Determine the two most fitting parents from previous generation
    # elitism is implemented so the parents will only be updated if the population has a member with a
    # a better fitness score
    
    [pop_size,D] = np.shape(DOF)
    fitness = np.zeros((pop_size,1))
    S = np.shape(bone)
    
    struct = ndimage.generate_binary_structure(2, 2)
    dynamic_slice = ndimage.zoom(dynamic_slice,2)
    dynamic_slice_canny = CannyEdgeDetector(dynamic_slice,1,.2,.1,(5,5),(500,500))
    
    for i in range(pop_size):
        translations = [DOF[i][0],DOF[i][1],DOF[i][2]]
        theta = [DOF[i][3],DOF[i][4],DOF[i][5]]
        fitness[i][0],I = fitness_level(bone,locs,translations,theta,dynamic_slice_canny)
        
        
        if fitness[i][0] > parent1_fitness:
            parent1_fitness = fitness[i][0]
            parent1 = np.array(translations+theta)
            COM1 = COM_f(I)
        elif fitness[i][0] > parent2_fitness:
            parent2_fitness = fitness[i][0]
            parent2 = np.array(translations+theta)
            COM2 = COM_f(I)
    
    return parent1_fitness, parent1, COM1, parent2_fitness, parent2, COM2

def mutation_rate_f(old_DOF,new_DOF):
    
    R = np.ones((1,6))/4
    for i in range(6):
        if abs(old_DOF[i] - new_DOF[i]) > 0:
            R[i] = .4
    
    return R
    

def genetic_algorithm(bone,locs,initial_pop_size,generations,I_D,true_DOF):
    
    I_S = combine_slices(bone,(48-3,48+3),(0,0),(255,255))
    
    COM_S = COM_f(I_S)
    COM_D = COM_f(I_D)
    
    population_1 = initial_pop(COM_S,COM_D,initial_pop_size,-45,45)
    
    fitness_1,parent_1,COM_P1,fitness_2,parent_2,COM_P2 = two_parents(bone,locs,population_1,I_D,(0,0,0,0,0,0),0,0,(0,0,0,0,0,0),0,0)
    
    new_DOF = (parent_1+parent_2)/2  
    
    mutation_rate = np.ones((1,6))/4
      
    for i in range(generations):
        population = crossover(parent_1,parent_2,10)
        COM_new = [(COM_P1[0]+COM_P2[0])/2, (COM_P1[0]+COM_P2[1])/2]
        population = mutation(population,mutation_rate,(COM_new))
        fitness_1,parent_1,COM_P1,fitness_2,parent_2,COM_P2 = two_parents(bone,locs,population,I_D,parent_1,fitness_1,COM_P1,parent_2,fitness_2,COM_P2)
        new_DOF = (parent_1+parent_2)/2    
        print(new_DOF)
    return new_DOF

def rand_DOF(DOF_size):
    
    DOF = np.random.uniform(-1,1,DOF_size)
    
    DOF[0] = DOF[0]*25
    DOF[1] = DOF[1]*25
    
#     DOF[2] = DOF[2]*4
#     DOF[3] = DOF[3]*5
#     DOF[4] = DOF[4]*5
    
    DOF[2] = 0
    DOF[3] = 0
    DOF[4] = 0
    
    DOF[5] = DOF[5]*45
    
    return DOF

## Main Run##
def main():
    scaphoid = bones[:,:,:,0]
    P = point_transformation(scaphoid)
    
    num_trials = 100
    
    Error = np.zeros((num_trials,6))
    
    for i in range(num_trials):
    
        print(i)
    
        DOF = rand_DOF(6)
        
        print(DOF)
    
        trans = (DOF[0],DOF[1],DOF[2])
        rotation = (DOF[3],DOF[4],DOF[5])
    
        dynamic_slice,dynamic_slice_canny = test_slice(scaphoid,P,trans,rotation)
        
        DOF_calc = genetic_algorithm(scaphoid,P,100,50,dynamic_slice,np.array(trans+rotation))
        
        Error[i][0] = DOF_calc[0]-DOF[0]
        Error[i][1] = DOF_calc[1]-DOF[1]
        Error[i][2] = DOF_calc[2]-DOF[2]
        Error[i][3] = DOF_calc[3]-DOF[3]
        Error[i][4] = DOF_calc[4]-DOF[4]
        Error[i][5] = DOF_calc[5]-DOF[5]
        
        
    filehandler = open('scaph2.obj', 'w')
    pickle.dump(Error, filehandler)
    
    return 

if __name__ == '__main__':
    main()





