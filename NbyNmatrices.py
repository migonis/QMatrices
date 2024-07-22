import numpy as np
from numpy import linalg as LA
from scipy.linalg import expm, sinm, cosm, eig
import math
import itertools

################ Input matrix ################
# Define (or read from file) transfer matrix
def transferMatrix(N):
    K_input = np.array([[-10., 0.],
       [10., 0.]])#[[-1. 0.], [1. 0.]]
    # Alternatively, one could read the rate matrix from a text file
    # K_input = np.loadtxt("input.txt", dtype='i', delimiter=',')
    # If reading from file, comment if statement:
    '''
    if N == 1:
        K_input = np.zeros([N,N]) # Exception for when N = 1
    else: # Generate an N by N matrix with numbers from 1 to N^2
        K_input = np.reshape(np.arange(1,(N)**2 + 1),(N,N))
        for i in range(N): # On diagonal, subtract column elements
            K_input[i,i] = -np.sum(K_input[:,i]) + K_input[i,i]
            '''
    return K_input

################ Decay matrix ################
# Create N+2xN+2 decay matrices
def decayMatrix(N, K_input):
    K_decay = np.zeros([N + 2, N + 2])
    K_R = np.ones(N) * 0.1 # If all segments get the same value
    K_NR = np.ones(N) * 1.0 # If all segments get the same value
    '''
    #K_R = np.arange(1, N + 1) * 10  # Radiative decay rates
    #K_NR = np.arange(1, N + 1) * 100  # Non-Radiative decay rates (10x bigger)
    '''

    # Copy rate matrix to decay matrix, fill in R/NR decay rates
    K_decay[0:N, 0:N] = K_input
    K_decay[N, 0:N] = K_NR  # Non-Radiative decay rates
    K_decay[N + 1, 0:N] = K_R # Radiative decay rates (10x bigger)

    # Subtract R/NR decay rates from diagonal
    K_decay[0:N,0:N] -= np.diag(K_NR)
    K_decay[0:N,0:N] -= np.diag(K_R)
    return K_NR, K_R, K_decay

################ Double annihilation A0 matrix ################
def annihilationA0(f_states,g_states):
    K_A0 = np.zeros((g_states, f_states))
    K_A0[0, :] = np.ones(f_states) * 0 # If all segments get the same value
    '''
    #K_A0[0, :] = np.arange(1, f_states + 1) * 0.1
    '''
    return K_A0

################ Single annihilation A1 matrix ################
def annihilationA1(N,c_w_r,f_states):
    K_A1 = np.zeros([N,f_states])
    for i in range(f_states):
        if c_w_r[i][0]==c_w_r[i][1]:
           K_A1[c_w_r[i][0]][i] += 3
#        for j in range(len(c_w_r)):
#            K_A1[i,j] += np.sum(c_w_r[j].count(i))
    return K_A1

################ Non-Radiative matrix for each f state ################
# Doubly excited state e.g. 00 goes to 0NR with probability 2*(0->NR)
# Doubly excited state e.g. 01 goes to 0NR with probability (1->NR)
def NRmatrix(N,c_w_r,K_NR):
    K_NRf = np.zeros([N, f_states])
    for i in range(f_states):
        for j in range(2):
            K_NRf[c_w_r[i][j - 1], i] += K_NR[c_w_r[i][j]]
    return K_NRf

################ Radiative matrix for each f state ################
def Rmatrix(N,c_w_r,K_R,f_states):
    K_Rf = np.zeros([N, f_states])
    for i in range(f_states):
        for j in range(2):
            K_Rf[c_w_r[i][j - 1], i] += K_R[c_w_r[i][j]]
    return K_Rf

################ Doubly excited f states matrix ################
def doublyMatrix(c_w_r,K_input):
    mati = np.zeros([len(c_w_r), len(c_w_r)])
    for To in range(mati.shape[0]):
        for From in range(mati.shape[1]):
            if To == From:
                continue
            for i in range(2):
                for j in range(2):
                    if c_w_r[To][i] == c_w_r[From][j]:
                        ind_To, ind_From = c_w_r[To][i - 1], c_w_r[From][j - 1]
                        if not(c_w_r[To][i]==c_w_r[To][i-1] and i==1):
                            mati[To, From] += K_input[ind_To, ind_From]
    return mati

################ Puzzle out FD matrix ################
def FDMatrix(f_states,e_states,g_states,K_input,K_NR,K_R,K_A0,K_A1,K_NRf,K_Rf,mati):
    K_FD = np.zeros([f_states+e_states+g_states,f_states+e_states+g_states])
    # Add NR decay from e states+NR (to NR NR)
    K_FD[f_states+e_states,f_states:f_states+N] = K_NR
    # Add NR decay from e states+R
    K_FD[f_states+e_states+1,f_states+N:f_states+2*N] = K_NR
    # Add R decay from e states+NR
    K_FD[f_states+e_states+1,f_states:f_states+N] = K_R
    # Add R decay from e states+R
    K_FD[f_states+e_states+2,f_states+N:f_states+2*N] = K_R
    # Add e states from transferMatrix
    K_FD[f_states:f_states+N,f_states:f_states+N] = K_input[0:N,0:N]
    K_FD[f_states+N:f_states+2*N,f_states+N:f_states+2*N] = K_input[0:N,0:N]
    # Add double annihilation
    K_FD[f_states+e_states:f_states+e_states+g_states,0:f_states] += K_A0
    # Add single annihilation
    K_FD[f_states:f_states+e_states//2, 0:f_states] += K_A1
    # Add non-radiative decay of f states
    K_FD[f_states:f_states+e_states//2,0:f_states] += K_NRf ########################################################## Just to fit with vivek.py
    '''
    K_FD[f_states:f_states+e_states//2,0:f_states] += K_NRf
    '''
    # Add radiative decay of f states
    K_FD[f_states+e_states//2:f_states+e_states,0:f_states] += K_Rf
    # Add f states matrix
    K_FD[0:f_states,0:f_states] = mati
    # Subtract K_R(NR) from diagonal
    K_FD[f_states:f_states+N,f_states:f_states+N] -= np.diag(K_NR)
    K_FD[f_states+N:f_states+2*N,f_states+N:f_states+2*N] -= np.diag(K_NR)
    K_FD[f_states:f_states+N,f_states:f_states+N] -= np.diag(K_R)
    K_FD[f_states+N:f_states+2*N,f_states+N:f_states+2*N] -= np.diag(K_R)
    # Subtract f states from diagonal
    for i in range(f_states):  # Only on the diagonal, subtract all other elements per column
        K_FD[i, i] -= np.sum(K_FD[:, i])
    return K_FD

N = 2 # N is the number of segments

# Find the segment configurations corresponding to each double excited state
# c_w_r[f_state_number][site_number(0/1)] returns the number of the excited segment in that site number
c_w_r = list(itertools.combinations_with_replacement(range(N), 2))

# All states 
f_states = N*(N+1)//2
e_states = 2*N
g_states = 3

# Construct K_FD from sub-matrices
K_input = transferMatrix(N)
K_NR, K_R, K_decay = decayMatrix(N, K_input)
K_A0 = annihilationA0(f_states,g_states)
K_A1 = annihilationA1(N,c_w_r,f_states)
K_NRf = NRmatrix(N,c_w_r,K_NR)
K_Rf = Rmatrix(N,c_w_r,K_R,f_states)
mati = doublyMatrix(c_w_r,K_input)
K_FD = FDMatrix(f_states,e_states,g_states,K_input,K_NR,K_R,K_A0,K_A1,K_NRf,K_Rf,mati)

np.set_printoptions(formatter={'float_kind':'{:6.1f}'.format})
print('K_FD')
print(K_FD)

print('Sum must be zero')
for i in range(f_states+e_states+g_states): # Quick check, print sum of each column. Sum must be zero
    print(np.round(abs(np.sum(K_FD[:,i]))))

########################################################################################################################

# Matrix diagonalization
def diagonalize_matrix(matrix):
    # Ensure the input is a numpy array
    matrix = np.array(matrix)

    # Check if the matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square")

    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = LA.eig(matrix)

    # Diagonal matrix of eigenvalues
    D = np.diag(eigenvalues)

    # Verify the diagonalization: matrix should be close to eigenvectors * D * inv(eigenvectors)
    A_reconstructed = eigenvectors @ D @ LA.inv(eigenvectors)

    return eigenvalues, eigenvectors, D, A_reconstructed


eigenvalues, eigenvectors, D, A_reconstructed = diagonalize_matrix(K_FD)

print("Original Matrix:")
print(K_FD)
print("\nEigenvalues:")
print(np.round(eigenvalues, 5))
print("\nEigenvectors:")
print(np.round(eigenvectors, 5))
print('\n Eigenv-1')
print(np.round(np.linalg.inv(eigenvectors), 5))
print("\nDiagonal Matrix:")
print(np.round(D, 5))
print("\nReconstructed Matrix (should be close to the original):")
print(np.round(A_reconstructed, 5))

#####
timet = 200
Product = eigenvectors @ expm(D*timet) @ LA.inv(eigenvectors)
print('\n exp at infinite time:')
print(np.round(Product,5))