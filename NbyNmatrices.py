import numpy as np
from numpy import linalg as LA
from scipy.linalg import expm, sinm, cosm, eig
import math
import itertools

N = 2 # N is the number of segments

######## Input matrix ########

if N == 1:
    K_input = np.zeros([N,N]) # Exception for when N = 1
else: # Generate an N by N matrix of random numbers
    K_input = np.reshape(np.arange(1,(N)**2+1),(N,N))
    for i in range(N): # Only on the diagonal, subtract all other elements per column
        K_input[i,i] = -np.sum(K_input[:,i]) + K_input[i,i]

print('Sum must be zero')
for i in range(N): # Quick check, print sum of each column. Sum must be zero
    print(np.round(abs((np.sum(K_input[:,i])))))

######## Decay matrix ########
# Create N+2xN+2 decay matrices
# K_R_decay = np.zeros([N+2,N+2])
# K_NR_decay = np.zeros([N+2,N+2])
K_decay = np.zeros([N+2,N+2])

# Copy rate matrix to decay matrix, fill R/NR decay rates randomly
# K_NR_decay[0:N,0:N] = K_input
# K_NR_decay[N,0:N] = np.arange(1,N+1)*10

# K_R_decay[0:N,0:N] = K_input
# K_R_decay[N+1,0:N] = np.arange(1,N+1)*100

K_decay[0:N,0:N] = K_input
K_decay[N,0:N] = np.arange(1,N+1)*10
K_decay[N+1,0:N] = np.arange(1,N+1)*100

# Subtract R/NR decay rates from diagonal
# for i in range(N):
#     K_NR_decay[i,i] -= K_NR_decay[N,i]
#     K_R_decay[i,i] -= K_R_decay[N+1,i]
for i in range(N):
    K_decay[i,i] -= K_decay[N,i]+K_decay[N+1,i]

print('K_decay')
print(K_decay) # Print the whole matrix

for i in range(N): # Sum of each column must be zero
    print(np.round(abs((np.sum(K_input[:, i])))))

######## FD matrix ########
f_states = N*(N+1)//2
e_states = 2*N
g_states = 3
K_FD = np.zeros([f_states+e_states+g_states,f_states+e_states+g_states])

# Annihilation
K_A0 = np.reshape(np.arange(1,g_states*f_states+1)*0.1,(g_states,f_states))
c_w_r = list(itertools.combinations_with_replacement(range(N), 2))
K_A1 = np.zeros([N,len(c_w_r)])
for i in range(N):
    for j in range(len(c_w_r)):
        K_A1[i,j] = np.sum(c_w_r[j].count(i))

mati = np.zeros([len(c_w_r), len(c_w_r)])
mati
for row in range(mati.shape[0]):
    for col in range(row, mati.shape[1]):  # solo en el triangulo superior
        if row == col:
            continue
        repeatedValue = False
        for i in range(2):
            for j in range(2):
                if c_w_r[row][i] == c_w_r[col][j]:
                    repeatedValue = True
                    ind_0, ind_1 = c_w_r[row][i - 1], c_w_r[col][j - 1]
                    mati[row, col] = K_input[ind_0, ind_1]
                    mati[col, row] = K_input[ind_1, ind_0]

print('mati')
print(mati)

K_FD[f_states:f_states+N,f_states:f_states+N] = K_decay[0:N,0:N]
K_FD[f_states+N:f_states+2*N,f_states+N:f_states+2*N] = K_decay[0:N,0:N]
K_FD[f_states+e_states:f_states+e_states+2,f_states:f_states+N] = [K_decay[N,0:N],K_decay[N+1,0:N]]
K_FD[f_states+e_states+1:f_states+e_states+g_states,f_states+N:f_states+2*N] = [K_decay[N,0:N],K_decay[N+1,0:N]]
K_FD[f_states+e_states:f_states+e_states+g_states,0:f_states] = K_A0
K_FD[f_states:f_states+e_states//2,0:f_states] = K_A1
K_FD[f_states+e_states//2:f_states+e_states,0:f_states] = K_A1
K_FD[0:f_states,0:f_states] = mati

print('K_A0')
print(K_A0)
print('K_A1')
print(K_A1)
np.set_printoptions(formatter={'float_kind':'{:1.2f}'.format})
print('K_FD')
print(K_FD)
print('K_input')
print(K_input)




