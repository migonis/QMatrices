import numpy as np
from numpy import linalg as LA
from scipy.linalg import expm, sinm, cosm, eig
import math
import itertools

N = 2 # N is the number of segments

# Find the segment configurations corresponding to each double excited state
# c_w_r[f_state_number][site_number(0/1)] returns the number of the excited segment in that site number
c_w_r = list(itertools.combinations_with_replacement(range(N), 2))

######## Input matrix ########

if N == 1:
    K_input = np.zeros([N,N]) # Exception for when N = 1
else: # Generate an N by N matrix with numbers from 1 to N^2
    K_input = np.reshape(np.arange(1,(N)**2+1),(N,N))
    # Alternatively, one could read the rate matrix from a text file
    # K_input = np.loadtxt("input.txt", dtype='i', delimiter=',')
    for i in range(N): # Only on the diagonal, subtract all other elements per column
        K_input[i,i] = -np.sum(K_input[:,i]) + K_input[i,i]

######## Decay matrix ########
# Create N+2xN+2 decay matrices
K_decay = np.zeros([N+2,N+2])

# Copy rate matrix to decay matrix, fill in R/NR decay rates
K_decay[0:N,0:N] = K_input
K_decay[N,0:N] = np.arange(1,N+1)*10 # Non-Radiative decay rates
K_decay[N+1,0:N] = np.arange(1,N+1)*100 # Radiative decay rates (10x bigger)

# Subtract R/NR decay rates from diagonal
for i in range(N):
    K_decay[i,i] -= K_decay[N,i]+K_decay[N+1,i]

K_R = np.arange(1,N+1)*100 # Radiative decay rates (10x bigger)
K_NR = np.arange(1,N+1)*10 # Non-Radiative decay rates

######## FD matrix ########
f_states = N*(N+1)//2
e_states = 2*N
g_states = 3
K_FD = np.zeros([f_states+e_states+g_states,f_states+e_states+g_states])

# Annihilation
#K_A0 = np.reshape(np.arange(1,g_states*f_states+1)*0.1,(g_states,f_states))
K_A0 = np.zeros((g_states,f_states))
K_A0[0,:] = np.arange(1,f_states+1)*0.1


K_A1 = np.zeros([N,len(c_w_r)])
for i in range(N):
    for j in range(len(c_w_r)):
        K_A1[i,j] += np.sum(c_w_r[j].count(i))

K_NRf = np.zeros([N,f_states])
for i in range(f_states):
    for j in range(2):
        K_NRf[c_w_r[i][j-1],i]+=K_NR[c_w_r[i][j]]

K_Rf = np.zeros([N,f_states])
for i in range(f_states):
    for j in range(2):
        K_Rf[c_w_r[i][j-1],i]+=K_R[c_w_r[i][j]]

mati = np.zeros([len(c_w_r), len(c_w_r)])
mati
#for row in range(mati.shape[0]):
#    for col in range(row, mati.shape[1]):  # solo en el triangulo superior
#        if row == col:
#            continue
#        repeatedValue = False
#        for i in range(2):
#            for j in range(2):
#                if c_w_r[row][i] == c_w_r[col][j]:
#                    repeatedValue = True
#                    ind_0, ind_1 = c_w_r[row][i - 1], c_w_r[col][j - 1]
#                    mati[row, col] += K_input[ind_0, ind_1]
#                    mati[col, row] += K_input[ind_1, ind_0]

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

#                    mati[col, row] += K_input[ind_1, ind_0]


print('mati')
print(mati)
print(K_input)

# Add NR decay from e states+NR (to NR NR)
K_FD[f_states+e_states,f_states:f_states+N] = K_NR
K_FD[f_states:f_states+N,f_states:f_states+N] -= np.diag(K_NR)
# Add NR decay from e states+R
K_FD[f_states+e_states+1,f_states+N:f_states+2*N] = K_NR
K_FD[f_states+N:f_states+2*N,f_states+N:f_states+2*N] -= np.diag(K_NR)
# Add R decay from e states+NR
K_FD[f_states+e_states+1,f_states:f_states+N] = K_R
K_FD[f_states:f_states+N,f_states:f_states+N] -= np.diag(K_R)
# Add R decay from e states+R
K_FD[f_states+e_states+2,f_states+N:f_states+2*N] = K_R
K_FD[f_states+N:f_states+2*N,f_states+N:f_states+2*N] -= np.diag(K_R)


K_FD[f_states:f_states+N,f_states:f_states+N] = K_decay[0:N,0:N]
K_FD[f_states+N:f_states+2*N,f_states+N:f_states+2*N] = K_decay[0:N,0:N]
#K_FD[f_states+e_states:f_states+e_states+2,f_states:f_states+N] = [K_decay[N,0:N],K_decay[N+1,0:N]]
#K_FD[f_states+e_states+1:f_states+e_states+g_states,f_states+N:f_states+2*N] = [K_decay[N,0:N],K_decay[N+1,0:N]]
# Add double annihilation
K_FD[f_states+e_states:f_states+e_states+g_states,0:f_states] += K_A0
# Add single annihilation
K_FD[f_states:f_states+e_states//2,0:f_states] += K_A1
# Add non-radiative decay of f states
K_FD[f_states:f_states+e_states//2,0:f_states] +=K_NRf
# Add radiative decay of f states
K_FD[f_states+e_states//2:f_states+e_states,0:f_states] +=K_Rf
#K_FD[f_states+e_states//2:f_states+e_states,0:f_states] += K_A1
K_FD[0:f_states,0:f_states] = mati

for i in range(f_states): # Only on the diagonal, subtract all other elements per column
    K_FD[i,i] = -np.sum(K_FD[:,i]) #+ K_FD[i,i]

print('Sum must be zero')
for i in range(f_states+e_states+g_states): # Quick check, print sum of each column. Sum must be zero
    print(np.round(abs(np.sum(K_FD[:,i]))))
'''
print('K_A0')
print(K_A0)
print('K_A1')
print(K_A1)
'''
np.set_printoptions(formatter={'float_kind':'{:1.2f}'.format})
print('K_FD')
print(K_FD)
print('K_input')
print(K_input)




