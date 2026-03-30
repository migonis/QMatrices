import numpy as np
from numpy import linalg as LA
from scipy.linalg import expm, sinm, cosm, eig
import math
import itertools
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def decayMatrix(N, K_input, KR, KNR):
    ' Decay matrix '
    ' Create (N+2)x(N+2) decay matrices'
    K_decay = np.zeros([N + 2, N + 2])
    K_R = np.ones(N) * KR  # If all segments get the same value
    K_NR = np.ones(N) * KNR  # If all segments get the same value
    # K_R = np.arange(1, N + 1) * 10  # Radiative decay rates
    # K_NR = np.arange(1, N + 1) * 100  # Non-Radiative decay rates (10x bigger)
    
    # Copy rate matrix to decay matrix, fill in R/NR decay rates
    K_decay[0:N, 0:N] = K_input
    K_decay[N, 0:N] = K_NR  # Non-Radiative decay rates
    K_decay[N + 1, 0:N] = K_R  # Radiative decay rates (10x bigger)

    # Subtract R/NR decay rates from diagonal
    K_decay[0:N, 0:N] -= np.diag(K_NR)
    K_decay[0:N, 0:N] -= np.diag(K_R)
    return K_NR, K_R, K_decay

def annihilationA0(f_states, g_states, KA0):
    ' Double annihilation A0 matrix '
    K_A0 = np.zeros((g_states, f_states))
    K_A0[0, :] = np.ones(f_states) * KA0  # If all segments get the same value
    #K_A0[0, :] = np.arange(1, f_states + 1) * 0.1
    return K_A0

def annihilationA1(N, c_w_r, f_states, KA1):
    'Single annihilation A1 matrix'
    K_A1 = np.zeros([N, f_states])
    for i in range(f_states):
        if c_w_r[i][0] == c_w_r[i][1]:
            K_A1[c_w_r[i][0]][i] += KA1
    #        for j in range(len(c_w_r)):
    #            K_A1[i,j] += np.sum(c_w_r[j].count(i))
    return K_A1

def NRmatrix(N, c_w_r, K_NR):
    ' Non-Radiative matrix for each f state '
    'Doubly excited state e.g. 00 goes to 0NR with probability 2*(0->NR)'
    'Doubly excited state e.g. 01 goes to 0NR with probability (1->NR)'
    K_NRf = np.zeros([N, f_states])
    for i in range(f_states):
        for j in range(2):
            K_NRf[c_w_r[i][j - 1], i] += K_NR[c_w_r[i][j]]
    return K_NRf

def Rmatrix(N, c_w_r, K_R, f_states):
    'Radiative matrix for each f state'
    K_Rf = np.zeros([N, f_states])
    for i in range(f_states):
        for j in range(2):
            K_Rf[c_w_r[i][j - 1], i] += K_R[c_w_r[i][j]]
    return K_Rf

def doublyMatrix(c_w_r, K_input):
    'Doubly excited f states matrix '
    mati = np.zeros([len(c_w_r), len(c_w_r)])
    for To in range(mati.shape[0]):
        for From in range(mati.shape[1]):
            if To == From:
                continue
            for i in range(2):
                for j in range(2):
                    if c_w_r[To][i] == c_w_r[From][j]:
                        ind_To, ind_From = c_w_r[To][i - 1], c_w_r[From][j - 1]
                        if not (c_w_r[To][i] == c_w_r[To][i - 1] and i == 1):
                            mati[To, From] += K_input[ind_To, ind_From]
    return mati

def FDMatrix(f_states, e_states, g_states, K_input, K_NR, K_R, K_A0, K_A1, K_NRf, K_Rf, mati):
    'Puzzle out FD matrix'
    K_FD = np.zeros([f_states + e_states + g_states, f_states + e_states + g_states])
    # Add NR decay from e states+NR (to NR NR)
    K_FD[f_states + e_states, f_states:f_states + N] = K_NR
    # Add NR decay from e states+R
    K_FD[f_states + e_states + 1, f_states + N:f_states + 2 * N] = K_NR
    # Add R decay from e states+NR
    K_FD[f_states + e_states + 1, f_states:f_states + N] = K_R
    # Add R decay from e states+R
    K_FD[f_states + e_states + 2, f_states + N:f_states + 2 * N] = K_R
    # Add e states from transferMatrix
    K_FD[f_states:f_states + N, f_states:f_states + N] = K_input[0:N, 0:N]
    K_FD[f_states + N:f_states + 2 * N, f_states + N:f_states + 2 * N] = K_input[0:N, 0:N]
    # Add double annihilation
    K_FD[f_states + e_states:f_states + e_states + g_states, 0:f_states] += K_A0
    # Add single annihilation
    K_FD[f_states:f_states + e_states // 2, 0:f_states] += K_A1
    # Add non-radiative decay of f states
    K_FD[f_states:f_states + e_states // 2, 0:f_states] += K_NRf
    # Add radiative decay of f states
    K_FD[f_states + e_states // 2:f_states + e_states, 0:f_states] += K_Rf
    # Add f states matrix
    K_FD[0:f_states, 0:f_states] = mati
    # Subtract K_R(NR) from diagonal
    K_FD[f_states:f_states + N, f_states:f_states + N] -= np.diag(K_NR)
    K_FD[f_states + N:f_states + 2 * N, f_states + N:f_states + 2 * N] -= np.diag(K_NR)
    K_FD[f_states:f_states + N, f_states:f_states + N] -= np.diag(K_R)
    K_FD[f_states + N:f_states + 2 * N, f_states + N:f_states + 2 * N] -= np.diag(K_R)
    # Subtract f states from diagonal
    for i in range(f_states):  # Only on the diagonal, subtract all other elements per column
        K_FD[i, i] -= np.sum(K_FD[:, i])
    return K_FD

def diagonalize_matrix(matrix):
    'Matrix diagonalization'
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

def is_time_infinite(K_FD, eigenvectors, D, tol=1e-10, N_iter=40000, step=100):
    'Long detection times, detection time is infinite'
    for t in range(10, N_iter, step):
        Product = expm(K_FD * t)
        sumRNR = sum(Product[f_states + e_states:f_states + e_states + g_states, 0])
        print("sumRNR is : ", sumRNR)
        if abs(sumRNR - 1) < tol:
            return t, Product
    print("The search did not converge!")
    print("Please, increase N_iter.")
    print("sumRNR is : ", sumRNR)
    exit(0)

def time_resolved(K_FD, t4, step=100):
    'One specific detection time'
    for t in np.linspace(0, t4, step):
        Product = expm(K_FD * t)
        sumRNR = sum(Product[f_states + e_states:f_states + e_states + g_states, 0])
        print("sumRNR is : ", sumRNR)
    return t, Product

def getQs(K_FD,it,N):
    Prod = expm(K_FD * it)
    roundProd = np.round(Prod, 5)
    Q2 = np.zeros(f_states)
    Q1 = np.zeros(N)
    Radiative_states = np.ones(N+3)# /(N+1)
    Radiative_states[-1] = 2
    Radiative_states[-3] = 0
    for i in range(f_states):
        Q2[i] += np.dot(Prod[-(N+3):, i], Radiative_states)
        #Q2[i] += np.dot(Prod[-3:, i], np.array([0, 1, 0])) + 2 * np.dot(Prod[-3:, i], np.array([0, 0, 1]))
    for i in range(N):
        Q1[i] += np.dot(Prod[-(N+3):, range(f_states, f_states + N)[i]], Radiative_states)
    return Prod, Q1, Q2

# Input values
system = 'LH2'  # 'AlphaBetaTrimer'
# Current directory (change if necessary by pwd)
path = '/scratch/p317440/Branch_Testing/Systems/FD-LH2/TimeResolution/' 
#path = '/Users/stephanie/Documents/GitHub/Current_NISE_OD/NISE_Tutorials/0QF/'  # Local
# Generate the filename with the system name or change name to filename
# filename = f'{path}{system}_transfer_matrix.dat'
filename = 'RateMatrix.dat'
# Read the matrix from the .dat file
K_input = np.loadtxt(filename)
N = len(K_input)

# Conversion Factors
ns2ps = 1e3
fs2ps = 1e-3

# Lifetimes from literature
tauF = 986  # 986 ps
tauR = 10*ns2ps  # 10 ns
tauA1 = 0.59  # 0.59 ps

# Compute Rates
kF = 1/tauF
kR = 1/tauR
kA1 = 1/tauA1
kNR = kF-kR  # kF = kR + kNR

# Input values for annihilation and decay
KA0 = 0  # Annihilation rate when no excitons are left behind
KA1 = kA1  # Annihilation rate leaves one exciton behind
KR = kR  # Radiative rate
KNR = kNR  # Non-radiative rate
tol = 1e-6
N_iter = 20000
# step = 1000

# Detection time
t4 = 0.1  # in ps
steps = 10

# Find the segment configurations corresponding to each 
# double excited state.
# c_w_r[f_state_number][site_number(0/1)] returns the 
# number of the excited segment in that site number
c_w_r = list(itertools.combinations_with_replacement(range(N), 2))

# All states
f_states = N * (N + 1) // 2
e_states = 2 * N
g_states = 3

# Construct K_FD from sub-matrices
K_NR, K_R, K_decay = decayMatrix(N, K_input, KR, KNR)
K_A0 = annihilationA0(f_states, g_states, KA0)
K_A1 = annihilationA1(N, c_w_r, f_states, KA1)
K_NRf = NRmatrix(N, c_w_r, K_NR)
K_Rf = Rmatrix(N, c_w_r, K_R, f_states)
mati = doublyMatrix(c_w_r, K_input)
K_FD = FDMatrix(f_states, e_states, g_states, K_input, K_NR, K_R, K_A0, K_A1, K_NRf, K_Rf, mati)

#########################################
# eigenvalues, eigenvectors, D, A_reconstructed = diagonalize_matrix(K_FD) # We don't need to diagonalize now.
######### Now compute Q1 and Q2 #########

Prod, Q1, Q2 = getQs(K_FD,t4,N)
np.savetxt('Q1s_TR_func.dat', Q1, delimiter=' ')
np.savetxt('Q2s_TR_func.dat', Q2, delimiter=' ')
np.savetxt('expK_FD_t4_func.dat', Prod, delimiter=' ')

##########################################
# All detection times to be considered
t4s = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 
       1., 2., 5., 10., 20., 50., 100., 200., 500., 1000.,
       2000., 5000., 10000., 20000., 50000., 100000.]
Nt4 = len(t4s) # How many detection times are there.
emptyQs = np.zeros((1+len(Q1)+len(Q2),len(t4s))) # Create empty array for storage

for i in range(len(t4s)):
    # Here we will store the Q1s and Q2s corresponding to each t4
    _, Q1t, Q2t = getQs(K_FD,t4s[i],N)
    emptyQs[0,i]            = t4s[i]
    emptyQs[1:len(Q1t)+1,i] = Q1t
    emptyQs[len(Q1t)+1:,i]  = Q2t
print(f'lenQ1 = {len(Q1)}')
np.savetxt('ManyQs.dat', emptyQs, delimiter=',')
'''
try:
    # Calculate Q factors at infinite time.
    it, Prod = is_time_infinite(K_FD, eigenvectors, D, tol, N_iter, step)
    Prod = expm(K_FD * it)
    roundProd = np.round(Prod, 5)
    Q2 = np.zeros(f_states)
    Q1 = np.zeros(N)
    Radiative_states = np.ones(N+3)# /(N+1)
    Radiative_states[-1] = 2
    Radiative_states[-3] = 0
    for i in range(f_states):
        Q2[i] += np.dot(Prod[-(N+3):, i], Radiative_states)
        #Q2[i] += np.dot(Prod[-3:, i], np.array([0, 1, 0])) + 2 * np.dot(Prod[-3:, i], np.array([0, 0, 1]))
    for i in range(N):
        Q1[i] += np.dot(Prod[-(N+3):, range(f_states, f_states + N)[i]], Radiative_states)

    np.savetxt('Q1s_TR.dat', Q1, delimiter=' ')
    np.savetxt('Q2s_TR.dat', Q2, delimiter=' ')
    np.savetxt('expK_FD_t4.dat', Prod, delimiter=' ')
    f = open('QFactors_Log.txt', 'a')
    f.write(
        f'For detection time t4 = {t4} ps:\n'
        f'For system {system} with N={int(N)} segments and rates R={round(KR, 5)}, NR={round(KNR, 5)}, A0={round(KA0, 5)}, and A1={round(KA1, 5)},\n'
        f'Q1:{Q1}\n'f'Q2:{Q2}\n')
    f.close()

except ValueError as err:
    print('ValueError:', err)

'''
np.set_printoptions(formatter={'float_kind': '{:6.1f}'.format})
print('Quick overview of calculated parameters:')
print('K_FD:')
print(K_FD)
print('t4:', t4, 'ps')
print('A1:', KA1, ', A0:', KA0)
print('kNR:', KNR, ', kR:', KR)
print('Q1:', Q1)
print('Q2:', Q2)
