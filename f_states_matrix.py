import numpy as np

# This program will create a rate matrix for doubly excited states based on the equation:
# k^f_{ij;pq} = (k_{jq}^{EET}𝛿_{ip}+k_{ip}^{EET}𝛿_{jq}+k_{jp}^{EET}𝛿_{iq}+k_{iq}^{EET}𝛿_{jp})/(1+𝛿_{pq})

# Move here the dimension of the "input" matrix
N = 2

# Make a list with the possible f-states
indices = []
for i in range(1,N+1):
    for j in range(i,N+1):
        indices.append((i,j))

# Create an empty matrix of strings
Matrix_dim = len(indices)
matrix = np.zeros((Matrix_dim,Matrix_dim),dtype=np.dtype('U20'))

# Matrix creation
for i in range(Matrix_dim):
    for j in range(Matrix_dim):
        To_indices = indices[i] # Save the resulting f-state for future reference
        From_indices = indices[j] # Also the starting f-state
        indices_set = {To_indices[0], To_indices[1], From_indices[0], From_indices[1]} # Create a set to get all different numbers from f-states
        temp = 0
        if i != j: # Don't do anything on the diagonal
            # In each of these we decide if we put something on the matrix or not
            # In the comments we reference i,j,p,q from the equation (not the ones in the code)
            if From_indices[0] == To_indices[0]: # If i == p
                temp_string = f'k({From_indices[1]},{To_indices[1]})' # Resulting k(j,q) string
                matrix[i,j] += temp_string # Add it to the matrix
                temp += 1
            if From_indices[0] == To_indices[1]: # If i == q
                if temp > 0:
                    matrix[i,j] += ' + ' # Add a plus sign if we already added a value before
                temp_string = f'k({From_indices[1]},{To_indices[0]})' # Resulting k(j,p)
                matrix[i,j] += temp_string
                temp += 1
            if From_indices[1] == To_indices[0]: # If j == p
                if temp > 0:
                    matrix[i,j] += ' + '
                temp_string = f'k({From_indices[0]},{To_indices[1]})' # Resulting k(i,q)
                matrix[i,j] += temp_string
                temp += 1
            if From_indices[1] == To_indices[1]: # If j == q
                if temp > 0:
                    matrix[i,j] += ' + '
                temp_string = f'k({From_indices[0]},{To_indices[0]})' # Resulting k(i,p)
                matrix[i,j] += temp_string
                temp += 1

            # Now, if we added two values and we are dealing with a resulting double f-state
            if temp == 2 and To_indices[0] == To_indices[1]:
                # Then we want to only show that one rate is applied instead of two
                temp_indices = sorted([indices_set.pop(),indices_set.pop()]) # Because the set will only have two values stored, we get them in a list
                matrix[i,j] = f'k({temp_indices[0]},{temp_indices[1]})' # And those values are the ones which will get into the matrix

# Save the matrix to a csv file
np.savetxt('Matrix.csv',matrix, delimiter='|', fmt = '%s')

