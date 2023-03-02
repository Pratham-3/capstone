import numpy as np

def hill_cipher(plaintext, key):
    # Convert plaintext to uppercase and remove spaces
    plaintext = plaintext.upper().replace(" ", "")
    # Calculate the size of the key matrix
    n = int(np.sqrt(len(key)))
    # Ensure that the plaintext length is a multiple of n
    plaintext += 'X' * (n - len(plaintext) % n)
    # Convert plaintext to numbers
    p_num = np.array([ord(c) - ord('A') for c in plaintext])
    # Reshape the plaintext to a matrix
    p_mat = np.reshape(p_num, (-1, n))
    # Compute the ciphertext as the product of the key and the plaintext matrix (modulo 26)
    c_mat = np.dot(p_mat, key) % 26
    # Convert the ciphertext matrix to a string
    ciphertext = ''.join([chr(c + ord('A')) for c in c_mat.flatten()])
    return ciphertext



def hill_cipher_decrypt(ciphertext, key):
    # Calculate the inverse of the key matrix
    det = int(np.round(np.linalg.det(key)))
    inv_det = 0
    for i in range(26):
        if (i * det) % 26 == 1:
            inv_det = i
            break
    key_inv = np.round(inv_det * np.linalg.inv(key)).astype(int) % 26

    # Convert ciphertext to numbers
    n = len(key)
    c_num = [ord(c) - ord('A') for c in ciphertext]
    c_num = np.array(c_num).reshape((-1, n))

    # Decrypt using matrix multiplication
    p_num = np.dot(c_num, key_inv) % 26
    plaintext = ''.join([chr(c + ord('A')) for c in p_num.flatten()])
    return plaintext.rstrip('X')


# Define the key matrix (in this case, a 2x2 matrix)
key = np.array([5, 17, 0, 21])
ciphertext = hill_cipher("HELLO", key)
print(ciphertext)

# Decrypt the ciphertext using the same key
plaintext = hill_cipher_decrypt(ciphertext, key)

# Print the result
print(plaintext)

