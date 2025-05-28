import numpy as np
import random
from typing import List, Tuple

def generate_latin_square(n: int) -> np.ndarray:
    """
    Generate a random n×n Latin square using Knuth shuffle method.
    
    Args:
        n: Size of the Latin square
        
    Returns:
        n×n numpy array representing a Latin square
    """
    # Start with a basic Latin square (row i, column j has value (i+j) % n)
    square = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            square[i, j] = (i + j) % n
    
    # Apply random permutations to rows, columns, and symbols
    # Permute rows
    row_perm = list(range(n))
    random.shuffle(row_perm)
    square = square[row_perm, :]
    
    # Permute columns
    col_perm = list(range(n))
    random.shuffle(col_perm)
    square = square[:, col_perm]
    
    # Permute symbols
    symbol_perm = list(range(n))
    random.shuffle(symbol_perm)
    for i in range(n):
        for j in range(n):
            square[i, j] = symbol_perm[square[i, j]]
    
    return square

def verify_latin_square(square: np.ndarray) -> bool:
    """
    Verify that a square is a valid Latin square.
    
    Args:
        square: n×n numpy array to verify
        
    Returns:
        True if valid Latin square, False otherwise
    """
    n = square.shape[0]
    expected_set = set(range(n))
    
    # Check rows
    for i in range(n):
        if set(square[i, :]) != expected_set:
            return False
    
    # Check columns
    for j in range(n):
        if set(square[:, j]) != expected_set:
            return False
    
    return True

def generate_latin_squares_dataset(num_squares: int, size: int = 10) -> np.ndarray:
    """
    Generate a dataset of random Latin squares.
    
    Args:
        num_squares: Number of Latin squares to generate
        size: Size of each Latin square (default: 10)
        
    Returns:
        numpy array of shape (num_squares, size, size)
    """
    print(f"Generating {num_squares} random {size}×{size} Latin squares...")
    
    squares = np.zeros((num_squares, size, size), dtype=int)
    
    for i in range(num_squares):
        if (i + 1) % 10000 == 0:
            print(f"Generated {i + 1}/{num_squares} squares...")
        
        # Generate a valid Latin square
        while True:
            square = generate_latin_square(size)
            if verify_latin_square(square):
                squares[i] = square
                break
    
    print(f"Successfully generated {num_squares} Latin squares!")
    return squares

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Generate 100,000 random 10×10 Latin squares
    num_squares = 100000
    size = 10
    
    print("Starting Latin square generation...")
    latin_squares = generate_latin_squares_dataset(num_squares, size)
    
    # Save to file
    output_file = "latins.npy"
    np.save(output_file, latin_squares)
    
    print(f"Saved {num_squares} Latin squares to {output_file}")
    print(f"Dataset shape: {latin_squares.shape}")
    print(f"File size: {latin_squares.nbytes / (1024**2):.2f} MB")
    
    # Verify a few random squares
    print("\nVerifying random samples...")
    for i in range(5):
        idx = random.randint(0, num_squares - 1)
        is_valid = verify_latin_square(latin_squares[idx])
        print(f"Square {idx}: {'Valid' if is_valid else 'Invalid'}")