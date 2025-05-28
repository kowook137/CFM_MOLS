import numpy as np
from typing import Tuple, Set
import itertools

def orthogonality(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> float:
    """
    Calculate the orthogonality of three Latin squares A, B, C.
    
    For three n×n Latin squares to be mutually orthogonal (MOLS), 
    when superimposed, each ordered triple (a,b,c) should appear exactly once.
    
    Orthogonality = (number of unique triples) / (total possible triples)
    Perfect orthogonality = 1.0, target threshold = 0.97
    
    Args:
        A, B, C: Three n×n numpy arrays representing Latin squares
        
    Returns:
        Float in [0,1] representing orthogonality measure
    """
    # Verify input shapes
    if not (A.shape == B.shape == C.shape):
        raise ValueError("All three Latin squares must have the same shape")
    
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Latin squares must be square matrices")
    
    n = A.shape[0]
    
    # Create triples by superimposing the three squares
    triples = set()
    for i in range(n):
        for j in range(n):
            triple = (A[i, j], B[i, j], C[i, j])
            triples.add(triple)
    
    # Calculate orthogonality
    num_unique_triples = len(triples)
    total_possible_triples = n * n  # For perfect MOLS, should equal n²
    
    orthogonality_score = num_unique_triples / total_possible_triples
    
    return orthogonality_score

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

def generate_test_latin_squares(n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate three test Latin squares for testing orthogonality function.
    
    Args:
        n: Size of the Latin squares
        
    Returns:
        Tuple of three n×n Latin squares
    """
    # Generate first Latin square (basic pattern)
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            A[i, j] = (i + j) % n
    
    # Generate second Latin square (row shift pattern)
    B = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            B[i, j] = (i + j + 1) % n
    
    # Generate third Latin square (different row shift pattern)
    C = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            C[i, j] = (i + j + 2) % n
    
    return A, B, C

def test_orthogonality_function():
    """
    Unit tests for the orthogonality function.
    """
    print("Running orthogonality function tests...")
    
    # Test 1: Perfect orthogonal case (small example)
    print("\nTest 1: Small perfect orthogonal case")
    A = np.array([[0, 1], [1, 0]])
    B = np.array([[0, 1], [1, 0]]) 
    C = np.array([[1, 0], [0, 1]])
    
    orth_score = orthogonality(A, B, C)
    print(f"Orthogonality score: {orth_score:.4f}")
    
    # Test 2: Non-orthogonal case (identical squares)
    print("\nTest 2: Non-orthogonal case (identical squares)")
    A = np.array([[0, 1], [1, 0]])
    B = np.array([[0, 1], [1, 0]])
    C = np.array([[0, 1], [1, 0]])
    
    orth_score = orthogonality(A, B, C)
    print(f"Orthogonality score: {orth_score:.4f}")
    
    # Test 3: Larger Latin squares
    print("\nTest 3: 4×4 Latin squares")
    A, B, C = generate_test_latin_squares(4)
    
    print("Square A:")
    print(A)
    print("Square B:")
    print(B)
    print("Square C:")
    print(C)
    
    # Verify they are valid Latin squares
    print(f"A is valid Latin square: {verify_latin_square(A)}")
    print(f"B is valid Latin square: {verify_latin_square(B)}")
    print(f"C is valid Latin square: {verify_latin_square(C)}")
    
    orth_score = orthogonality(A, B, C)
    print(f"Orthogonality score: {orth_score:.4f}")
    
    # Test 4: 10×10 Latin squares (target size)
    print("\nTest 4: 10×10 Latin squares")
    A, B, C = generate_test_latin_squares(10)
    
    print(f"A is valid Latin square: {verify_latin_square(A)}")
    print(f"B is valid Latin square: {verify_latin_square(B)}")
    print(f"C is valid Latin square: {verify_latin_square(C)}")
    
    orth_score = orthogonality(A, B, C)
    print(f"Orthogonality score: {orth_score:.4f}")
    
    # Test 5: Error handling
    print("\nTest 5: Error handling")
    try:
        A = np.array([[0, 1], [1, 0]])
        B = np.array([[0, 1, 2], [1, 0, 2], [2, 2, 1]])  # Different size
        C = np.array([[0, 1], [1, 0]])
        orthogonality(A, B, C)
        print("ERROR: Should have raised ValueError")
    except ValueError as e:
        print(f"Correctly caught error: {e}")
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    test_orthogonality_function()