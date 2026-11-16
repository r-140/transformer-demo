"""Numerical representation is deliberately distinct from token generation."""
from data import validate,transpose

def numeric_example(matrix,suite='fixed'):
    r,c=validate(matrix)
    if suite=='fixed':
        if (r,c)!=(3,3):raise ValueError('Fixed experiment accepts only3x3')
        return [v/9 for row in matrix for v in row],[v/9 for row in transpose(matrix) for v in row],[1]*9,[2,2]
    # Packed row-major, not a spatial padded grid: transpose depends on shape.
    values=[v/9 for row in matrix for v in row];mask=[1]*len(values)+[0]*(16-len(values))
    dims=[int(r==i) for i in range(1,5)]+[int(c==i) for i in range(1,5)]
    target=[v/9 for row in transpose(matrix) for v in row]
    return values+[0]*(16-len(values))+mask+dims,target+[0]*(16-len(target)),mask,[c-1,r-1]

def numeric_decode(values,shape,suite='fixed'):
    import math
    if suite=='fixed':r=c=3
    else:r,c=shape
    if not 1<=r<=4 or not 1<=c<=4:return None
    selected=values[:r*c]
    if len(selected)!=r*c or any(not math.isfinite(v) for v in selected):return None
    digits=[round(v*9) for v in selected]
    # Ordinary regression rounding; never clamp to valid digits or repair values.
    if any(not 0<=v<=9 for v in digits):return None
    return [digits[i*c:(i+1)*c] for i in range(r)]

def permutation_indices(rows,cols):return [i*cols+j for j in range(cols) for i in range(rows)]
