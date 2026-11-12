# Matrix multiplication refresher: 2x3 times 3x4

This is an exact mathematical exercise, not a neural-training result.
For A of shape m x n and B of shape n x p, C=AB has shape m x p:

C[i,j] = sum over k of A[i,k]*B[k,j].

Take:

```text
A = [[1,2,3],
     [4,5,6]]
B = [[1,0,2,1],
     [0,1,1,2],
     [2,1,0,1]]
```

Each output is a dot product of one ROW of A with one COLUMN of B.

| Element | Dot product | Value |
|---|---|---:|
| C11 | 1*1 + 2*0 + 3*2 | 7 |
| C12 | 1*0 + 2*1 + 3*1 | 5 |
| C13 | 1*2 + 2*1 + 3*0 | 4 |
| C14 | 1*1 + 2*2 + 3*1 | 8 |
| C21 | 4*1 + 5*0 + 6*2 | 16 |
| C22 | 4*0 + 5*1 + 6*1 | 11 |
| C23 | 4*2 + 5*1 + 6*0 | 13 |
| C24 | 4*1 + 5*2 + 6*1 | 20 |

```text
C = [[ 7, 5, 4, 8],
     [16,11,13,20]]
```

There are eight output entries, three multiplications each. BA is undefined
here: B has 4 columns but A has only 2 rows. In general AB != BA even when both
exist. Elementwise multiplication is a different operation and requires matching
shapes under the simplest definition.

For column-vector convention, if B maps a p-dimensional vector into n dimensions,
and A maps n into m, then AB applies B first, then A. This connects matrix
multiplication with composition and the multivariate chain rule.

Follow-up exercise: choose a 3x4 A and 4x2 B, predict the 3x2 output shape FIRST,
then compute all six row-column products. Larger rectangular examples are useful
for revising mathematics; neural training sizes should be chosen separately.
