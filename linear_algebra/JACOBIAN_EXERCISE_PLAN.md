# Optional extension: Jacobians of coordinate transformations

Status: planned mathematical exercise, NOT implemented neural support.
Place after dot products and matrix-vector operations. It connects partial
derivatives, local linear approximation, determinants and coordinate changes.
Current monomial/AST networks do not support general multivariate trigonometric
partial derivatives, so do not present this as merely another existing prompt.

## Start with polar coordinates

Define the FORWARD map F(r,phi) = (r*cos(phi), r*sin(phi)) = (x,y).
Rows correspond to output coordinates x,y; columns to input coordinates r,phi.

| Jacobian entry | Partial derivative |
|---|---|
| J11 | dx/dr = cos(phi) |
| J12 | dx/dphi = -r*sin(phi) |
| J21 | dy/dr = sin(phi) |
| J22 | dy/dphi = r*cos(phi) |

Thus J = [[cos(phi), -r*sin(phi)], [sin(phi), r*cos(phi)]], det J = r.
The Jacobian MATRIX and its DETERMINANT are different objects.
Local changes obey [dx,dy]^T approximately J*[dr,dphi]^T.
For area integration use absolute determinant: dx dy = |r| dr dphi;
with standard r>0, this is r dr dphi.

The inverse-coordinate Jacobian has determinant 1/r only where the inverse is
locally defined (r>0 and a chosen smooth angle branch). At the origin polar angle
is undefined and J is singular. Do not exchange forward/inverse determinants.

## Follow-ups

- Cylindrical: (r,phi,z) -> (r*cos(phi),r*sin(phi),z), determinant r.
- Spherical: (r,theta,phi) -> (r*sin(theta)*cos(phi),r*sin(theta)*sin(phi),r*cos(theta)),
  where theta is polar angle and phi is azimuth; determinant r**2*sin(theta)
  with these stated coordinate orders. Volume uses its absolute value.
- Multivariate chain rule: J_(F composed with G) = J_F(G(q))*J_G(q), preserving
  matrix order. This is an excellent reason to revise rectangular multiplication.

First implement exact reference calculations or finite-difference checks of
Jacobian entries at nonsingular sample points. A later neural experiment must
specify whether it learns symbolic partial derivatives, numerical Jacobian values,
or selects a deterministic differentiation tool. Keep the learning goal explicit.
