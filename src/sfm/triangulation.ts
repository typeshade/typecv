import { symmetricEigen } from './linear.ts';

/** Linear DLT triangulation from calibrated x1, x2 and X2=R X1+t. */
export function triangulate(x: number, y: number, u: number, v: number, R: ArrayLike<number>, t: ArrayLike<number>): Float64Array | undefined {
  const rows = [
    [-1,0,x,0], [0,-1,y,0],
    [u*R[6]! - R[0]!,u*R[7]! - R[1]!,u*R[8]! - R[2]!,u*t[2]! - t[0]!],
    [v*R[6]! - R[3]!,v*R[7]! - R[4]!,v*R[8]! - R[5]!,v*t[2]! - t[1]!],
  ];
  const ata = new Float64Array(16);
  for (const row of rows) for (let i=0; i<4;i++) for(let j=0;j<4;j++) ata[i*4+j]+=row[i]!*row[j]!;
  const eigen = symmetricEigen(ata,4);
  const homogeneous = [eigen.vectors[0]!,eigen.vectors[4]!,eigen.vectors[8]!,eigen.vectors[12]!];
  if(Math.abs(homogeneous[3]!) < 1e-12) return undefined;
  return Float64Array.of(homogeneous[0]!/homogeneous[3]!, homogeneous[1]!/homogeneous[3]!, homogeneous[2]!/homogeneous[3]!);
}
