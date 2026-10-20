import * as THREE from 'three';

/**
 * The geometry of the dense layer (`docs/dense-mvs.md` section 6): 12 B of position and 3 B of
 * normalized sRGB color per point. Both attributes alias the given arrays, so the viewer's later
 * writes into them reach the GPU through the update ranges. `THREE.Uint8BufferAttribute` copies
 * its array, so writes into the caller's array would never upload.
 */
export function denseGeometry(positions: Float32Array, colors: Uint8Array): THREE.BufferGeometry {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3, true));
  return geometry;
}
