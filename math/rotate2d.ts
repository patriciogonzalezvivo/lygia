/*
contributors: Patricio Gonzalez Vivo
description: returns a 2x2 rotation matrix
use: <mat2> rotate2d(<float> radians)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

import { Fn, cos, mat2, sin } from 'three/tsl'
import type { tsl_float, tsl_mat2 } from '../tsl-types/types.js'

/**
 * 2D rotation matrix.
 * @see https://en.wikipedia.org/wiki/Rotation_matrix
 */
export const rotate2d = /*@__PURE__*/ Fn(([r]: [tsl_float]): tsl_mat2 => {
  const c = cos(r).toVar('c')
  const s = sin(r).toVar('s')

  return mat2(c, s, s.negate(), c)
}).setLayout({
  name: 'rotate2d',
  type: 'mat2',
  inputs: [{ name: 'r', type: 'float' }],
})
