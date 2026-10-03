/*
contributors: Patricio Gonzalez Vivo
description: returns a 4x4 rotation matrix
use: <mat4> rotate4d(<vec3> axis, <float> radians)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

import { Fn, add, cos, mat4, mul, normalize, sin, sub, vec4 } from 'three/tsl'
import type { tsl_float, tsl_mat4, tsl_vec3 } from '../tsl-types/types.js'

/**
 * We are creating a homgenous rotation matrix from the vector `a` and rotation `r`
 * @see https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
 */
export const rotate4d = /*@__PURE__*/ Fn(
  ([a, r]: [tsl_vec3, tsl_float]): tsl_mat4 => {
    const axis = normalize(a).toVar('axis')

    const s = sin(r).toVar('s')
    const c = cos(r).toVar('c')
    const oc = sub(1, c).toVar('oc')

    const col1 = vec4(
      add(mul(oc, axis.x, axis.x), c),
      add(mul(oc, axis.x, axis.y), mul(axis.z, s)),
      sub(mul(oc, axis.z, axis.x), mul(axis.y, s)),
      0,
    ).toVar('col1')

    const col2 = vec4(
      sub(mul(oc, axis.x, axis.y), mul(axis.z, s)),
      add(mul(oc, axis.y, axis.y), c),
      add(mul(oc, axis.y, axis.z), mul(axis.x, s)),
      0,
    ).toVar('col2')

    const col3 = vec4(
      add(mul(oc, axis.z, axis.x), mul(axis.y, s)),
      sub(mul(oc, axis.y, axis.z), mul(axis.x, s)),
      add(mul(oc, axis.z, axis.z), c),
      0,
    ).toVar('col3')

    const col4 = vec4(0, 0, 0, 1).toVar('col4')

    return mat4(col1, col2, col3, col4)
  },
).setLayout({
  name: 'rotate4d',
  type: 'mat4',
  inputs: [
    { name: 'a', type: 'vec3' },
    { name: 'r', type: 'float' },
  ],
})
