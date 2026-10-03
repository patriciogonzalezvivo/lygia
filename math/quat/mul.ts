/*
contributors: Patricio Gonzalez Vivo
description: 'Quaternion multiplication. Based on http://mathworld.wolfram.com/Quaternion.html'
use: <QUAT> quatMul(<QUAT> a, <QUAT> b)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

import { Fn, add, cross, dot, mul, overloadingFn, sub, vec4 } from 'three/tsl'
import type {
  OverloadFnParams,
  tsl_float,
  tsl_vec4,
} from '../../tsl-types/types.js'

const quatMulQuat = /*@__PURE__*/ Fn(
  ([q1, q2]: [tsl_vec4, tsl_vec4]): tsl_vec4 => {
    return vec4(
      add(mul(q2.xyz, q1.w), mul(q1.xyz, q2.w), cross(q1.xyz, q2.xyz)),
      sub(mul(q1.w, q2.w), dot(q1.xyz, q2.xyz)),
    )
  },
).setLayout({
  name: 'quatMulQuat',
  type: 'vec4',
  inputs: [
    { name: 'q1', type: 'vec4' },
    { name: 'q2', type: 'vec4' },
  ],
})

const quatMulScalar = /*@__PURE__*/ Fn(
  ([q, s]: [tsl_vec4, tsl_float]): tsl_vec4 => {
    return vec4(mul(q.xyz, s), mul(q.w, s))
  },
).setLayout({
  name: 'quatMulScalar',
  type: 'vec4',
  inputs: [
    { name: 'q', type: 'vec4' },
    { name: 's', type: 'float' },
  ],
})

type QuatMulFnType = typeof quatMulQuat & typeof quatMulScalar

export const quatMul = /*@__PURE__*/ overloadingFn([
  quatMulQuat,
  quatMulScalar,
] as unknown as OverloadFnParams) as unknown as QuatMulFnType
