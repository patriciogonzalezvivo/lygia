/*
contributors: Patricio Gonzalez Vivo
description: rotate a 2D space by a radian r
use: rotate(<vec3|vec2> v, float r [, vec2 c])
options:
    - CENTER_2D
    - CENTER_3D
    - CENTER_4D
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_shapes.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

import type {
  OverloadFnParams,
  tsl_float,
  tsl_vec2,
  tsl_vec3,
  tsl_vec4,
} from '../tsl-types/types.js'

import {
  Fn,
  add,
  dot,
  mul,
  overloadingFn,
  sub,
  vec2,
  vec3,
  vec4,
} from 'three/tsl'

import { quatMul } from '../math/quat/mul.js'
import { rotate2d } from '../math/rotate2d.js'
import { rotate4d } from '../math/rotate4d.js'

const rotateVec2Center = /*@__PURE__*/ Fn(
  ([v, r, c]: [tsl_vec2, tsl_float, tsl_vec2]): tsl_vec2 => {
    return add(mul(rotate2d(r), sub(v, c)), c)
  },
).setLayout({
  name: 'rotateVec2Center',
  type: 'vec2',
  inputs: [
    { name: 'v', type: 'vec2' },
    { name: 'r', type: 'float' },
    { name: 'c', type: 'vec2' },
  ],
})

const rotateVec2 = /*@__PURE__*/ Fn(
  ([v, r]: [tsl_vec2, tsl_float]): tsl_vec2 => {
    return rotateVec2Center(v, r, vec2(0.5))
  },
).setLayout({
  name: 'rotateVec2',
  type: 'vec2',
  inputs: [
    { name: 'v', type: 'vec2' },
    { name: 'r', type: 'float' },
  ],
})

const rotateVec2Axis = /*@__PURE__*/ Fn(
  ([v, xAxis]: [tsl_vec2, tsl_vec2]): tsl_vec2 => {
    const rta = vec2(
      dot(v, vec2(mul(-1, xAxis.y), xAxis.x)),
      dot(v, xAxis),
    ).toVar('rta')

    return rta
  },
).setLayout({
  name: 'rotateVec2Axis',
  type: 'vec2',
  inputs: [
    { name: 'v', type: 'vec2' },
    { name: 'xAxis', type: 'vec2' },
  ],
})

const rotateVec3Center = /*@__PURE__*/ Fn(
  ([v, r, axis, c]: [tsl_vec3, tsl_float, tsl_vec3, tsl_vec3]): tsl_vec3 => {
    return add(mul(rotate4d(axis, r), vec4(sub(v, c), 1)).xyz, c)
  },
).setLayout({
  name: 'rotateVec3Center',
  type: 'vec3',
  inputs: [
    { name: 'v', type: 'vec3' },
    { name: 'r', type: 'float' },
    { name: 'axis', type: 'vec3' },
    { name: 'c', type: 'vec3' },
  ],
})

const rotateVec3 = /*@__PURE__*/ Fn(
  ([v, r, axis]: [tsl_vec3, tsl_float, tsl_vec3]): tsl_vec3 => {
    return rotateVec3Center(v, r, axis, vec3(0))
  },
).setLayout({
  name: 'rotateVec3',
  type: 'vec3',
  inputs: [
    { name: 'v', type: 'vec3' },
    { name: 'r', type: 'float' },
    { name: 'axis', type: 'vec3' },
  ],
})

const rotateVec4Center = /*@__PURE__*/ Fn(
  ([v, r, axis, c]: [tsl_vec4, tsl_float, tsl_vec3, tsl_vec4]): tsl_vec4 => {
    return add(mul(rotate4d(axis, r), sub(v, c)), c)
  },
).setLayout({
  name: 'rotateVec4Center',
  type: 'vec4',
  inputs: [
    { name: 'v', type: 'vec4' },
    { name: 'r', type: 'float' },
    { name: 'axis', type: 'vec3' },
    { name: 'c', type: 'vec4' },
  ],
})

const rotateVec4 = /*@__PURE__*/ Fn(
  ([v, r, axis]: [tsl_vec4, tsl_float, tsl_vec3]): tsl_vec4 => {
    return rotateVec4Center(v, r, axis, vec4(0))
  },
).setLayout({
  name: 'rotateVec4',
  type: 'vec4',
  inputs: [
    { name: 'v', type: 'vec4' },
    { name: 'r', type: 'float' },
    { name: 'axis', type: 'vec3' },
  ],
})

const rotateQuat = /*@__PURE__*/ Fn(
  ([q, v]: [tsl_vec4, tsl_vec3]): tsl_vec3 => {
    const qC = vec4(mul(-1, q.xyz), q.w).toVar('q_c')

    return quatMul(q, quatMul(vec4(v, 0), qC)).xyz
  },
).setLayout({
  name: 'rotateQuat',
  type: 'vec3',
  inputs: [
    { name: 'q', type: 'vec4' },
    { name: 'v', type: 'vec3' },
  ],
})

const rotateQuatCenter = /*@__PURE__*/ Fn(
  ([q, v, c]: [tsl_vec4, tsl_vec3, tsl_vec3]): tsl_vec3 => {
    const dir = sub(v, c).toVar('dir')

    return add(c, rotateQuat(q, dir))
  },
).setLayout({
  name: 'rotateQuatCenter',
  type: 'vec3',
  inputs: [
    { name: 'q', type: 'vec4' },
    { name: 'v', type: 'vec3' },
    { name: 'c', type: 'vec3' },
  ],
})

type RotateFnType = typeof rotateVec2Center &
  typeof rotateVec2 &
  typeof rotateVec2Axis &
  typeof rotateVec3Center &
  typeof rotateVec3 &
  typeof rotateVec4Center &
  typeof rotateVec4 &
  typeof rotateQuat &
  typeof rotateQuatCenter

export const rotate = /*@__PURE__*/ overloadingFn([
  rotateVec2Center,
  rotateVec2,
  rotateVec2Axis,
  rotateVec3Center,
  rotateVec3,
  rotateVec4Center,
  rotateVec4,
  rotateQuat,
  rotateQuatCenter,
] as unknown as OverloadFnParams) as unknown as RotateFnType
