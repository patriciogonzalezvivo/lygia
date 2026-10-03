import { float, mul, normalize, vec2, vec3, vec4 } from 'three/tsl'
import { test } from 'vitest'
import { rotate2d } from '../../math/rotate2d.js'
import { rotate4d } from '../../math/rotate4d.js'
import { expectCloseTo, tslTestCompute } from './testUtil.ts'

test('rotate2d - 90 degree rotation', async () => {
  const mat = rotate2d(float(Math.PI / 2))
  const v = vec2(1.0, 0.0)
  const result = await tslTestCompute(vec4(mul(mat, v), 0.0, 0.0))
  expectCloseTo([0.0, 1.0, 0.0, 0.0], result)
})

test('rotate4d - axis-angle rotation', async () => {
  const axis = normalize(vec3(0.0, 0.0, 1.0))
  const mat = rotate4d(axis, float(Math.PI / 2))
  const v = vec4(1.0, 0.0, 0.0, 1.0)
  const result = await tslTestCompute(mul(mat, v))
  expectCloseTo([0.0, 1.0, 0.0, 1.0], result)
})

test.each([
  [
    'rotate4dX',
    vec3(1.0, 0.0, 0.0),
    vec4(0.0, 1.0, 0.0, 1.0),
    [0.0, 0.0, 1.0, 1.0],
  ],
  [
    'rotate4dY',
    vec3(0.0, 1.0, 0.0),
    vec4(1.0, 0.0, 0.0, 1.0),
    [0.0, 0.0, -1.0, 1.0],
  ],
  [
    'rotate4dZ',
    vec3(0.0, 0.0, 1.0),
    vec4(1.0, 0.0, 0.0, 1.0),
    [0.0, 1.0, 0.0, 1.0],
  ],
] as const)('%s', async (_, axis, v, expected) => {
  const mat = rotate4d(axis, float(Math.PI / 2))
  const result = await tslTestCompute(mul(mat, v))
  expectCloseTo([...expected], result)
})
