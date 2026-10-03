import { float, vec2, vec3, vec4 } from 'three/tsl'
import { test } from 'vitest'
import { rotate } from '../../space/rotate.js'
import { expectCloseTo, tslTestCompute } from './testUtil.ts'

test.each([
  ['rotateX3', vec3(1.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), [1.0, 0.0, 1.0, 0.0]],
  ['rotateY3', vec3(1.0, 1.0, 0.0), vec3(0.0, 1.0, 0.0), [0.0, 1.0, -1.0, 0.0]],
  ['rotateZ3', vec3(1.0, 0.0, 1.0), vec3(0.0, 0.0, 1.0), [0.0, 1.0, 1.0, 0.0]],
] as const)('%s', async (_, v, axis, expected) => {
  const rotated = rotate(v, float(Math.PI / 2), axis)
  const result = await tslTestCompute(vec4(rotated, 0.0))
  expectCloseTo([...expected], result)
})

test('rotate', async () => {
  const rotated = rotate(vec2(1.0, 0.5), float(Math.PI / 2))
  const result = await tslTestCompute(vec4(rotated, 0.0, 0.0))
  expectCloseTo([0.5, 1.0], result.slice(0, 2))
})

test('rotate_c', async () => {
  const rotated = rotate(vec2(1.0, 0.0), float(Math.PI / 2), vec2(0.0))
  const result = await tslTestCompute(vec4(rotated, 0.0, 0.0))
  expectCloseTo([0.0, 1.0], result.slice(0, 2))
})

test('rotate3', async () => {
  const rotated = rotate(
    vec3(1.0, 0.0, 0.0),
    float(Math.PI / 2),
    vec3(0.0, 0.0, 1.0),
  )
  const result = await tslTestCompute(vec4(rotated, 0.0))
  const length = Math.sqrt(
    result[0] * result[0] + result[1] * result[1] + result[2] * result[2],
  )
  expectCloseTo([length], [1.0])
  expectCloseTo([0.0, 1.0, 0.0, 0.0], result)
})

// TSL passes WESL's center constants as explicit arguments.
test('rotate - with custom CENTER_2D via constants', async () => {
  const rotated = rotate(vec2(0.8, 0.3), float(Math.PI / 2), vec2(0.3, 0.3))
  const result = await tslTestCompute(vec4(rotated, 0.0, 0.0))
  expectCloseTo([0.3, 0.8], result.slice(0, 2))
})

test.each([
  ['rotateX3', vec3(1.0, 1.5, 0.5), vec3(1.0, 0.0, 0.0), [1.0, 0.5, 1.5, 0.0]],
  ['rotateY3', vec3(1.5, 1.0, 0.5), vec3(0.0, 1.0, 0.0), [0.5, 1.0, -0.5, 0.0]],
  ['rotateZ3', vec3(1.5, 0.5, 1.0), vec3(0.0, 0.0, 1.0), [0.5, 1.5, 1.0, 0.0]],
] as const)(
  '%s - with custom CENTER_3D via constants',
  async (_, v, axis, expected) => {
    const rotated = rotate(v, float(Math.PI / 2), axis, vec3(0.5, 0.5, 0.5))
    const result = await tslTestCompute(vec4(rotated, 0.0))
    expectCloseTo([...expected], result)
  },
)
