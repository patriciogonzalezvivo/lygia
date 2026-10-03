import { normalize, vec4 } from 'three/tsl'
import { test } from 'vitest'
import { quatMul } from '../../math/quat/mul.js'
import { expectCloseTo, tslTestCompute } from './testUtil.ts'

test('quatMul', async () => {
  const q1 = normalize(vec4(1.0, 0.0, 0.0, 1.0))
  const q2 = normalize(vec4(0.0, 1.0, 0.0, 1.0))
  expectCloseTo([0.5, 0.5, 0.5, 0.5], await tslTestCompute(quatMul(q1, q2)))
})
