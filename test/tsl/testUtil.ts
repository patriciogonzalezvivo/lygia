import { WGSLNodeBuilder } from 'three/webgpu'
import { getGPUDevice, testCompute } from 'wgsl-test'
import type { tsl_vec4 } from '../../tsl-types/types.js'

export { expectCloseTo } from '../wesl/testUtil.ts'

type ComputeBuilder = WGSLNodeBuilder & {
  setShaderStage(stage: 'compute'): void
  flowStagesNode(
    node: tsl_vec4,
    output: 'vec4',
  ): { vars: string; code: string; result: string }
  getCodes(stage: 'compute'): string
}

/**
 * Using TSL's Node Builder we are creating a Compute shader here to run and test the
 * Function nodes step-by-step.
 * Since TSL can be dead-code eliminated easily by any build system, we don't need
 * GLSL style constants or conditions
 * I've also not included `dispatchWorkgroups` to keep this logic simple.
 */
export async function tslTestCompute(node: tsl_vec4): Promise<number[]> {
  const renderer = {
    backend: {},
    debug: { diagnostics: { keywords: true } },
  } as unknown as ConstructorParameters<typeof WGSLNodeBuilder>[1]
  const builder = new WGSLNodeBuilder(null!, renderer) as ComputeBuilder
  builder.setShaderStage('compute')
  const flow = builder.flowStagesNode(node, 'vec4')

  return testCompute({
    device: await getGPUDevice(),
    src: `
      ${builder.getCodes('compute')}
      @compute @workgroup_size(1)
      fn main() {
        ${flow.vars}
        ${flow.code}
        env::results[0] = ${flow.result};
      }
    `,
    resultFormat: 'vec4f',
    size: 1,
  })
}
