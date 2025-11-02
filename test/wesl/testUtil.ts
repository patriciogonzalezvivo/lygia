import { expect } from "vitest";
import type {
  ComputeTestParams,
  FragmentTestParams,
  WgslElementType,
} from "wesl-test";
import { getGPUDevice, testCompute, testFragment } from "wesl-test";

const projectDir = new URL("../../", import.meta.url).href;

/** compare two arrays for approximate equality */
export function expectCloseTo(
  a: number[],
  b: number[],
  epsilon = 0.0001,
): void {
  const match = a.every((val, index) => Math.abs(val - b[index]) < epsilon);
  if (match) return;
  expect.fail(`arrays don't match:\n  ${a}\n  ${b}`);
}

type ModuleTestParams =
  | "src"
  | "moduleName"
  | "projectDir"
  | "useSourceShaders"
  | "device";

/** Options for testCompute function - omits params handled by lygiaTestCompute wrapper */
export interface LygiaTestCompute
  extends Omit<ComputeTestParams, ModuleTestParams | "resultFormat"> {
  /** Element type for result buffer (renamed from ComputeTestParams.resultFormat) */
  elem?: WgslElementType;
}

/** test WGSL compute shader with typical lygia defaults */
export async function lygiaTestCompute(
  src: string,
  options: LygiaTestCompute = {},
) {
  const { size, conditions, constants, dispatchWorkgroups } = options;
  const { elem: resultFormat = "f32" } = options;

  const device = await getGPUDevice();

  return testCompute({
    projectDir,
    device,
    src,
    resultFormat,
    size,
    conditions,
    dispatchWorkgroups,
    constants,
  });
}

/** Options for testFragment function - omits params handled by lygiaTestFragment wrapper */
export type LygiaTestFragment = Omit<FragmentTestParams, ModuleTestParams>;

/** test WGSL fragment shader with typical lygia defaults */
export async function lygiaTestFragment(
  src: string,
  options: LygiaTestFragment = {},
) {
  const { size, textureFormat = "rgba32float" } = options;
  const { conditions, constants, inputTextures, uniforms } = options;

  const device = await getGPUDevice();
  return await testFragment({
    projectDir,
    device,
    src,
    textureFormat,
    size,
    conditions,
    constants,
    uniforms,
    inputTextures,
  });
}

/**
 * Test distribution properties of a random function.
 * Collects multiple samples and returns them for statistical analysis.
 *
 * @param src - WESL shader source that writes samples to test::results
 * @param sampleCount - Number of samples to collect
 * @param elem - Element type (default "f32")
 * @param constants - Constants to pass via constants:: namespace (e.g., SAMPLE_COUNT)
 * @returns Array of sample values
 */
export async function testDistribution(
  src: string,
  sampleCount: number,
  elem: WgslElementType = "f32",
  constants?: Record<string, string | number>,
): Promise<number[]> {
  const device = await getGPUDevice();

  return testCompute({
    projectDir,
    device,
    src,
    resultFormat: elem,
    size: sampleCount,
    constants,
  });
}

/**
 * Validate that samples follow a uniform distribution.
 *
 * @param samples - Array of sample values
 * @param range - Expected [min, max] range
 * @param options - Test options (meanTolerance, bucketTolerance)
 */
export function expectDistribution(
  samples: number[],
  range: [number, number],
  options: {
    meanTolerance?: number;
    bucketTolerance?: number;
  } = {},
): void {
  const { meanTolerance = 0.05, bucketTolerance = 0.03 } = options;
  const [min, max] = range;
  const expectedMean = (min + max) / 2;

  // Test 1: Mean
  const actualMean = samples.reduce((sum, v) => sum + v, 0) / samples.length;
  const meanDiff = Math.abs(actualMean - expectedMean);
  if (meanDiff > meanTolerance) {
    expect.fail(
      `Mean ${actualMean.toFixed(4)} differs from expected ${expectedMean} by ${meanDiff.toFixed(4)} (threshold: ${meanTolerance})`,
    );
  }

  // Test 2: Bucket distribution
  const bucketCount = 10;
  const buckets = new Array(bucketCount).fill(0);
  const bucketWidth = (max - min) / bucketCount;

  for (const value of samples) {
    const bucketIndex = Math.min(
      Math.floor((value - min) / bucketWidth),
      bucketCount - 1,
    );
    buckets[bucketIndex]++;
  }

  const expectedRatio = 1.0 / bucketCount; // 0.1 for 10 buckets

  for (let i = 0; i < bucketCount; i++) {
    const ratio = buckets[i] / samples.length;
    const diff = Math.abs(ratio - expectedRatio);

    if (diff > bucketTolerance) {
      const bucketRange = [
        (min + i * bucketWidth).toFixed(2),
        (min + (i + 1) * bucketWidth).toFixed(2),
      ];
      expect.fail(
        `Bucket ${i} [${bucketRange[0]}, ${bucketRange[1]}) has ${(ratio * 100).toFixed(1)}% of samples (expected ${expectedRatio * 100}% ± ${bucketTolerance * 100}%)\n` +
          `Distribution: ${buckets.map((b) => `${((b / samples.length) * 100).toFixed(1)}%`).join(", ")}`,
      );
    }
  }
}
