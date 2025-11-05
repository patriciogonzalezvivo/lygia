import { test } from "vitest";
import { expectCloseTo, lygiaTestCompute } from "./testUtil.ts";

test("compositeSourceOver4", async () => {
  const src = `
    import lygia::color::composite::sourceOver::compositeSourceOver4;

    @compute @workgroup_size(1)
    fn foo() {
      let src = vec4f(1.0, 0.0, 0.0, 0.5);
      let dst = vec4f(0.0, 0.0, 1.0, 0.5);
      let result = compositeSourceOver4(src, dst);
      test::results[0] = result;
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // src + dst * (1 - src.a)
  // alpha: 0.5 + 0.5 * 0.5 = 0.75
  // rgb: src.rgb * src.a + dst.rgb * dst.a * (1 - src.a)
  expectCloseTo([0.5, 0.0, 0.25, 0.75], result);
});

test("compositeSourceIn4", async () => {
  const src = `
    import lygia::color::composite::sourceIn::compositeSourceIn4;

    @compute @workgroup_size(1)
    fn foo() {
      let src = vec4f(1.0, 0.0, 0.0, 0.8);
      let dst = vec4f(0.0, 0.0, 1.0, 0.5);
      let result = compositeSourceIn4(src, dst);
      test::results[0] = result;
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // src * dst.a
  // alpha: 0.8 * 0.5 = 0.4
  // rgb: src.rgb * dst.a = (1, 0, 0) * 0.5
  expectCloseTo([0.5, 0.0, 0.0, 0.4], result);
});

test("compositeSourceOut4", async () => {
  const src = `
    import lygia::color::composite::sourceOut::compositeSourceOut4;

    @compute @workgroup_size(1)
    fn foo() {
      let src = vec4f(1.0, 0.0, 0.0, 0.8);
      let dst = vec4f(0.0, 1.0, 0.0, 0.4);
      let result = compositeSourceOut4(src, dst);
      test::results[0] = result;
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Source out: src * (1 - dst.a)
  // rgb: src.rgb * (1 - dst.a) = (1,0,0) * (1-0.4) = (1,0,0) * 0.6 = (0.6,0,0)
  // alpha: src.a * (1 - dst.a) = 0.8 * 0.6 = 0.48
  expectCloseTo([0.6, 0.0, 0.0, 0.48], result);
});

test("compositeSourceAtop4", async () => {
  const src = `
    import lygia::color::composite::sourceAtop::compositeSourceAtop4;

    @compute @workgroup_size(1)
    fn foo() {
      let src = vec4f(1.0, 0.0, 0.0, 0.6);
      let dst = vec4f(0.0, 0.0, 1.0, 0.5);
      let result = compositeSourceAtop4(src, dst);
      test::results[0] = result;
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Source atop: src * dst.a + dst * (1 - src.a)
  // rgb: src.rgb * dst.a + dst.rgb * (1 - src.a) = (1,0,0)*0.5 + (0,0,1)*0.4 = (0.5,0,0.4)
  // alpha: src.a * dst.a + dst.a * (1 - src.a) = 0.6*0.5 + 0.5*0.4 = 0.3 + 0.2 = 0.5
  expectCloseTo([0.5, 0.0, 0.4, 0.5], result);
});

test("compositeDestinationOver4", async () => {
  const src = `
    import lygia::color::composite::destinationOver::compositeDestinationOver4;

    @compute @workgroup_size(1)
    fn foo() {
      let src = vec4f(1.0, 0.0, 0.0, 0.5);
      let dst = vec4f(0.0, 0.0, 1.0, 0.6);
      let result = compositeDestinationOver4(src, dst);
      test::results[0] = result;
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Destination over: dst + src * (1 - dst.a)
  // rgb: dst.rgb + src.rgb * (1 - dst.a) = (0,0,1) + (1,0,0)*0.4 = (0.4,0,1)
  // alpha: dst.a + src.a * (1 - dst.a) = 0.6 + 0.5*0.4 = 0.8
  expectCloseTo([0.4, 0.0, 1.0, 0.8], result);
});

test("compositeDestinationIn4", async () => {
  const src = `
    import lygia::color::composite::destinationIn::compositeDestinationIn4;

    @compute @workgroup_size(1)
    fn foo() {
      let src = vec4f(1.0, 0.0, 0.0, 0.6);
      let dst = vec4f(0.0, 1.0, 0.0, 0.8);
      let result = compositeDestinationIn4(src, dst);
      test::results[0] = result;
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // rgb: dst.rgb * src.a
  // rgb: (0,1,0) * 0.6 = (0,0.6,0)
  // alpha: dst.a * src.a = 0.8 * 0.6 = 0.48
  expectCloseTo([0.0, 0.6, 0.0, 0.48], result);
});

test("compositeDestinationOut4", async () => {
  const src = `
    import lygia::color::composite::destinationOut::compositeDestinationOut4;

    @compute @workgroup_size(1)
    fn foo() {
      let src = vec4f(1.0, 0.0, 0.0, 0.3);
      let dst = vec4f(0.0, 1.0, 0.0, 0.7);
      let result = compositeDestinationOut4(src, dst);
      test::results[0] = result;
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // rgb: dst.rgb * (1 - src.a)
  // rgb: (0,1,0) * (1 - 0.3) = (0,1,0) * 0.7 = (0,0.7,0)
  // alpha: dst.a * (1 - src.a) = 0.7 * 0.7 = 0.49
  expectCloseTo([0.0, 0.7, 0.0, 0.49], result);
});

test("compositeDestinationAtop4", async () => {
  const src = `
    import lygia::color::composite::destinationAtop::compositeDestinationAtop4;

    @compute @workgroup_size(1)
    fn foo() {
      let src = vec4f(1.0, 0.0, 0.0, 0.7);
      let dst = vec4f(0.0, 0.0, 1.0, 0.5);
      let result = compositeDestinationAtop4(src, dst);
      test::results[0] = result;
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // rgb: dst.rgb * src.a + src.rgb * (1 - dst.a)
  // rgb: (0,0,1)*0.7 + (1,0,0)*(1-0.5) = (0,0,0.7) + (0.5,0,0) = (0.5,0,0.7)
  // alpha: dst.a * src.a + src.a * (1 - dst.a) = 0.5*0.7 + 0.7*0.5 = 0.35 + 0.35 = 0.7
  expectCloseTo([0.5, 0.0, 0.7, 0.7], result);
});

test("compositeXor4", async () => {
  const src = `
    import lygia::color::composite::compositeXor::compositeXor4;

    @compute @workgroup_size(1)
    fn foo() {
      let src = vec4f(1.0, 0.0, 0.0, 0.6);
      let dst = vec4f(0.0, 0.0, 1.0, 0.4);
      let result = compositeXor4(src, dst);
      test::results[0] = result;
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // src * (1 - dst.a) + dst * (1 - src.a)
  // rgb: (1,0,0)*(1-0.4) + (0,0,1)*(1-0.6) = (0.6,0,0) + (0,0,0.4) = (0.6,0,0.4)
  // alpha: 0.6 * 0.6 + 0.4 * 0.4 = 0.36 + 0.16 = 0.52
  expectCloseTo([0.6, 0.0, 0.4, 0.52], result);
});

// Vec3 variant tests (with separate alpha parameters)

test("compositeSourceOver3", async () => {
  const src = `
    import lygia::color::composite::sourceOver::compositeSourceOver3;

    @compute @workgroup_size(1)
    fn foo() {
      let srcColor = vec3f(1.0, 0.0, 0.0);
      let dstColor = vec3f(0.0, 0.0, 1.0);
      let srcAlpha = 0.5;
      let dstAlpha = 0.5;
      let result = compositeSourceOver3(srcColor, dstColor, srcAlpha, dstAlpha);
      test::results[0] = vec4f(result, 0.0);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // rgb: src.rgb * src.a + dst.rgb * dst.a * (1 - src.a)
  // rgb: (1,0,0)*0.5 + (0,0,1)*0.5*0.5 = (0.5,0,0) + (0,0,0.25) = (0.5,0,0.25)
  expectCloseTo([0.5, 0.0, 0.25, 0.0], result);
});

test("compositeSourceIn3", async () => {
  const src = `
    import lygia::color::composite::sourceIn::compositeSourceIn3;

    @compute @workgroup_size(1)
    fn foo() {
      let srcColor = vec3f(1.0, 0.0, 0.0);
      let dstColor = vec3f(0.0, 0.0, 1.0);
      let srcAlpha = 0.8;
      let dstAlpha = 0.6;
      let result = compositeSourceIn3(srcColor, dstColor, srcAlpha, dstAlpha);
      test::results[0] = vec4f(result, 0.0);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // rgb: src.rgb * dst.a = (1,0,0) * 0.6 = (0.6,0,0)
  expectCloseTo([0.6, 0.0, 0.0, 0.0], result);
});

test("compositeSourceOut3", async () => {
  const src = `
    import lygia::color::composite::sourceOut::compositeSourceOut3;

    @compute @workgroup_size(1)
    fn foo() {
      let srcColor = vec3f(1.0, 0.0, 0.0);
      let dstColor = vec3f(0.0, 1.0, 0.0);
      let srcAlpha = 0.8;
      let dstAlpha = 0.3;
      let result = compositeSourceOut3(srcColor, dstColor, srcAlpha, dstAlpha);
      test::results[0] = vec4f(result, 0.0);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // rgb: src.rgb * (1 - dst.a) = (1,0,0) * (1 - 0.3) = (1,0,0) * 0.7 = (0.7,0,0)
  expectCloseTo([0.7, 0.0, 0.0, 0.0], result);
});

test("compositeSourceAtop3", async () => {
  const src = `
    import lygia::color::composite::sourceAtop::compositeSourceAtop3;

    @compute @workgroup_size(1)
    fn foo() {
      let srcColor = vec3f(1.0, 0.0, 0.0);
      let dstColor = vec3f(0.0, 0.0, 1.0);
      let srcAlpha = 0.6;
      let dstAlpha = 0.5;
      let result = compositeSourceAtop3(srcColor, dstColor, srcAlpha, dstAlpha);
      test::results[0] = vec4f(result, 0.0);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // rgb: src.rgb * dst.a + dst.rgb * (1 - src.a)
  // rgb: (1,0,0)*0.5 + (0,0,1)*(1-0.6) = (0.5,0,0) + (0,0,0.4) = (0.5,0,0.4)
  expectCloseTo([0.5, 0.0, 0.4, 0.0], result);
});

test("compositeDestinationOver3", async () => {
  const src = `
    import lygia::color::composite::destinationOver::compositeDestinationOver3;

    @compute @workgroup_size(1)
    fn foo() {
      let srcColor = vec3f(1.0, 0.0, 0.0);
      let dstColor = vec3f(0.0, 0.0, 1.0);
      let srcAlpha = 0.5;
      let dstAlpha = 0.6;
      let result = compositeDestinationOver3(srcColor, dstColor, srcAlpha, dstAlpha);
      test::results[0] = vec4f(result, 0.0);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // rgb: dst.rgb + src.rgb * (1 - dst.a) = (0,0,1) + (1,0,0)*(1-0.6) = (0,0,1) + (0.4,0,0) = (0.4,0,1)
  expectCloseTo([0.4, 0.0, 1.0, 0.0], result);
});

test("compositeDestinationIn3", async () => {
  const src = `
    import lygia::color::composite::destinationIn::compositeDestinationIn3;

    @compute @workgroup_size(1)
    fn foo() {
      let srcColor = vec3f(1.0, 0.0, 0.0);
      let dstColor = vec3f(0.0, 1.0, 0.0);
      let srcAlpha = 0.7;
      let dstAlpha = 0.8;
      let result = compositeDestinationIn3(srcColor, dstColor, srcAlpha, dstAlpha);
      test::results[0] = vec4f(result, 0.0);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // rgb: dst.rgb * src.a = (0,1,0) * 0.7 = (0,0.7,0)
  expectCloseTo([0.0, 0.7, 0.0, 0.0], result);
});

test("compositeDestinationOut3", async () => {
  const src = `
    import lygia::color::composite::destinationOut::compositeDestinationOut3;

    @compute @workgroup_size(1)
    fn foo() {
      let srcColor = vec3f(1.0, 0.0, 0.0);
      let dstColor = vec3f(0.0, 1.0, 0.0);
      let srcAlpha = 0.4;
      let dstAlpha = 0.7;
      let result = compositeDestinationOut3(srcColor, dstColor, srcAlpha, dstAlpha);
      test::results[0] = vec4f(result, 0.0);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // rgb: dst.rgb * (1 - src.a) = (0,1,0) * (1 - 0.4) = (0,1,0) * 0.6 = (0,0.6,0)
  expectCloseTo([0.0, 0.6, 0.0, 0.0], result);
});

test("compositeDestinationAtop3", async () => {
  const src = `
    import lygia::color::composite::destinationAtop::compositeDestinationAtop3;

    @compute @workgroup_size(1)
    fn foo() {
      let srcColor = vec3f(1.0, 0.0, 0.0);
      let dstColor = vec3f(0.0, 0.0, 1.0);
      let srcAlpha = 0.7;
      let dstAlpha = 0.5;
      let result = compositeDestinationAtop3(srcColor, dstColor, srcAlpha, dstAlpha);
      test::results[0] = vec4f(result, 0.0);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // rgb: dst.rgb * src.a + src.rgb * (1 - dst.a)
  // rgb: (0,0,1)*0.7 + (1,0,0)*(1-0.5) = (0,0,0.7) + (0.5,0,0) = (0.5,0,0.7)
  expectCloseTo([0.5, 0.0, 0.7, 0.0], result);
});

test("compositeXor3", async () => {
  const src = `
    import lygia::color::composite::compositeXor::compositeXor3;

    @compute @workgroup_size(1)
    fn foo() {
      let srcColor = vec3f(1.0, 0.0, 0.0);
      let dstColor = vec3f(0.0, 0.0, 1.0);
      let srcAlpha = 0.6;
      let dstAlpha = 0.4;
      let result = compositeXor3(srcColor, dstColor, srcAlpha, dstAlpha);
      test::results[0] = vec4f(result, 0.0);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // rgb: src.rgb * (1 - dst.a) + dst.rgb * (1 - src.a)
  // rgb: (1,0,0)*(1-0.4) + (0,0,1)*(1-0.6) = (0.6,0,0) + (0,0,0.4) = (0.6,0,0.4)
  expectCloseTo([0.6, 0.0, 0.4, 0.0], result);
});
