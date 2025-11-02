import { test } from "vitest";
import { expectCloseTo, lygiaTestCompute } from "./testUtil.ts";

test("backIn", async () => {
  const src = `
    import lygia::animation::easing::backIn::backIn;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = backIn(0.5); }
  `;
  const result = await lygiaTestCompute(src);
  expectCloseTo([-0.375], result);
});

test("backOut", async () => {
  const src = `
    import lygia::animation::easing::backOut::backOut;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = backOut(0.5); }
  `;
  const result = await lygiaTestCompute(src);
  expectCloseTo([1.375], result);
});

test("backInOut", async () => {
  const src = `
    import lygia::animation::easing::backInOut::backInOut;
    @compute @workgroup_size(1)
    fn foo() {
      // Test multiple points to show overshoot behavior
      test::results[0] = vec4f(
        backInOut(0.0),   // Start
        backInOut(0.25),  // First half with overshoot
        backInOut(0.75),  // Second half with overshoot
        backInOut(1.0)    // End
      );
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([0.0, -0.1875, 1.1875, 1.0], result);
});

test("bounceIn", async () => {
  const src = `
    import lygia::animation::easing::bounceIn::bounceIn;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = bounceIn(0.5); }
  `;
  const result = await lygiaTestCompute(src);
  expectCloseTo([0.28125], result);
});

test("bounceOut", async () => {
  const src = `
    import lygia::animation::easing::bounceOut::bounceOut;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = bounceOut(0.5); }
  `;
  const result = await lygiaTestCompute(src);
  expectCloseTo([0.71875], result);
});

test("bounceInOut", async () => {
  const src = `
    import lygia::animation::easing::bounceInOut::bounceInOut;
    @compute @workgroup_size(1)
    fn foo() {
      // Test multiple points to show bounce behavior
      test::results[0] = vec4f(
        bounceInOut(0.0),   // Start
        bounceInOut(0.25),  // First half bouncing in
        bounceInOut(0.75),  // Second half bouncing out
        bounceInOut(1.0)    // End
      );
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([0.0, 0.140625, 0.859375, 1.0], result);
});

test("circularIn", async () => {
  const src = `
    import lygia::animation::easing::circularIn::circularIn;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = circularIn(0.5); }
  `;
  const result = await lygiaTestCompute(src);
  expectCloseTo([0.134], result);
});

test("circularOut", async () => {
  const src = `
    import lygia::animation::easing::circularOut::circularOut;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = circularOut(0.5); }
  `;
  const result = await lygiaTestCompute(src);
  expectCloseTo([0.866], result);
});

test("circularInOut", async () => {
  const src = `
    import lygia::animation::easing::circularInOut::circularInOut;
    @compute @workgroup_size(1)
    fn foo() {
      // Test multiple points to show circular curve
      test::results[0] = vec4f(
        circularInOut(0.0),   // Start
        circularInOut(0.25),  // First quarter
        circularInOut(0.75),  // Third quarter
        circularInOut(1.0)    // End
      );
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([0.0, 0.067, 0.933, 1.0], result);
});

test("cubicIn", async () => {
  const src = `
    import lygia::animation::easing::cubicIn::cubicIn;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = cubicIn(0.5); }
  `;
  const result = await lygiaTestCompute(src);
  expectCloseTo([0.125], result);
});

test("cubicOut", async () => {
  const src = `
    import lygia::animation::easing::cubicOut::cubicOut;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = cubicOut(0.5); }
  `;
  const result = await lygiaTestCompute(src);
  expectCloseTo([0.875], result);
});

test("cubicInOut", async () => {
  const src = `
    import lygia::animation::easing::cubicInOut::cubicInOut;
    @compute @workgroup_size(1)
    fn foo() {
      // Test multiple points to show cubic curve
      test::results[0] = vec4f(
        cubicInOut(0.0),   // Start
        cubicInOut(0.25),  // First quarter
        cubicInOut(0.75),  // Third quarter
        cubicInOut(1.0)    // End
      );
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([0.0, 0.0625, 0.9375, 1.0], result);
});

test("elasticIn", async () => {
  const src = `
    import lygia::animation::easing::elasticIn::elasticIn;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = elasticIn(0.5); }
  `;
  const result = await lygiaTestCompute(src);
  expectCloseTo([-0.0221], result);
});

test("elasticOut", async () => {
  const src = `
    import lygia::animation::easing::elasticOut::elasticOut;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = elasticOut(0.5); }
  `;
  const result = await lygiaTestCompute(src);
  expectCloseTo([1.0221], result);
});

test("elasticInOut", async () => {
  const src = `
    import lygia::animation::easing::elasticInOut::elasticInOut;
    @compute @workgroup_size(1)
    fn foo() {
      // Test multiple points to show elastic oscillation
      test::results[0] = vec4f(
        elasticInOut(0.0),   // Start
        elasticInOut(0.25),  // First quarter with oscillation
        elasticInOut(0.75),  // Third quarter with oscillation
        elasticInOut(1.0)    // End
      );
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([0.0, -0.01105, 1.01105, 1.0], result);
});

test("exponentialIn", async () => {
  const src = `
    import lygia::animation::easing::exponentialIn::exponentialIn;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = exponentialIn(0.5); }
  `;
  const result = await lygiaTestCompute(src);
  expectCloseTo([0.03125], result);
});

test("exponentialOut", async () => {
  const src = `
    import lygia::animation::easing::exponentialOut::exponentialOut;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = exponentialOut(0.5); }
  `;
  const result = await lygiaTestCompute(src);
  expectCloseTo([0.96875], result);
});

test("exponentialInOut", async () => {
  const src = `
    import lygia::animation::easing::exponentialInOut::exponentialInOut;
    @compute @workgroup_size(1)
    fn foo() {
      // Test multiple points to show exponential curve
      test::results[0] = vec4f(
        exponentialInOut(0.0),   // Start
        exponentialInOut(0.25),  // First quarter
        exponentialInOut(0.75),  // Third quarter
        exponentialInOut(1.0)    // End
      );
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([0.0, 0.015625, 0.984375, 1.0], result);
});

test("linearIn", async () => {
  const src = `
    import lygia::animation::easing::linearIn::linearIn;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = linearIn(0.5); }
  `;
  const result = await lygiaTestCompute(src);
  expectCloseTo([0.5], result);
});

test("linearOut", async () => {
  const src = `
    import lygia::animation::easing::linearOut::linearOut;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = linearOut(0.5); }
  `;
  const result = await lygiaTestCompute(src);
  expectCloseTo([0.5], result);
});

test("linearInOut", async () => {
  const src = `
    import lygia::animation::easing::linearInOut::linearInOut;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = linearInOut(0.5); }
  `;
  const result = await lygiaTestCompute(src);
  expectCloseTo([0.5], result);
});

test("quadraticIn", async () => {
  const src = `
    import lygia::animation::easing::quadraticIn::quadraticIn;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = quadraticIn(0.5); }
  `;
  const result = await lygiaTestCompute(src);
  expectCloseTo([0.25], result);
});

test("quadraticOut", async () => {
  const src = `
    import lygia::animation::easing::quadraticOut::quadraticOut;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = quadraticOut(0.5); }
  `;
  const result = await lygiaTestCompute(src);
  expectCloseTo([0.75], result);
});

test("quadraticInOut", async () => {
  const src = `
    import lygia::animation::easing::quadraticInOut::quadraticInOut;
    @compute @workgroup_size(1)
    fn foo() {
      // Test multiple points to show quadratic curve
      test::results[0] = vec4f(
        quadraticInOut(0.0),   // Start
        quadraticInOut(0.25),  // First quarter
        quadraticInOut(0.75),  // Third quarter
        quadraticInOut(1.0)    // End
      );
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([0.0, 0.125, 0.875, 1.0], result);
});

test("quarticIn", async () => {
  const src = `
    import lygia::animation::easing::quarticIn::quarticIn;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = quarticIn(0.5); }
  `;
  const result = await lygiaTestCompute(src);
  expectCloseTo([0.0625], result);
});

test("quarticOut", async () => {
  const src = `
    import lygia::animation::easing::quarticOut::quarticOut;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = quarticOut(0.5); }
  `;
  const result = await lygiaTestCompute(src);
  expectCloseTo([0.9375], result);
});

test("quarticInOut", async () => {
  const src = `
    import lygia::animation::easing::quarticInOut::quarticInOut;
    @compute @workgroup_size(1)
    fn foo() {
      // Test multiple points to show quartic curve
      test::results[0] = vec4f(
        quarticInOut(0.0),   // Start
        quarticInOut(0.25),  // First quarter
        quarticInOut(0.75),  // Third quarter
        quarticInOut(1.0)    // End
      );
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([0.0, 0.03125, 0.96875, 1.0], result);
});

test("quinticIn", async () => {
  const src = `
    import lygia::animation::easing::quinticIn::quinticIn;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = quinticIn(0.5); }
  `;
  const result = await lygiaTestCompute(src);
  expectCloseTo([0.03125], result);
});

test("quinticOut", async () => {
  const src = `
    import lygia::animation::easing::quinticOut::quinticOut;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = quinticOut(0.5); }
  `;
  const result = await lygiaTestCompute(src);
  expectCloseTo([1.03125], result);
});

test("quinticInOut", async () => {
  const src = `
    import lygia::animation::easing::quinticInOut::quinticInOut;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = quinticInOut(0.5); }
  `;
  const result = await lygiaTestCompute(src);
  expectCloseTo([1.5], result);
});

test("sineIn", async () => {
  const src = `
    import lygia::animation::easing::sineIn::sineIn;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = sineIn(0.5); }
  `;
  const result = await lygiaTestCompute(src);
  expectCloseTo([0.2929], result);
});

test("sineOut", async () => {
  const src = `
    import lygia::animation::easing::sineOut::sineOut;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = sineOut(0.5); }
  `;
  const result = await lygiaTestCompute(src);
  expectCloseTo([Math.SQRT1_2], result);
});

test("sineInOut", async () => {
  const src = `
    import lygia::animation::easing::sineInOut::sineInOut;
    @compute @workgroup_size(1)
    fn foo() {
      // Test multiple points to show sine curve
      test::results[0] = vec4f(
        sineInOut(0.0),   // Start
        sineInOut(0.25),  // First quarter
        sineInOut(0.75),  // Third quarter
        sineInOut(1.0)    // End
      );
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([0.0, 0.1464, 0.8536, 1.0], result);
});
