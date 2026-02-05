import { test } from "vitest";
import { expectCloseTo, lygiaTestCompute, lygiaTestWesl } from "./testUtil.ts";

await lygiaTestWesl("test/wesl/shaders/color_space_roundtrip.test");

// xyY roundtrip needs relaxed precision (0.01 tolerance)
test("rgb2xyY4 -> xyY2rgb4 roundtrip (note: precision issues in xyY)", async () => {
  const src = `
     import lygia::color::space::rgb2xyY::rgb2xyY4;
     import lygia::color::space::xyY2rgb::xyY2rgb4;

     @compute @workgroup_size(1)
     fn foo() {
       // Using a brighter color to avoid precision issues at low values
       let original = vec4f(0.9, 0.8, 0.7, 0.6);
       let xyY = rgb2xyY4(original);
       let back = xyY2rgb4(xyY);
       test::results[0] = back;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Note: xyY conversion chain has some precision loss
  // The conversion goes: RGB -> XYZ (0-100) -> xyY -> XYZ (0-100) -> RGB
  // which accumulates rounding errors
  expectCloseTo([0.9, 0.8, 0.7, 0.6], result, 0.01);
});
