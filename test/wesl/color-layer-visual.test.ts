import { afterAll, beforeAll, test } from "vitest";
import { imageMatcher } from "vitest-image-snapshot";
import { destroySharedDevice, getGPUDevice } from "wesl-test";
import { expectBlend } from "./testUtil.ts";

imageMatcher();

beforeAll(async () => {
  await getGPUDevice(); // Initialize shared device
});

afterAll(() => {
  destroySharedDevice();
});

// Contrast modes
test("hardLight blend mode", async () => {
  await expectBlend(
    `
    import lygia::color::layer::hardLightSourceOver::layerHardLightSourceOver4;
    import lygia::test::wesl_util::blendInputs::blendInputs;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let inputs = blendInputs(pos);
      return layerHardLightSourceOver4(inputs.src, inputs.dst);
    }
  `,
    "layer-hardlight",
  );
});

test("softLight blend mode", async () => {
  await expectBlend(
    `
    import lygia::color::layer::softLightSourceOver::layerSoftLightSourceOver4;
    import lygia::test::wesl_util::blendInputs::blendInputs;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let inputs = blendInputs(pos);
      return layerSoftLightSourceOver4(inputs.src, inputs.dst);
    }
  `,
    "layer-softlight",
  );
});

test("vividLight blend mode", async () => {
  await expectBlend(
    `
    import lygia::color::layer::vividLightSourceOver::layerVividLightSourceOver4;
    import lygia::test::wesl_util::blendInputs::blendInputs;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let inputs = blendInputs(pos);
      return layerVividLightSourceOver4(inputs.src, inputs.dst);
    }
  `,
    "layer-vividlight",
  );
});

test("linearLight blend mode", async () => {
  await expectBlend(
    `
    import lygia::color::layer::linearLightSourceOver::layerLinearLightSourceOver4;
    import lygia::test::wesl_util::blendInputs::blendInputs;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let inputs = blendInputs(pos);
      return layerLinearLightSourceOver4(inputs.src, inputs.dst);
    }
  `,
    "layer-linearlight",
  );
});

test("pinLight blend mode", async () => {
  await expectBlend(
    `
    import lygia::color::layer::pinLightSourceOver::layerPinLightSourceOver4;
    import lygia::test::wesl_util::blendInputs::blendInputs;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let inputs = blendInputs(pos);
      return layerPinLightSourceOver4(inputs.src, inputs.dst);
    }
  `,
    "layer-pinlight",
  );
});

test("hardMix blend mode", async () => {
  await expectBlend(
    `
    import lygia::color::layer::hardMixSourceOver::layerHardMixSourceOver4;
    import lygia::test::wesl_util::blendInputs::blendInputs;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let inputs = blendInputs(pos);
      return layerHardMixSourceOver4(inputs.src, inputs.dst);
    }
  `,
    "layer-hardmix",
  );
});

// Darken modes
test("colorBurn blend mode", async () => {
  await expectBlend(
    `
    import lygia::color::layer::colorBurnSourceOver::layerColorBurnSourceOver4;
    import lygia::test::wesl_util::blendInputs::blendInputs;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let inputs = blendInputs(pos);
      return layerColorBurnSourceOver4(inputs.src, inputs.dst);
    }
  `,
    "layer-colorburn",
  );
});

test("linearBurn blend mode", async () => {
  await expectBlend(
    `
    import lygia::color::layer::linearBurnSourceOver::layerLinearBurnSourceOver4;
    import lygia::test::wesl_util::blendInputs::blendInputs;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let inputs = blendInputs(pos);
      return layerLinearBurnSourceOver4(inputs.src, inputs.dst);
    }
  `,
    "layer-linearburn",
  );
});

// Lighten modes
test("colorDodge blend mode", async () => {
  await expectBlend(
    `
    import lygia::color::layer::colorDodgeSourceOver::layerColorDodgeSourceOver4;
    import lygia::test::wesl_util::blendInputs::blendInputs;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let inputs = blendInputs(pos);
      return layerColorDodgeSourceOver4(inputs.src, inputs.dst);
    }
  `,
    "layer-colordodge",
  );
});

test("linearDodge blend mode", async () => {
  await expectBlend(
    `
    import lygia::color::layer::linearDodgeSourceOver::layerLinearDodgeSourceOver4;
    import lygia::test::wesl_util::blendInputs::blendInputs;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let inputs = blendInputs(pos);
      return layerLinearDodgeSourceOver4(inputs.src, inputs.dst);
    }
  `,
    "layer-lineardodge",
  );
});

// HSL modes
test("color blend mode", async () => {
  await expectBlend(
    `
    import lygia::color::layer::colorSourceOver::layerColorSourceOver4;
    import lygia::test::wesl_util::blendInputs::blendInputs;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let inputs = blendInputs(pos);
      return layerColorSourceOver4(inputs.src, inputs.dst);
    }
  `,
    "layer-color",
  );
});

test("hue blend mode", async () => {
  await expectBlend(
    `
    import lygia::color::layer::hueSourceOver::layerHueSourceOver4;
    import lygia::test::wesl_util::blendInputs::blendInputs;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let inputs = blendInputs(pos);
      return layerHueSourceOver4(inputs.src, inputs.dst);
    }
  `,
    "layer-hue",
  );
});

test("saturation blend mode", async () => {
  await expectBlend(
    `
    import lygia::color::layer::saturationSourceOver::layerSaturationSourceOver4;
    import lygia::test::wesl_util::blendInputs::blendInputs;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let inputs = blendInputs(pos);
      return layerSaturationSourceOver4(inputs.src, inputs.dst);
    }
  `,
    "layer-saturation",
  );
});

test("luminosity blend mode", async () => {
  await expectBlend(
    `
    import lygia::color::layer::luminositySourceOver::layerLuminositySourceOver4;
    import lygia::test::wesl_util::blendInputs::blendInputs;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let inputs = blendInputs(pos);
      return layerLuminositySourceOver4(inputs.src, inputs.dst);
    }
  `,
    "layer-luminosity",
  );
});

// Other modes
test("average blend mode", async () => {
  await expectBlend(
    `
    import lygia::color::layer::averageSourceOver::layerAverageSourceOver4;
    import lygia::test::wesl_util::blendInputs::blendInputs;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let inputs = blendInputs(pos);
      return layerAverageSourceOver4(inputs.src, inputs.dst);
    }
  `,
    "layer-average",
  );
});

test("negation blend mode", async () => {
  await expectBlend(
    `
    import lygia::color::layer::negationSourceOver::layerNegationSourceOver4;
    import lygia::test::wesl_util::blendInputs::blendInputs;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let inputs = blendInputs(pos);
      return layerNegationSourceOver4(inputs.src, inputs.dst);
    }
  `,
    "layer-negation",
  );
});

test("reflect blend mode", async () => {
  await expectBlend(
    `
    import lygia::color::layer::reflectSourceOver::layerReflectSourceOver4;
    import lygia::test::wesl_util::blendInputs::blendInputs;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let inputs = blendInputs(pos);
      return layerReflectSourceOver4(inputs.src, inputs.dst);
    }
  `,
    "layer-reflect",
  );
});

test("glow blend mode", async () => {
  await expectBlend(
    `
    import lygia::color::layer::glowSourceOver::layerGlowSourceOver4;
    import lygia::test::wesl_util::blendInputs::blendInputs;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let inputs = blendInputs(pos);
      return layerGlowSourceOver4(inputs.src, inputs.dst);
    }
  `,
    "layer-glow",
  );
});
