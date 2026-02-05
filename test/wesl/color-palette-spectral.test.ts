import { lygiaTestWesl } from "./testUtil.ts";
import "./shaders/color_palette_spectral.test.wesl?raw"; // trigger watch mode rebuild

await lygiaTestWesl("test/wesl/shaders/color_palette_spectral.test");
