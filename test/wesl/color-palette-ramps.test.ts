import { lygiaTestWesl } from "./testUtil.ts";
import "./shaders/color_palette_ramps_test.wesl?raw"; // trigger watch mode rebuild

await lygiaTestWesl("test/wesl/shaders/color_palette_ramps_test");
