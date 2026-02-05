import { lygiaTestWesl } from "./testUtil.ts";
import "./shaders/color_luma.test.wesl?raw"; // not used, but nice to trigger watch mode rebuild in vitest

await lygiaTestWesl("test/wesl/shaders/color_luma.test");
