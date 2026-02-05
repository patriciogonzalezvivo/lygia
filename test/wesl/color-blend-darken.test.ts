import { test } from "vitest";
import { lygiaTestWesl } from "./testUtil.ts";

await lygiaTestWesl("test/wesl/shaders/color_blend_darken.test");

test("");
