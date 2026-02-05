import { test } from "vitest";
import { lygiaTestWesl } from "./testUtil.ts";

await lygiaTestWesl("test/wesl/shaders/color_blend_dodge.test");

test("");
