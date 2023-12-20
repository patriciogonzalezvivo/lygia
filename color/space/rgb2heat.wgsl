#include "rgb2hue.wgsl"

fn rgb2heat(c: vec3f) -> f32 { return 1.025 - rgb2hue(c) * 1.538461538; }
