#include "rgb2hue.wgsl"

fn rgb2heat(c: vec3<f32>) -> f32 {
    return 1.025 - rgb2hue(c) * 1.538461538;
}
