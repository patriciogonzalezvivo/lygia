#include "rgb2hsv.wgsl"

fn rgb2hue(color: vec3<f32>) -> f32 {
    return rgb2hsv(color).x;
}