#include "rgb2hsv.wgsl"

fn rgb2hue(color: vec3f) -> f32 {
    return rgb2hsv(color).x;
}