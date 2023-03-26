#include "space/rgb2luma.glsl"

fn luma(color: vec3<f32>) -> f32 {
    return rgb2luma(color);
}
