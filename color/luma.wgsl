#include "space/rgb2luma.glsl"

fn luma(color: vec3f) -> f32 {
    return rgb2luma(color);
}
