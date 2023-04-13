#include "../math/rotate2d.glsl"

fn rotate(st: vec2<f32>, radians: f32) -> vec2<f32> {
    return rotate2d(radians) * (st - 0.5) + 0.5;
}
