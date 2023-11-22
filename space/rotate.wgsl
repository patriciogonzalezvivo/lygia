#include "../math/rotate2d.glsl"

fn rotate(st: vec2f, radians: f32) -> vec2f {
    return rotate2d(radians) * (st - 0.5) + 0.5;
}
