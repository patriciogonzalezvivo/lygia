#include "common/schlick.glsl"

fn fresnel(f0: f32, NoV: f32) -> f32 {
    return schlick(f0, 1.0, NoV);
}
