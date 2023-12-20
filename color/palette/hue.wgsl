#include "../../math/mod.wgsl"
fn hue(x: f32, r: f32) -> vec3f { 
    let v = abs( mod3(x + vec3f(0.0,1.0,2.0) * r, vec3f(1.0)) * 2.0 - 1.0); 
    return v * v * (3.0 - 2.0 * v);
}