#include "rgb2xyz.glsl"
#include "xyz2lab.glsl"

fn rgb2lab(c: vec3<f32> ) -> vec3<f32> {
    let lab = xyz2lab(rgb2xyz(c));
    return vec3<f32>(   lab.x / 100.0,
                        0.5 + 0.5 * (lab.y / 127.0),
                        0.5 + 0.5 * (lab.z / 127.0));
}
