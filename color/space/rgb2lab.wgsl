#include "rgb2xyz.wgsl"
#include "xyz2lab.wgsl"

fn rgb2lab(c: vec3f ) -> vec3f {
    let lab = xyz2lab(rgb2xyz(c));
    return vec3f(   lab.x / 100.0,
                        0.5 + 0.5 * (lab.y / 127.0),
                        0.5 + 0.5 * (lab.z / 127.0));
}
