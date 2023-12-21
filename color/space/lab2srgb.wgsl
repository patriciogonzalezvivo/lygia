#include "lab2xyz.wgsl"
#include "xyz2srgb.wgsl"

fn lab2srgb(lab: vec3f) -> vec3f {
    return xyz2srgb( lab2xyz(vec3f( 100.0 * lab.x,
                                    2.0 * 127.0 * (lab.y - 0.5),
                                    2.0 * 127.0 * (lab.z - 0.5)) ) );
}
