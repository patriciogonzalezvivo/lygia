#include "lab2xyz.wgsl"
#include "xyz2rgb.wgsl"

fn lab2rgb(lab : vec3f) -> vec3f {
    return xyz2rgb( lab2xyz( vec3f( 100.0 * lab.x,
                                    2.0 * 127.0 * (lab.y - 0.5),
                                    2.0 * 127.0 * (lab.z - 0.5)) ) );
}
