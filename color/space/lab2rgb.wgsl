#include "lab2xyz.wgsl"
#include "xyz2rgb.wgsl"

fn lab2rgb(lab : vec3<f32>) -> vec3<f32> {
    return xyz2rgb( lab2xyz( vec3<f32>( 100.0 * lab.x,
                                        2.0 * 127.0 * (lab.y - 0.5),
                                        2.0 * 127.0 * (lab.z - 0.5)) ) );
}
