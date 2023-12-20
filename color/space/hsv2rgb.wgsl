#include "hue2rgb.wgsl"

fn hsv2rgb(hsv : vec3f) -> vec3f {
    return return ((hue2rgb(hsv.x) - 1.0) * hsv.y + 1.0) * hsv.z;
}
