#include "2DCube.glsl"
#include "../space/scale.glsl"
#include "../sdf/boxSDF.glsl"
#include "../sdf/opIntersection.glsl"

#ifndef FNC_SAMPLE3DSDF
#define FNC_SAMPLE3DSDF
vec4 sample3DSdf(sampler2D tex, vec3 pos) {
    pos += .5;
    pos = scale(pos, .25);
    vec4 s = sample2DCube(tex, pos);
    float sdf = (s.a * 2.0 - 1.0) * 1.5;
    return vec4(s.rgb, opIntersection( boxSDF(pos - 0.5, vec3(.5)), sdf));
}
#endif