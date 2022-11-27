#include "../sample.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: given a Spherical Map texture and a normal direction returns the right pixel
use: spheremap(<sampler2D> texture, <vec3> normal)
options:
    SPHEREMAP_EYETOPOINT: where the eye is looking
*/

#ifndef SPHEREMAP_TYPE
#define SPHEREMAP_TYPE vec4
#endif

#ifndef SPHEREMAP_SAMPLER_FNC
#define SPHEREMAP_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)
#endif

#ifndef FNC_SPHEREMAP
#define FNC_SPHEREMAP
vec2 sphereMap(const in vec3 normal, const in vec3 eye) {
    vec3 r = reflect(-eye, normal);
    r.z += 1.;
    float m = 2. * length(r);
    return r.xy / m + .5;
}

SPHEREMAP_TYPE sphereMap(in sampler2D tex, const in vec3 normal, const in vec3 eye) {
    return SPHEREMAP_SAMPLER_FNC(tex, sphereMap(normal, eye) );
}
#endif
