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
#define SPHEREMAP_SAMPLER_FNC(POS_UV) texture2D(tex, POS_UV)
#endif

#ifndef FNC_SPHEREMAP
#define FNC_SPHEREMAP
vec2 sphereMap(vec3 normal, vec3 eye) {
    // vec3 reflected = reflect(eye, normal);
    // float m = 2.8284271247461903 * sqrt( reflected.z+1.0 );
    // return reflected.xy / m + 0.5;
    vec3 r = reflect(eye, normal);
    r.z += 1.;
    float m = 2. * length(r);
    vec2 uv = r.xy / m + .5;
}

SPHEREMAP_TYPE sphereMap(in sampler2D tex, in vec3 normal, in vec3 eye) {
    return SPHEREMAP_SAMPLER_FNC( sphereMap(normal, eye) );
}
#endif
