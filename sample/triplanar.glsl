#include "../sample.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: triplanar mapping
use: <vec4> sample2DCube(in <SAMPLER_TYPE> lut, in <vec3> xyz) 
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - SAMPLE_2DCUBE_CELL_SIZE
    - SAMPLE_2DCUBE_CELLS_PER_SIDE: defaults to 8
    - SAMPLE_2DCUBE_FNC
*/


#ifndef SAMPLE_TRIPLANAR_FNC
#define SAMPLE_TRIPLANAR_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)
#endif

#ifndef FNC_SAMPLETRIPLANAR
#define FNC_SAMPLETRIPLANAR
vec3 sampleTriplanar(SAMPLER_TYPE tex, in vec3 d) {
    vec3 colx = SAMPLE_TRIPLANAR_FNC(tex, d.yz).xyz;
    vec3 coly = SAMPLE_TRIPLANAR_FNC(tex, d.zx).xyz;
    vec3 colz = SAMPLE_TRIPLANAR_FNC(tex, d.xy).xyz;
    
    vec3 n = d*d;
    return (colx*n.x + coly*n.y + colz*n.z)/(n.x+n.y+n.z);
}

// iq's cubemap function
vec3 sampleTriplanar(SAMPLER_TYPE tex, in vec3 d, in float s) {
    vec3 colx = SAMPLE_TRIPLANAR_FNC(tex, 0.5 + s*d.yz/d.x).xyz;
    vec3 coly = SAMPLE_TRIPLANAR_FNC(tex, 0.5 + s*d.zx/d.y).xyz;
    vec3 colz = SAMPLE_TRIPLANAR_FNC(tex, 0.5 + s*d.xy/d.z).xyz;
    
    vec3 n = d*d;
    
    return (colx*n.x + coly*n.y + colz*n.z)/(n.x+n.y+n.z);
}
#endif