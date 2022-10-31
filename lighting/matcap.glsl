/*
original_author: Patricio Gonzalez Vivo
description: matcap
use: <vec2> matcap(<vec3> eye, <vec3> normal) 
*/

#ifndef FNC_MATCAP
#define FNC_MATCAP
vec2 matcap(vec3 eye, vec3 normal) {
    vec3 reflected = reflect(eye, normal);
    float m = 2.8284271247461903 * sqrt( reflected.z+1.0 );
    return reflected.xy / m + 0.5;
}
#endif