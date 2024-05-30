#include "../math/const.glsl"
#include "random.glsl"

/*
contributors: Inigo Quiles
description: Cell noise https://iquilezles.org/articles/voronoise/
use: <float> voronoi(<vec3|vec2> pos, float voronoi, float _smoothness);
options:
    VORONOI_RANDOM_FNC: nan
examples:
    - /shaders/generative_voronoise.frag
*/

#ifndef VORONOISE_RANDOM_FNC 
#define VORONOISE_RANDOM_FNC(XYZ) random3(XYZ) 
#endif

#ifndef FNC_VORONOISE
#define FNC_VORONOISE
float voronoise( in vec2 p, in float u, float v) {
    float k = 1.0+63.0*pow(1.0-v,6.0);
    vec2 i = floor(p);
    vec2 f = fract(p);
    
    vec2 a = vec2(0.0, 0.0);
    
    #if defined(PLATFORM_WEBGL)
    for ( float y = -2.0; y <= 2.0; y++ )
    for ( float x = -2.0; x <= 2.0; x++ ) {
        vec2 g = vec2(x, y);
        
    #else
    vec2 g = vec2(-2.0);
    for ( g.y = -2.0; g.y <= 2.0; g.y++ )
    for ( g.x = -2.0; g.x <= 2.0; g.x++ ) {
        
    #endif
        vec3  o = VORONOISE_RANDOM_FNC(i + g) * vec3(u, u, 1.0);
        vec2  d = g - f + o.xy;
        float w = pow(1.0-smoothstep(0.0,1.414, length(d)), k );
        a += vec2(o.z*w,w);
    }
        
    return a.x/a.y;
}

float voronoise(vec3 p, float u, float v)  {
    float k = 1.0+63.0*pow(1.0-v,6.0);
    vec3 i = floor(p);
    vec3 f = fract(p);

    float s = 1.0 + 31.0 * v;
    vec2 a = vec2(0.0, 0.0);

    #if defined(PLATFORM_WEBGL)
    for ( float z = -2.0; z <= 2.0; z++ )
    for ( float y = -2.0; y <= 2.0; y++ )
    for ( float x = -2.0; x <= 2.0; x++ ) {
        vec3 g = vec3(x, y, z);

    #else

    vec3 g = vec3(-2.0);
    for (g.z = -2.0; g.z <= 2.0; g.z++ )
    for (g.y = -2.0; g.y <= 2.0; g.y++ )
    for (g.x = -2.0; g.x <= 2.0; g.x++ ) {
    #endif

        vec3 o = VORONOISE_RANDOM_FNC(i + g) * vec3(u, u, 1.);
        vec3 d = g - f + o + 0.5;
        float w = pow(1.0 - smoothstep(0.0, 1.414, length(d)), k);
        a += vec2(o.z*w, w);
     }
     return a.x / a.y;
}

#endif
