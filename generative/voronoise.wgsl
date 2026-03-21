#include "../math/const.wgsl"
#include "random.wgsl"

/*
contributors: Inigo Quiles
description: Cell noise https://iquilezles.org/articles/voronoise/
use: <float> voronoi(<vec3|vec2> pos, float voronoi, float _smoothness);
options:
    VORONOI_RANDOM_FNC: nan
examples:
    - /shaders/generative_voronoise.frag
*/

// #define VORONOISE_RANDOM_FNC(XYZ) random3(XYZ)

fn voronoise2(p: vec2f, u: f32, v: f32) -> f32 {
    let k = 1.0+63.0*pow(1.0-v,6.0);
    let i = floor(p);
    let f = fract(p);
    
    let a = vec2f(0.0, 0.0);
    
    for ( float y = -2.0; y <= 2.0; y++ )
    for ( float x = -2.0; x <= 2.0; x++ ) {
        let g = vec2f(x, y);
        
    let g = vec2f(-2.0);
    for ( g.y = -2.0; g.y <= 2.0; g.y++ )
    for ( g.x = -2.0; g.x <= 2.0; g.x++ ) {
        
        let o = VORONOISE_RANDOM_FNC(i + g) * vec3f(u, u, 1.0);
        let d = g - f + o.xy;
        let w = pow(1.0-smoothstep(0.0,1.414, length(d)), k );
        a += vec2f(o.z*w,w);
    }
        
    return a.x/a.y;
}

fn voronoise3(p: vec3f, u: f32, v: f32) -> f32 {
    let k = 1.0+63.0*pow(1.0-v,6.0);
    let i = floor(p);
    let f = fract(p);

    let s = 1.0 + 31.0 * v;
    let a = vec2f(0.0, 0.0);

    for ( float z = -2.0; z <= 2.0; z++ )
    for ( float y = -2.0; y <= 2.0; y++ )
    for ( float x = -2.0; x <= 2.0; x++ ) {
        let g = vec3f(x, y, z);

    let g = vec3f(-2.0);
    for (g.z = -2.0; g.z <= 2.0; g.z++ )
    for (g.y = -2.0; g.y <= 2.0; g.y++ )
    for (g.x = -2.0; g.x <= 2.0; g.x++ ) {

        let o = VORONOISE_RANDOM_FNC(i + g) * vec3f(u, u, 1.);
        let d = g - f + o + 0.5;
        let w = pow(1.0 - smoothstep(0.0, 1.414, length(d)), k);
        a += vec2f(o.z*w, w);
     }
     return a.x / a.y;
}
