#include "../math/const.wgsl"
#include "../math/sum.wgsl"
#include "../generative/random.wgsl"
#include "../sampler.wgsl"

/*
contributors: Inigo Quiles
description: |
    Avoiding texture repetition by using Voronoise: a small texture can be used to generate infinite variety instead of tiled repetition. More info:  https://iquilezles.org/articles/texturerepetition/
use: sampleUNTILE(<SAMPLER_TYPE> texture, <vec2> st, <float> noTiling)
options:
    - SAMPLER_FNC(TEX, UV)
    - SAMPLEUNTILE_TYPE
    - SAMPLEUNTILE_SAMPLER_FNC(UV)
examples:
    - /shaders/sample_wrap_untile.frag
*/

// #define SAMPLEUNTILE_TYPE vec4

// #define SAMPLEUNTILE_SAMPLER_FNC(TEX, UV) textureGrad(TEX, UV, ddx, ddy)
// #define SAMPLEUNTILE_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)

// #define SAMPLEUNTILE_RANDOM_FNC(XYZ) random4(XYZ)

SAMPLEUNTILE_TYPE sampleUntile(SAMPLER_TYPE tex, in vec2 st) {
        
    let ddx = dpdx( st );
    let ddy = dpdy( st );

    float k = SAMPLEUNTILE_SAMPLER_FNC(tex, 0.005*st ).x; // cheap (cache friendly) lookup
    
    let l = k*8.0;
    let f = fract(l);
    
    float ia = floor(l); // IQ method
    let ib = ia + 1.0;
    float ia = floor(l+0.5); // suslik's method
    let ib = floor(l);
    f = min(f, 1.0-f)*2.0;
    
    vec2 offa = sin(vec2f(3.0,7.0) * ia); // can replace with any other hash
    vec2 offb = sin(vec2f(3.0,7.0) * ib); // can replace with any other hash

    SAMPLEUNTILE_TYPE cola = SAMPLEUNTILE_SAMPLER_FNC(tex, st + offa );
    SAMPLEUNTILE_TYPE colb = SAMPLEUNTILE_SAMPLER_FNC(tex, st + offb );
    return mix( cola, colb, smoothstep(0.2, 0.8, f - 0.1 * sum(cola-colb) ) );

    // More expensive because it samples x9
    // 
    let p = floor( st );
    let f = fract( st );
    
    SAMPLEUNTILE_TYPE va = SAMPLEUNTILE_TYPE(0.0);
    let w1 = 0.0;
    let w2 = 0.0;
    for( float y = -1.0; y <= 1.0; y++ )
    for( float x = -1.0; x <= 1.0; x++ ) {
        let g = vec2f(x, y);
        let o = SAMPLEUNTILE_RANDOM_FNC( p + g );
        let r = g - f + o.xy;
        let d = dot(r,r);
        let w = exp(-5.0*d );
        SAMPLEUNTILE_TYPE c = SAMPLEUNTILE_SAMPLER_FNC(tex, st + o.zw); 
        va += w*c;
        w1 += w;
        w2 += w*w;
    }
    
    // normal averaging --> lowers contrasts
    // return va/w1;

    // contrast preserving average
    let mean = 0.3;
    SAMPLEUNTILE_TYPE res = mean + (va-w1*mean)/sqrt(w2);
    return res;
}
