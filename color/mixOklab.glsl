/*
original_author: Bjorn Ottosson (@bjornornorn), Inigo Quiles
description: oklab to linear RGB https://bottosson.github.io/posts/oklab/
use: <vec3\vec4> mixOklab(<vec3|vec4> colorA, <vec3|vec4> colorB, float pct)
examples:
    - /shaders/color_mix.frag
*/

#ifndef FNC_MIXOKLAB
#define FNC_MIXOKLAB
vec3 mixOklab( vec3 colA, vec3 colB, float h ) {
    // https://bottosson.github.io/posts/oklab
    const mat3 kCONEtoLMS = mat3(                
         0.4121656120,  0.2118591070,  0.0883097947,
         0.5362752080,  0.6807189584,  0.2818474174,
         0.0514575653,  0.1074065790,  0.6302613616);
    const mat3 kLMStoCONE = mat3(
         4.0767245293, -1.2681437731, -0.0041119885,
        -3.3072168827,  2.6093323231, -0.7034763098,
         0.2307590544, -0.3411344290,  1.7068625689);
                    
    // rgb to cone (arg of pow can't be negative)
    vec3 lmsA = pow( kCONEtoLMS*colA, vec3(1.0/3.0) );
    vec3 lmsB = pow( kCONEtoLMS*colB, vec3(1.0/3.0) );
    // lerp
    vec3 lms = mix( lmsA, lmsB, h );
    
    // gain in the middle (no oaklab anymore, but looks better?)
    // lms *= 1.0+0.2*h*(1.0-h);

    // cone to rgb
    return kLMStoCONE*(lms*lms*lms);
}

vec4 mixOklab( vec4 colA, vec4 colB, float h ) {
    return vec4( mixOklab(colA.rgb, colB.rgb, h), mix(colA.a, colB.a, h) );
}
#endif