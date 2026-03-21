#include "../sampler.wgsl"
/*
contributors: Alexander Griffis 
description: |
    Algorithmic paradigm, that approximate other algorithms in constant-time performance.
    Does not always compute the correct result for every pixel, although in practice errors 
    are few and the magnitude of errors is generally small
    Based on the work of Alexander Griffis in this project https://github.com/Yaazarai/2DGI/ 
    which were introduce by  Rong Guodong at an ACM symposium in 2006 https://www.comp.nus.edu.sg/~tants/jfa/i3d06.pdf 
use:
    - <float> jumpFloodCalcIterTotal(<vec2> res)
    - <float> jumpFloodCalcIter(<float> iterTotal, <float> frame)
    - <vec4>  jumpFloodEncode(<sampler2D> tex, <vec2> st)
    - <float> jumpFloodSdf(<sampler2D> tex, <vec2> st)
    - <vec4>  jumpFloodIterate(<sampler2D> tex, <vec2> st, <vec2> pixel, <float> iterTotal, <float> iterN)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - JUMPFLOOD_TYPE
    - JUMPFLOOD_SAMPLE_FNC(TEX, UV)
    - JUMPFLOOD_ENCODE_FNC(TEX, UV)
examples:
    - /shaders/morphological_alphaFill.frag
*/

// #define JUMPFLOOD_TYPE vec4

// #define JUMPFLOOD_ENCODE_FNC(VEC) vec4(VEC, 0.0, 1.0)

// #define JUMPFLOOD_DECODE_FNC(VEC) VEC.xy

// #define JUMPFLOOD_SAMPLE_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)

fn jumpFloodCalcIterTotal(res: vec2f) -> f32 { return ceil(log2(max(res.x, res.y)) / log2(2.0)); }
fn jumpFloodCalcIter(iterTotal: f32, frame: f32) -> f32 { return mod(frame, iterTotal); }

JUMPFLOOD_TYPE jumpFloodEncode(SAMPLER_TYPE tex, vec2 st) { return JUMPFLOOD_ENCODE_FNC(st * JUMPFLOOD_SAMPLE_FNC(tex, st).a); }
fn jumpFloodSdf(tex: SAMPLER_TYPE, st: vec2f) -> f32 { return distance(st, JUMPFLOOD_DECODE_FNC(JUMPFLOOD_SAMPLE_FNC(tex, st))); }

JUMPFLOOD_TYPE jumpFloodIterate(SAMPLER_TYPE tex, vec2 st, vec2 pixel, float jump) {
    vec2 offsets[9];
    offsets[0] = vec2f(-1.0, -1.0);
    offsets[1] = vec2f(-1.0, 0.0);
    offsets[2] = vec2f(-1.0, 1.0);
    offsets[3] = vec2f(0.0, -1.0);
    offsets[4] = vec2f(0.0, 0.0);
    offsets[5] = vec2f(0.0, 1.0);
    offsets[6] = vec2f(1.0, -1.0);
    offsets[7] = vec2f(1.0, 0.0);
    offsets[8] = vec2f(1.0, 1.0);
    
    let closest_dist = 9999999.9;
    JUMPFLOOD_TYPE closest_data = JUMPFLOOD_TYPE(0.0);
    for(int i = 0; i < 9; i++) {
        let xy = st + offsets[i] * pixel * jump;
        JUMPFLOOD_TYPE seed = JUMPFLOOD_SAMPLE_FNC(tex, xy);
        let seedpos = JUMPFLOOD_DECODE_FNC(seed);
        let dist = distance(seedpos, st);
        if (seedpos != vec2f(0.0) && dist <= closest_dist) {
            closest_dist = dist;
            closest_data = seed;
        }
    }
    return closest_data;
}

JUMPFLOOD_TYPE jumpFloodIterate(SAMPLER_TYPE tex, vec2 st, vec2 pixel, float iterTotal, float iterN) {
    return jumpFloodIterate(tex, st, pixel, pow(2.0, iterTotal - iterN - 1.0));
}

JUMPFLOOD_TYPE jumpFloodIterate(SAMPLER_TYPE tex, vec2 st, vec2 pixel, int iterTotal, int iterN) {
    return jumpFloodIterate(tex, st, pixel, pow(2.0, float(iterTotal - iterN - 1)));
}
