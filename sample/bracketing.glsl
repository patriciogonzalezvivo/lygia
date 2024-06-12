#include "../space/rotate.glsl"
#include "../space/bracketing.glsl"

#include "../sampler.glsl"

/*
contributors: Huw Bowles
description: |
    'Bracketing' technique maps a texture to a plane using any arbitrary 2D vector field to give orientatio. From https://www.shadertoy.com/view/NddcDr
use: sampleBracketing(<SAMPLER_TYPE> texture, <vec2> st, <vec2> direction [, <float> scale] )
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - SAMPLEBRACKETING_TYPE:
    - SAMPLEBRACKETING_SAMPLER_FNC(UV):
    - SAMPLEBRACKETING_REPLACE_DIVERGENCE: 
examples:
    - /shaders/sample_bracketing.frag
license: MIT license (MIT) Copyright Huw Bowles May 2022
*/

#ifndef SAMPLEBRACKETING_TYPE
#define SAMPLEBRACKETING_TYPE vec4
#endif

#ifndef SAMPLEBRACKETING_SAMPLER_FNC
#define SAMPLEBRACKETING_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)
#endif

#ifndef FNC_SAMPLEBRACKETING
#define FNC_SAMPLEBRACKETING

SAMPLEBRACKETING_TYPE sampleBracketing(SAMPLER_TYPE tex, vec2 st, vec2 dir, float scale) {
    vec2 vAxis0 = vec2(0.0);
    vec2 vAxis1 = vec2(0.0);
    float blendAlpha = 0.0;

    bracketing(dir, vAxis0, vAxis1, blendAlpha);
    
    // Compute the function for the two canonical directions
    vec2 uv0 = scale * rotate(st, vAxis0);
    vec2 uv1 = scale * rotate(st, vAxis1);
    
    // Now sample function/texture
    SAMPLEBRACKETING_TYPE sample0 = SAMPLEBRACKETING_SAMPLER_FNC(tex, uv0);
    SAMPLEBRACKETING_TYPE sample1 = SAMPLEBRACKETING_SAMPLER_FNC(tex, uv1);

    // Blend to get final result, based on how close the vector was to the first snapped angle
    SAMPLEBRACKETING_TYPE result = mix( sample0, sample1, blendAlpha);

#ifdef SAMPLEBRACKETING_REPLACE_DIVERGENCE
    float strength = smoothstep(0.0, 0.2, dot(dir, dir)*6.0);
    result = mix(   SAMPLEBRACKETING_SAMPLER_FNC(tex, scale * rotate(st, -vec2(cos(0.5),sin(0.5)))), 
                    result, 
                    strength);
#endif
    
    return result;
}

SAMPLEBRACKETING_TYPE sampleBracketing(SAMPLER_TYPE tex, vec2 st, vec2 dir) { return sampleBracketing(tex, st, dir, 1.0); }
#endif