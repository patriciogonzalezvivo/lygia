#include "../space/rotate.glsl"

/*
author: Huw Bowles ( @hdb1 )
description: 'Bracketing' technique maps a texture to a plane using any arbitrary 2D vector field to give orientatio. From https://www.shadertoy.com/view/NddcDr
use: textureBracketing(<sampler2D> texture, <vec2> st, <vec2> direction [, <float> scale] )
options:
    TEXTUREBRACKETING_TYPE:
    TEXTUREBRACKETING_SAMPLE_FNC(UV):
    TEXTUREBRACKETING_ANGLE_DELTA:
    TEXTUREBRACKETING_REPLACE_DIVERGENCE
license: Copyright Huw Bowles May 2022 on MIT license
*/

// Parameter for bracketing - bracket size in radians. Large values create noticeable linear structure,
// small values prone to simply replicating the issues with the brute force approach. In my use cases it
// was quick and easy to find a sweet spot.
#ifndef TEXTUREBRACKETING_ANGLE_DELTA
#define TEXTUREBRACKETING_ANGLE_DELTA PI / 20.0
#endif

#ifndef TEXTUREBRACKETING_TYPE
#define TEXTUREBRACKETING_TYPE vec4
#endif

#ifndef TEXTUREBRACKETING_SAMPLE_FNC
#define TEXTUREBRACKETING_SAMPLE_FNC(UV) texture2D(tex, UV)
#endif

#ifndef FNC_TEXTUREBRACKETING
#define FNC_TEXTUREBRACKETING

// Vector field direction is used to drive UV coordinate frame, but instead
// of directly taking the vector directly, take two samples of the texture
// using coordinate frames at snapped angles, and then blend them based on
// the angle of the original vector.
void textureBracketing(vec2 dir, out vec2 vAxis0, out vec2 vAxis1, out float blendAlpha) {
    // Heading angle of the original vector field direction
    float angle = atan(dir.y, dir.x) + 2.0*PI;

    float AngleDelta = TEXTUREBRACKETING_ANGLE_DELTA;

    // Snap to a first canonical direction by subtracting fractional angle
    float fractional = mod(angle, AngleDelta);
    float angle0 = angle - fractional;
    
    // Compute one V axis of UV frame. Given angle0 is snapped, this could come from LUT, but would
    // need testing on target platform to verify that a LUT is faster.
    vAxis0 = vec2(cos(angle0), sin(angle0));

    // Compute the next V axis by rotating by the snap angle size

    mat2 RotateByAngleDelta = mat2(cos(AngleDelta), sin(AngleDelta), -sin(AngleDelta), cos(AngleDelta));
    vAxis1 = RotateByAngleDelta * vAxis0;

    // Blend to get final result, based on how close the vector was to the first snapped angle
    blendAlpha = fractional / AngleDelta;
}

TEXTUREBRACKETING_TYPE textureBracketing(sampler2D tex, vec2 st, vec2 dir, float scale) {
    vec2 vAxis0 = vec2(0.0);
    vec2 vAxis1 = vec2(0.0);
    float blendAlpha = 0.0;

    textureBracketing(dir, vAxis0, vAxis1, blendAlpha);
    
    // Compute the function for the two canonical directions
    vec2 uv0 = scale * rotate(st, vAxis0);
    vec2 uv1 = scale * rotate(st, vAxis1);
    
    // Now sample function/texture
    TEXTUREBRACKETING_TYPE sample0 = TEXTUREBRACKETING_SAMPLE_FNC(uv0);
    TEXTUREBRACKETING_TYPE sample1 = TEXTUREBRACKETING_SAMPLE_FNC(uv1);

    // Blend to get final result, based on how close the vector was to the first snapped angle
    TEXTUREBRACKETING_TYPE result = mix( sample0, sample1, blendAlpha);

#ifdef TEXTUREBRACKETING_REPLACE_DIVERGENCE
    float strength = smoothstep(0.0, 0.2, dot(dir, dir)*6.0);
    result = mix(   TEXTUREBRACKETING_SAMPLE_FNC(scale * rotate(st, -vec2(cos(0.5),sin(0.5)))), 
                    result, 
                    strength);
#endif
    
    return result;
}

TEXTUREBRACKETING_TYPE textureBracketing(sampler2D tex, vec2 st, vec2 dir) { return textureBracketing(tex, st, dir, 1.0); }
#endif