#include "../math/const.wgsl"

/*
contributors: Huw Bowles ( @hdb1 )
description: Bracketing technique maps a texture to a plane using any arbitrary 2D vector field to give orientation. From https://www.shadertoy.com/view/NddcDr
use: sampleBracketing(<SAMPLER_TYPE> texture, <vec2> st, <vec2> direction [, <float> scale] )
options:
    - BRACKETING_ANGLE_DELTA
license: MIT license (MIT) Copyright Huw Bowles May 2022
*/

// Parameter for bracketing - bracket size in radians. Large values create noticeable linear structure,
// small values prone to simply replicating the issues with the brute force approach. In my use cases it
// was quick and easy to find a sweet spot.
// #define BRACKETING_ANGLE_DELTA PI / 20.0

// Vector field direction is used to drive UV coordinate frame, but instead
// of directly taking the vector directly, take two samples of the texture
// using coordinate frames at snapped angles, and then blend them based on
// the angle of the original vector.
fn bracketing(dir: vec2f, vAxis0: vec2f, vAxis1: vec2f, blendAlpha: f32) {
    // Heading angle of the original vector field direction
    let angle = atan(dir.y, dir.x) + TWO_PI;

    let AngleDelta = BRACKETING_ANGLE_DELTA;

    // Snap to a first canonical direction by subtracting fractional angle
    let fractional = mod(angle, AngleDelta);
    let angle0 = angle - fractional;
    
    // Compute one V axis of UV frame. Given angle0 is snapped, this could come from LUT, but would
    // need testing on target platform to verify that a LUT is faster.
    vAxis0 = vec2f(cos(angle0), sin(angle0));

    // Compute the next V axis by rotating by the snap angle size
    mat2 RotateByAngleDelta = mat2x2<f32>( cos(AngleDelta), sin(AngleDelta), 
                                    -sin(AngleDelta), cos(AngleDelta));
                                    
    vAxis1 = RotateByAngleDelta * vAxis0;

    // Blend to get final result, based on how close the vector was to the first snapped angle
    blendAlpha = fractional / AngleDelta;
}
