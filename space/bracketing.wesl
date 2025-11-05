import lygia::math::consts::PI;
import lygia::math::consts::TWO_PI;

/*
contributors: Huw Bowles ( @hdb1 )
description: |
    Bracketing technique maps a texture to a plane using any arbitrary 2D vector field to give orientation. From https://www.shadertoy.com/view/NddcDr

    WESL conditionals:
    - @if(BRACKETING_ANGLE_DELTA): Uses custom angle delta constant, otherwise uses PI/20.0 default
use: bracketing(dir: vec2f) -> BracketingResult
license: MIT license (MIT) Copyright Huw Bowles May 2022
*/

@if(BRACKETING_ANGLE_DELTA)
import constants::BRACKETING_ANGLE_DELTA;

// Parameter for bracketing - bracket size in radians. Large values create noticeable linear structure,
// small values prone to simply replicating the issues with the brute force approach. In my use cases it
// was quick and easy to find a sweet spot.
const BRACKETING_ANGLE_DELTA_DEFAULT: f32 = PI / 20.0;

// Vector field direction is used to drive UV coordinate frame, but instead
// of directly taking the vector directly, take two samples of the texture
// using coordinate frames at snapped angles, and then blend them based on
// the angle of the original vector.
struct BracketingResult {
    vAxis0: vec2f,
    vAxis1: vec2f,
    blendAlpha: f32,
};

fn bracketing(dir: vec2f) -> BracketingResult {
    // Heading angle of the original vector field direction
    let angle = atan2(dir.y, dir.x) + TWO_PI;

    @if(BRACKETING_ANGLE_DELTA)
    let AngleDelta = constants::BRACKETING_ANGLE_DELTA;
    @else
    let AngleDelta = BRACKETING_ANGLE_DELTA_DEFAULT;

    // Snap to a first canonical direction by subtracting fractional angle
    let fractional = angle % AngleDelta;
    let angle0 = angle - fractional;

    // Compute one V axis of UV frame
    let vAxis0 = vec2f(cos(angle0), sin(angle0));

    // Compute the next V axis by rotating by the snap angle size
    let RotateByAngleDelta = mat2x2f(
        cos(AngleDelta), sin(AngleDelta),
        -sin(AngleDelta), cos(AngleDelta)
    );

    let vAxis1 = RotateByAngleDelta * vAxis0;

    // Blend to get final result, based on how close the vector was to the first snapped angle
    let blendAlpha = fractional / AngleDelta;

    return BracketingResult(vAxis0, vAxis1, blendAlpha);
}
