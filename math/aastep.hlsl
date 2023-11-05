/*
contributors: Ronja (@totallyRonja)
description: Performs a smoothstep using standard derivatives for anti-aliased edges at any level of magnification. https://www.ronja-tutorials.com/post/046-fwidth/#non-aliased-step
use: aastep(<float> threshold, <float> value)
*/

#ifndef FNC_AASTEP
#define FNC_AASTEP
float aastep(float compValue, float gradient){
    float halfChange = fwidth(gradient) / 2;
    //base the range of the inverse lerp on the change over one pixel
    float lowerEdge = compValue - halfChange;
    float upperEdge = compValue + halfChange;
    //do the inverse interpolation
    float stepped = (gradient - lowerEdge) / (upperEdge - lowerEdge);
    stepped = saturate(stepped);
    return stepped;
}
#endif

