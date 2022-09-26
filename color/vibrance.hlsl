/*
original_author: Brad Larson
description: vibrance from https://github.com/BradLarson/GPUImage/blob/master/framework/Source/GPUImageVibranceFilter.m Vibrance is a smart-tool which cleverly increases the intensity of the more muted colors and leaves the already well-saturated colors alone. Prevents skin tones from becoming overly saturated and unnatural. 
use: <float3|float4> vibrance(<float3|float4> color, <float> v) 
*/

#ifndef FNC_VIBRANCE
#define FNC_VIBRANCE
float3 vibrance(in float3 color, in float v) {
    float average = (color.r + color.g + color.b) / 3.0;
    float mx = max(color.r, max(color.g, color.b));
    float amt = (mx - average) * (-v * 3.0);
    return lerp(color.rgb, float3(mx, mx, mx), amt);
}
float4 vibrance(in float4 color, in float v) { return float4( vibrance(color.rgb, v), color.a); }
#endif