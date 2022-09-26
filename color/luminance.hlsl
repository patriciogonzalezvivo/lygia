/*
function: luminance
description: Computes the luminance of the specified linear RGB color using the luminance coefficients from Rec. 709.
use: luminance(<float3|float4> color)
*/

#ifndef FNC_LUMINANCE
#define FNC_LUMINANCE
float luminance(in float3 _linear) { return dot(_linear, float3(0.2126, 0.7152, 0.0722)); }
float luminance(in float4 _linear) { return luminance( _linear.rgb ); }
#endif