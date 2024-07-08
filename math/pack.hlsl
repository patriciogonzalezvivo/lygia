/*
contributors: Ricardo Caballero
description: |
    Pack a float into a 4D vector. From https://github.com/mrdoob/three.js/blob/acdda10d5896aa10abdf33e971951dbf7bd8f074/src/renderers/shaders/ShaderChunk/packing.glsl
use: <float4> pack(<float> v)
*/

#ifndef CONST_PACKING
#define CONST_PACKING
static const float PackUpscale = 256. / 255.; // fraction -> 0..1 (including 1)
static const float UnpackDownscale = 255. / 256.; // 0..1 -> fraction (excluding 1)
static const float3 PackFactors = float3( 256. * 256. * 256., 256. * 256.,  256. );
static const float4 UnpackFactors = UnpackDownscale / float4( PackFactors, 1. );
static const float ShiftRight8 = 1. / 256.;
#endif

#ifndef FNC_PACK
#define FNC_PACK

float4 pack( const in float v ) {
	float4 r = float4( frac( v * PackFactors ), v );
	r.yzw -= r.xyz * ShiftRight8; // tidy overflow
	return r * PackUpscale;
}

#endif