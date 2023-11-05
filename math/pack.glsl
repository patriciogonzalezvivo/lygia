/*
contributors: Ricardo Caballero
description: |
    Pack a float into a 4D vector. From https://github.com/mrdoob/three.js/blob/acdda10d5896aa10abdf33e971951dbf7bd8f074/src/renderers/shaders/ShaderChunk/packing.glsl
use: <vec4> pack(<float> v)
*/

#ifndef CONST_PACKING
#define CONST_PACKING
const float PackUpscale = 256. / 255.; // fraction -> 0..1 (including 1)
const float UnpackDownscale = 255. / 256.; // 0..1 -> fraction (excluding 1)
const vec3 PackFactors = vec3( 256. * 256. * 256., 256. * 256.,  256. );
const vec4 UnpackFactors = UnpackDownscale / vec4( PackFactors, 1. );
const float ShiftRight8 = 1. / 256.;
#endif

#ifndef FNC_PACK
#define FNC_PACK

vec4 pack( const in float v ) {
	vec4 r = vec4( fract( v * PackFactors ), v );
	r.yzw -= r.xyz * ShiftRight8; // tidy overflow
	return r * PackUpscale;
}

#endif