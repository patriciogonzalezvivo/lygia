/*
contributors: Patricio Gonzalez Vivo
description: Unpack a 3D vector into a float. Default base is 256.0
use: <float> unpack(<vec3> v, [, <float> base])
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef UNPACK_FNC
#define UNPACK_FNC unpack256
#endif 

// https://github.com/mrdoob/three.js/blob/acdda10d5896aa10abdf33e971951dbf7bd8f074/src/renderers/shaders/ShaderChunk/packing.glsl
#ifndef CONST_PACKING
#define CONST_PACKING
const float PackUpscale = 256. / 255.; // fraction -> 0..1 (including 1)
const float UnpackDownscale = 255. / 256.; // 0..1 -> fraction (excluding 1)
const vec3 PackFactors = vec3( 256. * 256. * 256., 256. * 256.,  256. );
const vec4 UnpackFactors = UnpackDownscale / vec4( PackFactors, 1. );
const float ShiftRight8 = 1. / 256.;
#endif

#ifndef FNC_UNPACK
#define FNC_UNPACK

float unpack8(vec3 v) {
    vec3 f = vec3( 8.0, 8.0 * 8.0, 8.0 * 8.0 * 8.0 );
    return dot(v, f) / 512.0;
}

float unpack16(vec3 v) {
    vec3 f = vec3( 16.0, 16.0 * 16.0, 16.0 * 16.0 * 16.0 );
    return dot(v, f) / 4096.0;
}

float unpack32(vec3 v) {
    vec3 f = vec3( 32.0, 32.0 * 32.0, 32.0 * 32.0 * 32.0 );
    return dot(v, f) / 32768.0;
}

float unpack64(vec3 v) {
    vec3 f = vec3( 64.0, 64.0 * 64.0, 64.0 * 64.0 * 64.0 );
    return dot(v, f) / 262144.0;
}

float unpack128(vec3 v) {
    vec3 f = vec3( 128.0, 128.0 * 128.0, 128.0 * 128.0 * 128.0 );
    return dot(v, f) / 2097152.0;
}

float unpack256(vec3 v) {
    vec3 f = vec3( 256.0, 256.0 * 256.0, 256.0 * 256.0 * 256.0 );
    return dot(v, f) / 16581375.0;
}

float unpack(vec3 v, float base) {
    float base3 = base * base * base;
    vec3 f = vec3( base, base * base, base3);
    return dot(v, f) / base3;
}

float unpack(vec3 v) {
    return UNPACK_FNC(v);
}

// https://github.com/mrdoob/three.js/blob/acdda10d5896aa10abdf33e971951dbf7bd8f074/src/renderers/shaders/ShaderChunk/packing.glsl
float unpack( const in vec4 v ) {
	return dot( v, UnpackFactors );
}

#endif