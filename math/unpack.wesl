/*
contributors: Patricio Gonzalez Vivo
description: Unpack a 3D vector into a float. Default base is 256.0
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

// https://github.com/mrdoob/three.js/blob/acdda10d5896aa10abdf33e971951dbf7bd8f074/src/renderers/shaders/ShaderChunk/packing.glsl

const PackUpscale: f32 = 256. / 255.; // fraction -> 0..1 (including 1)
const UnpackDownscale: f32 = 255. / 256.; // 0..1 -> fraction (excluding 1)
const PackFactors = vec3f( 256. * 256. * 256., 256. * 256.,  256. );
const UnpackFactors = UnpackDownscale / vec4f( PackFactors, 1. );
const ShiftRight8: f32 = 1. / 256.;

fn unpack8(v: vec3) -> f32 {
    let f = vec3( 8.0, 8.0 * 8.0, 8.0 * 8.0 * 8.0 );
    return dot(v, f) / 512.0;
}

fn unpack16(v: vec3) -> f32 {
    let f = vec3( 16.0, 16.0 * 16.0, 16.0 * 16.0 * 16.0 );
    return dot(v, f) / 4096.0;
}

fn unpack32(v: vec3) -> f32 {
    let f = vec3( 32.0, 32.0 * 32.0, 32.0 * 32.0 * 32.0 );
    return dot(v, f) / 32768.0;
}

fn unpack64(v: vec3) -> f32 {
    let f = vec3( 64.0, 64.0 * 64.0, 64.0 * 64.0 * 64.0 );
    return dot(v, f) / 262144.0;
}

fn unpack128(v: vec3) -> f32 {
    let f = vec3( 128.0, 128.0 * 128.0, 128.0 * 128.0 * 128.0 );
    return dot(v, f) / 2097152.0;
}

fn unpack256(v: vec3) -> f32  {
    let f = vec3( 256.0, 256.0 * 256.0, 256.0 * 256.0 * 256.0 );
    return dot(v, f) / 16581375.0;
}

fn unpack(v: vec3f, base: f32) -> f32 {
    let base3 = base * base * base;
    let f = vec3( base, base * base, base3);
    return dot(v, f) / base3;
}

fn unpack(v: vec3f) -> f32 { return unpack256(v); }

// https://github.com/mrdoob/three.js/blob/acdda10d5896aa10abdf33e971951dbf7bd8f074/src/renderers/shaders/ShaderChunk/packing.glsl
fn unpack4(v: vec4f ) -> f32 { return dot( v, UnpackFactors ); }
