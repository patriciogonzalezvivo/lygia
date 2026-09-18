/*
contributors: Patricio Gonzalez Vivo
description: Unpack a 3D vector into a float. Default base is 256.0
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

// https://github.com/mrdoob/three.js/blob/acdda10d5896aa10abdf33e971951dbf7bd8f074/src/renderers/shaders/ShaderChunk/packing.glsl

fn unpack8(vec3 v) -> f32 {
    let f = vec3( 8.0, 8.0 * 8.0, 8.0 * 8.0 * 8.0 );
    return dot(v, f) / 512.0;
}

fn unpack16(vec3 v) -> f32 {
    let f = vec3( 16.0, 16.0 * 16.0, 16.0 * 16.0 * 16.0 );
    return dot(v, f) / 4096.0;
}

fn unpack32(vec3 v) -> f32 {
    let f = vec3( 32.0, 32.0 * 32.0, 32.0 * 32.0 * 32.0 );
    return dot(v, f) / 32768.0;
}

fn unpack64(vec3 v) -> f32 {
    let f = vec3( 64.0, 64.0 * 64.0, 64.0 * 64.0 * 64.0 );
    return dot(v, f) / 262144.0;
}

fn unpack128(vec3 v) -> f32 {
    let f = vec3( 128.0, 128.0 * 128.0, 128.0 * 128.0 * 128.0 );
    return dot(v, f) / 2097152.0;
}

fn unpack256(vec3 v) -> f32  {
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
