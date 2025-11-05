import lygia::math::inverse::inverse4;

@if(CAMERA_PROJECTION_MATRIX)
import constants::CAMERA_PROJECTION_MATRIX;

@if(INVERSE_CAMERA_PROJECTION_MATRIX)
import constants::INVERSE_CAMERA_PROJECTION_MATRIX;

/*
contributors: Patricio Gonzalez Vivo
description: |
    Derive view surface position from screen coordinates and depth

    WESL conditionals:
    - @if(CAMERA_PROJECTION_MATRIX && INVERSE_CAMERA_PROJECTION_MATRIX): Uses provided inverse matrix
    - @if(CAMERA_PROJECTION_MATRIX && !INVERSE_CAMERA_PROJECTION_MATRIX): Computes inverse matrix on the fly
    - Note: CAMERA_PROJECTION_MATRIX is required; function not defined if absent
use: screen2viewPosition(screenPosition: vec2f, depth: f32, viewZ: f32) -> vec4f
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

@if(CAMERA_PROJECTION_MATRIX && INVERSE_CAMERA_PROJECTION_MATRIX)
fn screen2viewPosition(screenPosition: vec2f, depth: f32, viewZ: f32) -> vec4f {
    let clipW = constants::CAMERA_PROJECTION_MATRIX[2][3] * viewZ + constants::CAMERA_PROJECTION_MATRIX[3][3];
    let clipPosition = vec4f((vec3f(screenPosition, depth) - 0.5) * 2.0, 1.0) * clipW;
    return constants::INVERSE_CAMERA_PROJECTION_MATRIX * clipPosition;
}

@if(CAMERA_PROJECTION_MATRIX && !INVERSE_CAMERA_PROJECTION_MATRIX)
fn screen2viewPosition(screenPosition: vec2f, depth: f32, viewZ: f32) -> vec4f {
    let INVERSE_CAMERA_PROJECTION_MATRIX = inverse4(constants::CAMERA_PROJECTION_MATRIX);
    let clipW = constants::CAMERA_PROJECTION_MATRIX[2][3] * viewZ + constants::CAMERA_PROJECTION_MATRIX[3][3];
    let clipPosition = vec4f((vec3f(screenPosition, depth) - 0.5) * 2.0, 1.0) * clipW;
    return INVERSE_CAMERA_PROJECTION_MATRIX * clipPosition;
}
