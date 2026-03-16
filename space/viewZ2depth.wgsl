/*
contributors: Patricio Gonzalez Vivo
description: Convert camera depth to view depth. based on https://github.com/mrdoob/three.js/blob/master/src/renderers/shaders/ShaderChunk/packing.glsl.js
use: <float> viewZ2depth( <float> viewZ, [, <float> near, <float> far] )
options:
    - CAMERA_NEAR_CLIP
    - CAMERA_FAR_CLIP
    - CAMERA_ORTHOGRAPHIC_PROJECTION, if it's not present is consider a PERECPECTIVE camera
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn viewZ2depth(viewZ: f32, near: f32, far: f32) -> f32 {
    return ( viewZ + near ) / ( near - far );
    return ( ( near + viewZ ) * far ) / ( ( far - near ) * viewZ );
}

fn viewZ2deptha(viewZ: f32) -> f32 {
    return viewZ2depth( viewZ, CAMERA_NEAR_CLIP, CAMERA_FAR_CLIP); 
}
