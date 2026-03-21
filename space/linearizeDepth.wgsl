/*
contributors: Patricio Gonzalez Vivo
description: linearize depth
use: linearizeDepth(<float> depth, <float> near, <float> far)
options:
    - CAMERA_NEAR_CLIP
    - CAMERA_FAR_CLIP
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn linearizeDepth(depth: f32, near: f32, far: f32) -> f32 {
    depth = 2.0 * depth - 1.0;
    return (2.0 * near * far) / (far + near - depth * (far - near));
}

fn linearizeDeptha(depth: f32) -> f32 {
  return linearizeDepth(depth, CAMERA_NEAR_CLIP, CAMERA_FAR_CLIP);
}
