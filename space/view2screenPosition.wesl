@if(CAMERA_PROJECTION_MATRIX)
import constants::CAMERA_PROJECTION_MATRIX;

/*
contributors: Patricio Gonzalez Vivo
description: |
    Get screen coordinates from view position

    WESL conditionals:
    - @if(CAMERA_PROJECTION_MATRIX): Function is only defined if CAMERA_PROJECTION_MATRIX constant is available
use: view2screenPosition(viewPosition: vec3f) -> vec2f
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

@if(CAMERA_PROJECTION_MATRIX)
fn view2screenPosition(viewPosition: vec3f) -> vec2f {
    let clip = constants::CAMERA_PROJECTION_MATRIX * vec4f(viewPosition, 1.0);
    return (clip.xy / clip.w + 1.0) * 0.5;
}
