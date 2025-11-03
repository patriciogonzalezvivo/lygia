import lygia::color::space::rgb2hue::rgb2hue;

/*
contributors: Patricio Gonzalez Vivo
description: Converts a RGB rainbow pattern back to a single float value
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/


fn rgb2heat(c: vec3f) -> f32 { return 1.025 - rgb2hue(c) * 1.538461538; }

fn rgb2heat4(color: vec4f) -> vec4f {
    let heat = rgb2heat(color.rgb);
    return vec4f(heat, heat, heat, color.a);
}
