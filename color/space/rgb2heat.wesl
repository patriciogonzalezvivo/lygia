import lygia::color::space::rgb2hue::rgb2hue;

/*
contributors: Patricio Gonzalez Vivo
description: Converts a RGB rainbow pattern back to a single float value
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/


fn rgb2heat(c: vec3f) -> f32 { return 1.025 - rgb2hue(c) * 1.538461538; }
