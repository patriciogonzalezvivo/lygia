/*
contributors: Patricio Gonzalez Vivo
description: Modify brightness and contrast
use: brightnessContrast(<float|vec3|vec4> color, <float> brightness, <float> amcontrastount)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/color_brightnessContrast.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn brightnessContrast(v : f32, b : f32, c : f32) -> f32 { return (v - 0.5) * c + 0.5 + b; }

fn brightnessContrast3(v : vec3f, b : f32, c : f32) -> vec3f { return (v - 0.5) * c + 0.5 + b; }

fn brightnessContrast4(v : vec4f, b : f32, c : f32) -> vec4f { return vec4((v.rgb - 0.5) * c + 0.5 + b, v.a); }