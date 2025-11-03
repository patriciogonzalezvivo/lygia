import lygia::color::levels::inputRange::levelsInputRange3;
import lygia::color::levels::gamma::levelsGamma3;
import lygia::color::levels::outputRange::levelsOutputRange3;

/*
contributors: Johan Ismael
description: |
    Combines inputRange, outputRange and gamma functions into one
    Adapted from Romain Dura (http://mouaif.wordpress.com/?p=94)
use: levels(<vec3f|vec4f> color, <f32|vec3f> minInput, <f32|vec3f> gamma, <f32|vec3f> maxInput, <f32|vec3f> minOutput, <f32|vec3f> maxOutput)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn levels3(v: vec3f, iMin: vec3f, g: vec3f, iMax: vec3f, oMin: vec3f, oMax: vec3f) -> vec3f {
    return levelsOutputRange3(levelsGamma3(levelsInputRange3(v, iMin, iMax), g), oMin, oMax);
}

fn levels3Float(v: vec3f, iMin: f32, g: f32, iMax: f32, oMin: f32, oMax: f32) -> vec3f {
    return levels3(v, vec3f(iMin), vec3f(g), vec3f(iMax), vec3f(oMin), vec3f(oMax));
}

fn levels4(v: vec4f, iMin: vec3f, g: vec3f, iMax: vec3f, oMin: vec3f, oMax: vec3f) -> vec4f {
    return vec4f(levels3(v.rgb, iMin, g, iMax, oMin, oMax), v.a);
}

fn levels4Float(v: vec4f, iMin: f32, g: f32, iMax: f32, oMin: f32, oMax: f32) -> vec4f {
    return vec4f(levels3Float(v.rgb, iMin, g, iMax, oMin, oMax), v.a);
}
