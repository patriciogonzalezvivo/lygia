import lygia::color::space::{
    rgb2lab::rgb2lab, 
    rgb2YCbCr::rgb2YCbCr, 
    rgb2YPbPr::rgb2YPbPr, 
    rgb2yuv::rgb2yuv, 
    rgb2oklab::rgb2oklab};

/*
contributors: Patricio Gonzalez Vivo
description: Perceptual distance between two color according to CIE94 https://en.wikipedia.org/wiki/Color_difference#CIE94
use: colorDistance(<vec3|vec4> rgbA, <vec3|vec4> rgbA)
options:
    - COLORDISTANCE_FNC: |
        colorDistanceLABCIE94, colorDistanceLAB, colorDistanceYCbCr,
        colorDistanceYPbPr, colorDistanceYUV, colorDistanceOKLAB
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn colorDistanceLABCIE94(rgb1 : vec3<f32>, rgb2 : vec3<f32>) -> f32 {
    let lab1 = rgb2lab(rgb1);
    let lab2 = rgb2lab(rgb2);

    let delta = lab1 - lab2;
    let c1 = sqrt(lab1.y * lab1.y + lab1.z * lab1.z);
    let c2 = sqrt(lab2.y * lab2.y + lab2.z * lab2.z);
    let delta_c = c1 - c2;
    var delta_h = delta.x * delta.x + delta.z * delta.z - delta_c * delta_c;
    delta_h = mix(0., sqrt(delta_h), step(0., delta_h));

    let sc = 1. + .045 * c1;
    let sh = 1. + .015 * c1;

    let delta_ckcsc = delta_c / sc;
    let delta_hkhsh = delta_h / sh;

    let delta_e = delta.x * delta.x + delta_ckcsc * delta_ckcsc + delta_hkhsh * delta_hkhsh;
    return mix(0., sqrt(delta_e), step(0., delta_e));
}

fn colorDistanceLAB(rgb1 : vec3<f32>, rgb2 : vec3<f32>) -> f32 {
    return distance(rgb2lab(rgb1), rgb2lab(rgb2));
}
fn colorDistanceYCbCr(rgb1 : vec3<f32>, rgb2 : vec3<f32>) -> f32 {
    return distance(rgb2YCbCr(rgb1).yz, rgb2YCbCr(rgb2).yz);
}
fn colorDistanceYPbPr(rgb1 : vec3<f32>, rgb2 : vec3<f32>) -> f32 {
    return distance(rgb2YPbPr(rgb1).yz, rgb2YPbPr(rgb2).yz);
}
fn colorDistanceYUV(rgb1 : vec3<f32>, rgb2 : vec3<f32>) -> f32 {
    return distance(rgb2yuv(rgb1), rgb2yuv(rgb2));
}
fn colorDistanceOKLAB(rgb1 : vec3<f32>, rgb2 : vec3<f32>) -> f32 {
    return distance(rgb2oklab(rgb1), rgb2oklab(rgb2));
}
