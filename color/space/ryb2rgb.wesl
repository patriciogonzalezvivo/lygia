import package::math::cubicMix::cubicMix3;

/*
contributors: Patricio Gonzalez Vivo
description: Convert from RYB to RGB color space. Based on http://nishitalab.org/user/UEI/publication/Sugita_IWAIT2015.pdf http://vis.computer.org/vis2004/DVD/infovis/papers/gossett.pdf
use: <vec3> ryb2rgb(<vec3> ryb)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/color_ryb.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn ryb2rgb(ryb: vec3f) -> vec3f {
    let ryb000 = vec3f(1., 1., 1.);       // White
    let ryb001 = vec3f(.163, .373, .6);   // Blue
    let ryb010 = vec3f(1., 1., 0.);       // Yellow
    let ryb100 = vec3f(1., 0., 0.);       // Red
    let ryb011 = vec3f(0., .66, .2);      // Green
    let ryb101 = vec3f(.5, 0., .5);       // Violet
    let ryb110 = vec3f(1., .5, 0.);       // Orange
    let ryb111 = vec3f(0., 0., 0.);       // Black
    return 
        cubicMix3(
            cubicMix3(
                cubicMix3(ryb000, ryb001, vec3f(ryb.z)),
                cubicMix3(ryb010, ryb011, vec3f(ryb.z)),
                vec3f(ryb.y)
            ), 
            cubicMix3(
                cubicMix3(ryb100, ryb101, ryb.z),
                cubicMix3(ryb110, ryb111, ryb.z),
                vec3f(ryb.y)
            ), 
            vec3f(ryb.x)
        );
}
