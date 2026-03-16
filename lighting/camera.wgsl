/*
contributors: Patricio Gonzalez Vivo
description: Generic Camera Structure
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

struct Camera {
    var pos: vec3f;
    var dir: vec3f;

    var up: vec3f;
    var side: vec3f;

    var invhalffov: f32;
    var maxdist: f32;
};
