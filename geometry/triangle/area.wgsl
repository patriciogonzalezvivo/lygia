#include "triangle.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: Returns the area of a triangle
use: <float> area(<Triangle> tri)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn area(_tri: Triangle) -> f32 { return length( cross( _tri.b - _tri.a, _tri.c - _tri.a) ) * 0.5; }
