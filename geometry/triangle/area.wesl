
import lygia::geometry::triangle::triangle::Triangle;

/*
contributors: Patricio Gonzalez Vivo
description: Returns the area of a triangle
use: <f32> area(<Triangle> tri)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/


fn area(tri: Triangle) -> f32 {
    return length(cross(tri.b - tri.a, tri.c - tri.a)) * 0.5;
}
