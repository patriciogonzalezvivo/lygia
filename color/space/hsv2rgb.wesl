import package::color::space::hue2rgb::hue2rgb;

/*
contributors: Inigo Quiles
description: 'Convert from HSV to linear RGB'

*/

fn hsv2rgb(hsv : vec3f) -> vec3f {
    return ((hue2rgb(hsv.x) - 1.0) * hsv.y + 1.0) * hsv.z;
}
