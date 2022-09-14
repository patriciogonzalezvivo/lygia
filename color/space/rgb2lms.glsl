/*
original_author: Patricio Gonzalez Vivo  
description: convert rgb to LMS. LMS (long, medium, short), is a color space which represents the response of the three types of cones of the human eye, named for their responsivity (sensitivity) peaks at long, medium, and short wavelengths. https://en.wikipedia.org/wiki/LMS_color_space
use: <vec3|vec4> rgb2lms(<vec3|vec4> rgb)
*/

#ifndef FNC_RGB2LMS
#define FNC_RGB2LMS
vec3 rgb2lms(vec3 rgb) {
    return vec3(
        (17.8824 * rgb.r) + (43.5161 * rgb.g) + (4.11935 * rgb.b),
        (3.45565 * rgb.r) + (27.1554 * rgb.g) + (3.86714 * rgb.b),
        (0.0299566 * rgb.r) + (0.184309 * rgb.g) + (1.46709 * rgb.b)
    );
}
vec4 rgb2lms(vec4 rgb) { vec4(rgb.rgb, rgb.a); }
#endif