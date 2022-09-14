/*
original_author: Patricio Gonzalez Vivo
description: modify brightness and contrast
use: brightnessContrast(<float|vec3|vec4> color, <float> brightness, <float> amcontrastount)
*/

#ifndef FNC_BRIGHTNESSCONTRAST
#define FNC_BRIGHTNESSCONTRAST
float brightnessContrast( float value, float brightness, float contrast ) {
    return ( value - 0.5 ) * contrast + 0.5 + brightness;
}

vec3 brightnessContrast( vec3 color, float brightness, float contrast ) {
    return ( color - 0.5 ) * contrast + 0.5 + brightness;
}

vec4 brightnessContrast( vec4 color, float brightness, float contrast ) {
    return vec4(brightnessContrast(color.rgb, brightness, contrast), color.a);
}
#endif