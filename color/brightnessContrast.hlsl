/*
contributors: Patricio Gonzalez Vivo
description: Modify brightness and contrast
use: brightnessContrast(<float|float3|float4> color, <float> brightness, <float> amcontrastount)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/color_brightnessContrast.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_BRIGHTNESSCONTRAST
#define FNC_BRIGHTNESSCONTRAST
float brightnessContrast( float value, float brightness, float contrast ) {
    return ( value - 0.5 ) * contrast + 0.5 + brightness;
}

float3 brightnessContrast( float3 color, float brightness, float contrast ) {
    return ( color - 0.5 ) * contrast + 0.5 + brightness;
}

float4 brightnessContrast( float4 color, float brightness, float contrast ) {
    return float4(brightnessContrast(color.rgb, brightness, contrast), color.a);
}
#endif