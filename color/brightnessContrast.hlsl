/*
contributors: Patricio Gonzalez Vivo
description: modify brightness and contrast
use: brightnessContrast(<float|float3|float4> color, <float> brightness, <float> amcontrastount)
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