#include "../palette/spectral.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: Wavelength between 400~700 nm to RGB
use: <vec3> w2rgb(<float> wavelength)
options:
    W2RGB_FNC(X): spectral_zucconi, spectral_zucconi6, spectral_gems, spectral_geoffrey,
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_W2RGB
#define FNC_W2RGB

vec3 w2rgb(const in float w) {

#if defined( W2RGB_FNC )
    float x = saturate((w - 400.0) * 0.00333333333);
	#if defined(W2RGB_ITERATIONS)
    vec3 sum = vec3(0.0, 0.0, 0.0);
    for (float i = 0.; i < W2RGB_ITERATIONS ;i++) {
        sum +=  W2RGB_FNC(x * (1. - i * .003)) / W2RGB_ITERATIONS;
    }
    return sum;

	#else
    return W2RGB_FNC (x);

	#endif

#else 
    vec3 c = vec3(0.0, 0.0, 0.0);

	if ((w>=400.0)&&(w<410.0)) {        float t=(w-400.0)/(410.0-400.0); c.r=    +(0.33*t)-(0.20*t*t); }
	else if ((w>=410.0)&&(w<475.0)) {   float t=(w-410.0)/(475.0-410.0); c.r=0.14         -(0.13*t*t); }
	else if ((w>=545.0)&&(w<595.0)) {   float t=(w-545.0)/(595.0-545.0); c.r=    +(1.98*t)-(     t*t); }
	else if ((w>=595.0)&&(w<650.0)) {   float t=(w-595.0)/(650.0-595.0); c.r=0.98+(0.06*t)-(0.40*t*t); }
	else if ((w>=650.0)&&(w<700.0)) {   float t=(w-650.0)/(700.0-650.0); c.r=0.65-(0.84*t)+(0.20*t*t); }
		
    if ((w>=415.0)&&(w<475.0)) {        float t=(w-415.0)/(475.0-415.0); c.g=          +(0.80*t*t); }
	else if ((w>=475.0)&&(w<590.0)) {   float t=(w-475.0)/(590.0-475.0); c.g=0.8 +(0.76*t)-(0.80*t*t); }
	else if ((w>=585.0)&&(w<639.0)) {   float t=(w-585.0)/(639.0-585.0); c.g=0.82-(0.80*t)           ; }
		
    if ((w>=400.0)&&(w<475.0)) {        float t=(w-400.0)/(475.0-400.0); c.b=    +(2.20*t)-(1.50*t*t); }
	else if ((w>=475.0)&&(w<560.0)) {   float t=(w-475.0)/(560.0-475.0); c.b=0.7 -(     t)+(0.30*t*t); }

	return c;
#endif

}
#endif