/*
contributors: Shadi El Hajj
description: Light attenuation equation
license: MIT License (MIT) Copyright (c) 2024 Shadi EL Hajj
*/

#ifndef LIGHT_ATTENUATION_CONSTANT
#define LIGHT_ATTENUATION_CONSTANT 0.0
#endif

#ifndef LIGHT_ATTENUATION_LINEAR
#define LIGHT_ATTENUATION_LINEAR 0.0
#endif

#ifndef LIGHT_ATTENUATION_EXPONENTIAL
#define LIGHT_ATTENUATION_EXPONENTIAL 1.0
#endif

#ifndef LIGHT_ATTENUATION_EXPONENT
#define LIGHT_ATTENUATION_EXPONENT 2.0
#endif

#ifndef FNC_LIGHT_ATTENUATION
#define FNC_LIGHT_ATTENUATION

float attenuation(float dist) {
    return 1.0 / (
        LIGHT_ATTENUATION_CONSTANT + 
        LIGHT_ATTENUATION_LINEAR * dist +
        LIGHT_ATTENUATION_EXPONENTIAL * pow(dist, LIGHT_ATTENUATION_EXPONENT)
        );
}

#endif
