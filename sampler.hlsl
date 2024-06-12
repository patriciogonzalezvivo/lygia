/*
contributors: Patricio Gonzalez Vivo
description: It defines the default sampler type and function for the shader.
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/
#ifndef SAMPLER_FNC
#define SAMPLER_FNC(TEX, UV) tex2D(TEX, UV)
#endif

#ifndef SAMPLER_TYPE
#define SAMPLER_TYPE sampler2D
#endif