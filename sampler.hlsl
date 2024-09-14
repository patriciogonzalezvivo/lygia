/*
contributors: Patricio Gonzalez Vivo
description: It defines the default sampler type and function for the shader.
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#if defined(__SHADER_TARGET_MAJOR) && __SHADER_TARGET_MAJOR < 4

    #ifndef SAMPLER_FNC
    #define SAMPLER_FNC(TEX, UV) tex2D(TEX, UV)
    #endif

    #ifndef SAMPLER_TYPE
    #define SAMPLER_TYPE sampler2D
    #endif

#else

    // https://docs.unity3d.com/Manual/SL-SamplerStates.html
    #ifndef SAMPLER_BILINEAR_CLAMP
    #define SAMPLER_BILINEAR_CLAMP defaultLinearClampSampler
    SamplerState SAMPLER_BILINEAR_CLAMP
    {
        Filter = MIN_MAG_LINEAR_MIP_POINT;
        AddressU = Clamp;
        AddressV = Clamp;
    };
    #endif

    #ifndef SAMPLER_TRILINEAR_CLAMP
    #define SAMPLER_TRILINEAR_CLAMP defaultTrilinearClampSampler
    SamplerState SAMPLER_TRILINEAR_CLAMP
    {
        Filter = MIN_MAG_MIP_LINEAR;
        AddressU = Clamp;
        AddressV = Clamp;
    };
    #endif

    #ifndef SAMPLER_FNC
    #define SAMPLER_FNC(TEX, UV) TEX.Sample(SAMPLER_BILINEAR_CLAMP, UV)
    #endif

    #ifndef SAMPLER_TYPE
    #define SAMPLER_TYPE Texture2D
    #endif

#endif