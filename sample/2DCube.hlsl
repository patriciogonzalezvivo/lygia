#include "../sampler.hlsl"
#include "../math/mod.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: Use a 2D texture as a 3D one
use: <float4> sample2DCube(in <SAMPLER_TYPE> lut, in <float3> xyz)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - SAMPLE2DCUBE_CELL_SIZE
    - SAMPLE2DCUBE_CELLS_PER_SIDE
    - SAMPLE2DCUBE_FNC
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef SAMPLE2DCUBE_CELLS_PER_SIDE
#define SAMPLE2DCUBE_CELLS_PER_SIDE 8.0
#endif

#ifndef SAMPLE2DCUBE_FNC
#define SAMPLE2DCUBE_FNC(TEX, UV) SAMPLER_FNC(TEX, saturate(UV))
#endif

#ifndef FNC_SAMPLE2DCUBE
#define FNC_SAMPLE2DCUBE
float4 sample2DCube(in SAMPLER_TYPE lut, in float3 xyz) {

#if defined(SAMPLE2DCUBE_CELL_SIZE)
    const float cellsSize = SAMPLE2DCUBE_CELL_SIZE;
    float cellsPerSide = sqrt(cellsSize);
    float cellsFactor = 1.0/cellsPerSide;
    float lutSize = cellsPerSide * cellsSize;
    float lutSizeFactor = 1.0/lutSize;

#elif defined(SAMPLE2DCUBE_CELLS_PER_SIDE)
    const float cellsPerSide = SAMPLE2DCUBE_CELLS_PER_SIDE;
    const float cellsSize = cellsPerSide * cellsPerSide;
    const float cellsFactor = 1.0/cellsPerSide;
    const float lutSize = cellsPerSide * cellsSize;
    const float lutSizeFactor = 1.0/lutSize;
#endif

    xyz *= (cellsSize-1.0);
    float iz = floor(xyz.z);

    float x0 = mod(iz, cellsPerSide) * cellsSize;
    float y0 = floor(iz * cellsFactor) * cellsSize;

    float x1 = mod(iz + 1.0, cellsPerSide) * cellsSize;
    float y1 = floor((iz + 1.0) * cellsFactor) * cellsSize;

    float2 uv0 = float2(x0 + xyz.x + 0.5, y0 + xyz.y + 0.5) * lutSizeFactor;
    float2 uv1 = float2(x1 + xyz.x + 0.5, y1 + xyz.y + 0.5) * lutSizeFactor;

    #ifndef SAMPLE2DCUBE_FLIP_Y
    uv0.y = 1.0 - uv0.y;
    uv1.y = 1.0 - uv1.y;
    #endif

    return lerp(SAMPLE2DCUBE_FNC(lut, uv0), 
                SAMPLE2DCUBE_FNC(lut, uv1), 
                xyz.z - iz );
}
#endif 