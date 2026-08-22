#include "../sampler.wgsl"
#include "../math/saturate.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: Use a 2D texture as a 3D one
use: <vec4> sample2DCube(in <SAMPLER_TYPE> lut, in <vec3> xyz)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - SAMPLE2DCUBE_CELL_SIZE
    - SAMPLE2DCUBE_CELLS_PER_SIDE: defaults to 8
    - SAMPLE2DCUBE_FNC
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

// #define SAMPLE2DCUBE_FNC(TEX, UV) SAMPLER_FNC(TEX, saturate(UV))

fn sample2DCube(lut: SAMPLER_TYPE, xyz: vec3f) -> vec4f {
    const SAMPLE2DCUBE_CELLS_PER_SIDE: f32 = 8.0;

    let cellsSize = SAMPLE2DCUBE_CELL_SIZE;
    let cellsPerSide = sqrt(cellsSize);
    let cellsFactor = 1.0/cellsPerSide;
    let lutSize = cellsPerSide * cellsSize;
    let lutSizeFactor = 1.0/lutSize;

    let cellsPerSide = SAMPLE2DCUBE_CELLS_PER_SIDE;
    let cellsSize = cellsPerSide * cellsPerSide;
    let cellsFactor = 1.0/cellsPerSide;
    let lutSize = cellsPerSide * cellsSize;
    let lutSizeFactor = 1.0/lutSize;

    xyz.z = 1.0 - xyz.z;

    xyz *= (cellsSize-1.0);
    let iz = floor(xyz.z);

    let x0 = mod(iz, cellsPerSide) * cellsSize;
    let y0 = floor(iz * cellsFactor) * cellsSize;

    let x1 = mod(iz + 1.0, cellsPerSide) * cellsSize;
    let y1 = floor((iz + 1.0) * cellsFactor) * cellsSize;

    let uv0 = vec2f(x0 + xyz.x + 0.5, y0 + xyz.y + 0.5) * lutSizeFactor;
    let uv1 = vec2f(x1 + xyz.x + 0.5, y1 + xyz.y + 0.5) * lutSizeFactor;

    uv0.y = 1.0 - uv0.y;
    uv1.y = 1.0 - uv1.y;

    return mix( SAMPLE2DCUBE_FNC(lut, uv0), 
                SAMPLE2DCUBE_FNC(lut, uv1), 
                xyz.z - iz);
}
