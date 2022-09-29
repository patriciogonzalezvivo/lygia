
#ifndef SAMPLER_FNC
#define SAMPLER_FNC(TEX, UV) tex2D(TEX, UV)
#endif

#ifndef SAMPLE_2DCUBE_CELL_SIZE
#define SAMPLE_2DCUBE_CELL_SIZE 64.0
#endif

#ifndef SAMPLE_2DCUBE_CELLS_PER_SIDE
#define SAMPLE_2DCUBE_CELLS_PER_SIDE 8.0
#endif

#ifndef SAMPLE_2DCUBE_FNC
#define SAMPLE_2DCUBE_FNC

float4 sample2DCube(in sampler2D tex_lut, in float3 xyz) {
    float Z = xyz.z * SAMPLE_2DCUBE_CELL_SIZE;

    const float cells_factor = 1.0/SAMPLE_2DCUBE_CELLS_PER_SIDE;
    const float pixel = 1.0/ (SAMPLE_2DCUBE_CELLS_PER_SIDE * SAMPLE_2DCUBE_CELL_SIZE);
    const float halt_pixel = pixel * 0.5;

    float2 cellA = float2(0.0, 0.0);
    cellA.y = floor(floor(Z) / SAMPLE_2DCUBE_CELLS_PER_SIDE);
    cellA.x = floor(Z) - (cellA.y * SAMPLE_2DCUBE_CELLS_PER_SIDE);
    
    float2 cellB = float2(0.0, 0.0);
    cellB.y = floor(ceil(Z) / SAMPLE_2DCUBE_CELLS_PER_SIDE);
    cellB.x = ceil(Z) - (cellB.y * SAMPLE_2DCUBE_CELLS_PER_SIDE);
    
    float2 uvA = (cellA * cells_factor) + halt_pixel + ((cells_factor - pixel) * xyz.xy);
    float2 uvB = (cellB * cells_factor) + halt_pixel + ((cells_factor - pixel) * xyz.xy);

    float4 b0 = SAMPLER_FNC(tex_lut, saturate(uvA));
    float4 b1 = SAMPLER_FNC(tex_lut, saturate(uvB));

    return lerp(b0, b1, fract(Z));
}

#endif 