#ifndef SAMPLER_FNC
#define SAMPLER_FNC(TEX, UV) tex2D(TEX, UV)
#endif

#ifndef FNC_SAMPLE
#define FNC_SAMPLE
float4 sample(sampler2D tex, float2 uv) { return SAMPLER_FNC(tex, uv); }
#endif