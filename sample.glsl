#ifndef SAMPLER_FNC
#if __VERSION__ >= 300
#define SAMPLER_FNC(TEX, UV) texture(TEX, UV)
#else
#define SAMPLER_FNC(TEX, UV) texture2D(TEX, UV)
#endif
#endif

#ifndef FNC_SAMPLE
#define FNC_SAMPLE
vec4 sample(sampler2D tex, vec2 uv) { return SAMPLER_FNC(tex, uv); }
#endif