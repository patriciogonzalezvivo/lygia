/*
contributors: Patricio Gonzalez Vivo
description: clamp a value between 0 and the medium precision max (65504.0) for floating points
use: <float|vec2|vec3|vec4> saturateMediump(<float|vec2|vec3|vec4> value)
*/

#ifndef FNC_SATURATEMEDIUMP
#define FNC_SATURATEMEDIUMP

#ifndef MEDIUMP_FLT_MAX
#define MEDIUMP_FLT_MAX    65504.0
#endif

#if defined(TARGET_MOBILE) || defined(PLATFORM_WEBGL) || defined(PLATFORM_RPI)
#define saturateMediump(V) min(V, MEDIUMP_FLT_MAX)
#else
#define saturateMediump(V) V
#endif

#endif