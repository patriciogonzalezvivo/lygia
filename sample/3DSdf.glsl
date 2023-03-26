#include "2DCube.glsl"
#include "../space/scale.glsl"
#include "../sdf/boxSDF.glsl"
#include "../sdf/opIntersection.glsl"

#ifndef SAMPLE3DSDF_TYPE
#define SAMPLE3DSDF_TYPE float
#endif

#ifndef SAMPLE3DSDF_COMPONENT
#define SAMPLE3DSDF_COMPONENT r
#endif

#ifndef FNC_SAMPLE3DSDF
#define FNC_SAMPLE3DSDF
SAMPLE3DSDF_TYPE sample3DSdf(sampler2D tex, vec3 pos) {
    pos += .5;
    pos = scale(pos, .5);
    SAMPLE3DSDF_TYPE sdf = (sample2DCube(tex, pos).SAMPLE3DSDF_COMPONENT * 2.0 - 1.0) * 1.5;
    return opIntersection( boxSDF(pos - 0.5, vec3(.5)), sdf);
}
#endif