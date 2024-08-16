/*
contributors: Shadi El Hajj
description: Volume Material Structure
*/

#ifndef STR_VOLUME_MATERIAL
#define STR_VOLUME_MATERIAL

struct VolumeMaterial {
    vec3    color;
    float   density;
    float   sdf;
};

#endif
