// #ifndef LIGHT_POSITION
// #if defined(GLSLVIEWER)
// #define LIGHT_POSITION u_light
// #else
// #define LIGHT_POSITION vec3(0.0, 10.0, -50.0)
// #endif
// #endif

#ifndef LIGHT_COLOR
#if defined(GLSLVIEWER)
#define LIGHT_COLOR u_lightColor
#else
#define LIGHT_COLOR vec3(0.5)
#endif
#endif

#ifndef RAYMARCH_AMBIENT
#define RAYMARCH_AMBIENT vec3(1.0)
#endif

#ifndef RAYMARCH_BACKGROUND
#define RAYMARCH_BACKGROUND vec3(0.0)
#endif

#ifndef RAYMARCH_SAMPLES
#define RAYMARCH_SAMPLES 64
#endif

#ifndef FNC_RAYMARCHVOLUMERENDER
#define FNC_RAYMARCHVOLUMERENDER

vec4 raymarchVolume( in vec3 ro, in vec3 rd ) {

    const float tmin        = 1.0;
    const float tmax        = 10.0;
    const float fSamples    = float(RAYMARCH_SAMPLES);
    const float tstep       = tmax/fSamples;
    const float absorption  = 100.;

    #ifdef LIGHT_POSITION
    const int   nbSampleLight   = 6;
    const float fSampleLight    = float(nbSampleLight);
    const float tstepl          = tmax/fSampleLight;
    vec3 sun_direction          = normalize( LIGHT_POSITION );
    #endif

    float T = 1.;
    float t = tmin;
    vec3 col = vec3(0.0);
    vec3 p = ro;
    for(int i = 0; i < RAYMARCH_SAMPLES; i++) {
        vec4 res    = raymarchMap(p);
        float density = (0.1 - res.a);
        if (density > 0.0) {
            float tmp = density / fSamples;
            T *= 1.0 - tmp * absorption;
            if( T <= 0.001)
                break;

            col += res.rgb * fSamples * tmp * T;
                
            // //Light scattering
            #ifdef LIGHT_POSITION
            float Tl = 1.0;
            for(int j = 0; j<nbSampleLight; j++) {
                float densityLight = raymarchMap( p + sun_direction * float(j) * tstepl ).a;
                if(densityLight>0.)
                    Tl *= 1. - densityLight * absorption/fSamples;
                if (Tl <= 0.01)
                    break;
            }
            col += LIGHT_COLOR * 80. * tmp * T * Tl;
            #endif
        }
        p += rd * tstep;
    }

    return vec4(saturate(col), t);
}

#endif