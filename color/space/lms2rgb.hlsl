/*
original_author: Patricio Gonzalez Vivo  
description: convert LST to RGB. LMS (long, medium, short), is a color space which represents the response of the three types of cones of the human eye, named for their responsivity (sensitivity) peaks at long, medium, and short wavelengths. https://en.wikipedia.org/wiki/LMS_color_space
use: <float3\float4> lms2rgb(<float3|float4> lms)
*/

#ifndef FNC_LMS2RGB
#define FNC_LMS2RGB
float3 lms2rgb(float3 lms) {
    return float3( 
        (0.0809444479 * lms.x) + (-0.130504409 * lms.y) + (0.116721066 * lms.z),
        (-0.0102485335 * lms.x) + (0.0540193266 * lms.y) + (-0.113614708 * lms.z),
        (-0.000365296938 * lms.x) + (-0.00412161469 * lms.y) + (0.693511405 * lms.z)
    );
}
float4 lms2rgb(float4 lms) { return float4( lms2rgb(lms.xyz), lms.a ); }
#endif