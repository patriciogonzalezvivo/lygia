/*
author: Patricio Gonzalez Vivo  
description: convert LST to RGB. LMS (long, medium, short), is a color space which represents the response of the three types of cones of the human eye, named for their responsivity (sensitivity) peaks at long, medium, and short wavelengths. https://en.wikipedia.org/wiki/LMS_color_space
use: <float3\float4> lms2rgb(<float3|float4> lms)
license: |
    Copyright (c) 2021 Patricio Gonzalez Vivo.
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

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