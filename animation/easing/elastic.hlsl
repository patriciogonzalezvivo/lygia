#include "../../math/const.hlsl"

/*
author: Hugh Kennedy (https://github.com/hughsk)
description: elastic easing. From https://github.com/stackgl/glsl-easings
use: elastic<In|Out|InOut>(<float> x)
license: |
  This software is released under the MIT license:
  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef FNC_ELASTICIN
#define FNC_ELASTICIN
float elasticIn(in float t) {
    return sin(13.0 * t * HALF_PI) * pow(2.0, 10.0 * (t - 1.0));
}
#endif 

#ifndef FNC_ELASTICOUT
#define FNC_ELASTICOUT
float elasticOut(in float t) {
    return sin(-13.0 * (t + 1.0) * HALF_PI) * pow(2.0, -10.0 * t) + 1.0;
}
#endif

#ifndef FNC_ELASTICINOUT
#define FNC_ELASTICINOUT
float elasticInOut(in float t) {
    return t < 0.5
        ? 0.5 * sin(+13.0 * HALF_PI * 2.0 * t) * pow(2.0, 10.0 * (2.0 * t - 1.0))
        : 0.5 * sin(-13.0 * HALF_PI * ((2.0 * t - 1.0) + 1.0)) * pow(2.0, -10.0 * (2.0 * t - 1.0)) + 1.0;
}
#endif
