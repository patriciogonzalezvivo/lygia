#ifndef FNC_PRE_FILTERED_IMPORTANCE_SAMPLING
#define FNC_PRE_FILTERED_IMPORTANCE_SAMPLING

float prefilteredImportanceSampling(float ipdf, float omegaP, int numSamples) {
    const float K = 4.0;
    float omegaS = ipdf / float(numSamples);
    float mipLevel = log2(K * omegaS / omegaP) * 0.5; // log4
    return mipLevel;
}

#endif