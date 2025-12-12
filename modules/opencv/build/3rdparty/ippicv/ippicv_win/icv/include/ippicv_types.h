/* 
// Copyright 1999 Intel Corporation All Rights Reserved.
// 
// 
// This software and the related documents are Intel copyrighted materials, and your use of them is governed by
// the express license under which they were provided to you ('License'). Unless the License provides otherwise,
// you may not use, modify, copy, publish, distribute, disclose or transmit this software or the related
// documents without Intel's prior written permission.
// This software and the related documents are provided as is, with no express or implied warranties, other than
// those that are expressly stated in the License.
// 
*/

/* 
//              Intel(R) Integrated Performance Primitives (Intel(R) IPP)
//              Derivative Types and Macro Definitions
// 
//              The main purpose of this header file is 
//              to support compatibility with the legacy
//              domains until their end of life. 
// 
*/


#ifndef __IPPICV_TYPES_H__
#define __IPPICV_TYPES_H__

#ifdef __cplusplus
extern "C" {
#endif

#if defined( IPPTYPES_H__ )
#error "You try to include ippicv_types.h and ipptypes.h together. You can't use Intel(R) IPP and ICV libraries together, please use only one and please use header files only for one library"
#endif

/*****************************************************************************/
/*                   Below are ippCore domain specific definitions           */
/*****************************************************************************/
#if !defined( _OWN_BLDPCS )

#ifndef IPP_CPU_FEATURES__
#define IPP_CPU_FEATURES__

#define   ippCPUID_MMX             0x00000001   /* Intel(R) Architecture MMX technology supported                              */
#define   ippCPUID_SSE             0x00000002   /* Intel(R) Streaming SIMD Extensions                                          */
#define   ippCPUID_SSE2            0x00000004   /* Intel(R) Streaming SIMD Extensions 2                                        */
#define   ippCPUID_SSE3            0x00000008   /* Intel(R) Streaming SIMD Extensions 3                                        */
#define   ippCPUID_SSSE3           0x00000010   /* Supplemental Streaming SIMD Extensions 3                                    */
#define   ippCPUID_MOVBE           0x00000020   /* Intel(R) MOVBE instruction                                                  */
#define   ippCPUID_SSE41           0x00000040   /* Intel(R) Streaming SIMD Extensions 4.1                                      */
#define   ippCPUID_SSE42           0x00000080   /* Intel(R) Streaming SIMD Extensions 4.2                                      */
#define   ippCPUID_AVX             0x00000100   /* Intel(R) Advanced Vector Extensions instruction set                         */
#define   ippAVX_ENABLEDBYOS       0x00000200   /* Intel(R) Advanced Vector Extensions instruction set is supported by OS      */
#define   ippCPUID_AES             0x00000400   /* Intel(R) AES New Instructions                                               */
#define   ippCPUID_CLMUL           0x00000800   /* Intel(R) CLMUL instruction                                                  */
#define   ippCPUID_ABR             0x00001000   /* Reserved                                                                    */
#define   ippCPUID_RDRAND          0x00002000   /* Intel(R) RDRAND instruction                                                 */
#define   ippCPUID_F16C            0x00004000   /* Intel(R) F16C new instructions                                              */
#define   ippCPUID_AVX2            0x00008000   /* Intel(R) Advanced Vector Extensions 2 instruction set                       */
#define   ippCPUID_ADCOX           0x00010000   /* Intel(R) ADOX/ADCX new instructions                                         */
#define   ippCPUID_RDSEED          0x00020000   /* Intel(R) RDSEED instruction                                                 */
#define   ippCPUID_PREFETCHW       0x00040000   /* Intel(R) PREFETCHW instruction                                              */
#define   ippCPUID_SHA             0x00080000   /* Intel(R) SHA new instructions                                               */
#define   ippCPUID_AVX512F         0x00100000   /* Intel(R) Advanced Vector Extensions 3.1 instruction set                     */
#define   ippCPUID_AVX512CD        0x00200000   /* Intel(R) Advanced Vector Extensions CD (Conflict Detection) instruction set */
#define   ippCPUID_AVX512ER        0x00400000   /* Intel(R) Advanced Vector Extensions ER instruction set                      */
#define   ippCPUID_AVX512PF        0x00800000   /* Intel(R) Advanced Vector Extensions PF instruction set                      */
#define   ippCPUID_AVX512BW        0x01000000   /* Intel(R) Advanced Vector Extensions BW instruction set                      */
#define   ippCPUID_AVX512DQ        0x02000000   /* Intel(R) Advanced Vector Extensions DQ instruction set                      */
#define   ippCPUID_AVX512VL        0x04000000   /* Intel(R) Advanced Vector Extensions VL instruction set                      */
#define   ippCPUID_AVX512VBMI      0x08000000   /* Intel(R) Advanced Vector Extensions Bit Manipulation instructions           */
#define   ippCPUID_MPX             0x10000000   /* Intel(R) Memory Protection Extensions                                       */
#define   ippCPUID_AVX512_4FMADDPS 0x20000000   /* Intel(R) Advanced Vector Extensions DL floating-point single precision      */
#define   ippCPUID_AVX512_4VNNIW   0x40000000   /* Intel(R) Advanced Vector Extensions DL enhanced word variable precision     */
#define   ippCPUID_KNC             0x80000000   /* Intel(R) Xeon Phi(TM) Coprocessor                                           */
#if defined( _WIN32 ) || defined ( _WIN64 )
 #define   ippCPUID_AVX512IFMA     0x100000000L   /* Intel(R) Advanced Vector Extensions IFMA (PMADD52) instruction set        */
 #define   ippCPUID_NOCHECK    0x8000000000000000L   /* Force ippSetCpuFeatures to set CPU features without check              */
 #define   ippCPUID_GETINFO_A  0x616f666e69746567L   /* Force ipp_GetCpuFeatures to work as cpuid instruction                  */  
 #define   ippCPUID_AVX512VPOPCNTDQ 0x400000000L     /* Intel(R) Advanced Vector Extensions 512 VPOPCNTDQ instruction set      */
 #define   ippCPUID_AVX512_BITALG   0x800000000L     /* Intel(R) Advanced Vector Extensions 512 BITALG instruction set         */
#else
 #define   ippCPUID_AVX512IFMA     0x100000000LL   /* Intel(R) Advanced Vector Extensions IFMA (PMADD52) instruction set       */
 #define   ippCPUID_NOCHECK    0x8000000000000000LL
 #define   ippCPUID_GETINFO_A  0x616f666e69746567LL
 #define   ippCPUID_AVX512VPOPCNTDQ 0x400000000LL    /* Intel(R) Advanced Vector Extensions 512 VPOPCNTDQ instruction set      */
 #define   ippCPUID_AVX512_BITALG   0x800000000LL    /* Intel(R) Advanced Vector Extensions 512 BITALG instruction set         */
#endif

#endif /* IPP_CPU_FEATURES__ */

#define IPP_COUNT_OF( obj )  (sizeof(obj)/sizeof(obj[0]))

typedef struct {
    int type;
    int level;
    int size;
} IppCache;

/*****************************************************************************/
/*                   Below are ippSP domain specific definitions             */
/*****************************************************************************/
typedef enum {
    ippRndZero,
    ippRndNear,
    ippRndFinancial,
    ippRndHintAccurate=0x10
} IppRoundMode;


typedef enum {
    ippAlgHintNone,
    ippAlgHintFast,
    ippAlgHintAccurate
} IppHintAlgorithm;

typedef enum {
    ippCmpLess,
    ippCmpLessEq,
    ippCmpEq,
    ippCmpGreaterEq,
    ippCmpGreater
} IppCmpOp;

typedef enum {
    ippAlgAuto    = 0x00000000,
    ippAlgDirect  = 0x00000001,
    ippAlgFFT     = 0x00000002,
    ippAlgMask    = 0x000000FF
} IppAlgType;

typedef enum {
    ippsNormNone  = 0x00000000, /* default */
    ippsNormA     = 0x00000100, /* biased normalization */
    ippsNormB     = 0x00000200, /* unbiased normalization */
    ippsNormMask  = 0x0000FF00
} IppsNormOp;

typedef enum {
    ippNormInf  =   0x00000001,
    ippNormL1   =   0x00000002,
    ippNormL2   =   0x00000004
} IppNormType;

enum {
    IPP_FFT_DIV_FWD_BY_N = 1,
    IPP_FFT_DIV_INV_BY_N = 2,
    IPP_FFT_DIV_BY_SQRTN = 4,
    IPP_FFT_NODIV_BY_ANY = 8
};

enum {
    IPP_DIV_FWD_BY_N = 1,
    IPP_DIV_INV_BY_N = 2,
    IPP_DIV_BY_SQRTN = 4,
    IPP_NODIV_BY_ANY = 8
};

typedef struct {
    Ipp32f rho;
    Ipp32f theta;
} IppPointPolar;

typedef enum {ippWinBartlett,ippWinBlackman,ippWinHamming,ippWinHann,ippWinRect} IppWinType;

typedef enum { ippButterworth, ippChebyshev1 } IppsIIRFilterType;

typedef enum  { ippZCR=0,   ippZCXor,   ippZCC } IppsZCType;

#if !defined( _OWN_BLDPCS )

typedef struct {
    int left;
    int right;
} IppsROI;

typedef struct RandUniState_8u IppsRandUniState_8u;
typedef struct RandUniState_16s IppsRandUniState_16s;
typedef struct RandUniState_32f IppsRandUniState_32f;
typedef struct RandUniState_64f IppsRandUniState_64f;

typedef struct RandGaussState_8u IppsRandGaussState_8u;
typedef struct RandGaussState_16s IppsRandGaussState_16s;
typedef struct RandGaussState_32f IppsRandGaussState_32f;
typedef struct RandGaussState_64f IppsRandGaussState_64f;

typedef struct FFTSpec_C_32fc   IppsFFTSpec_C_32fc;
typedef struct FFTSpec_C_32f    IppsFFTSpec_C_32f;
typedef struct FFTSpec_R_32f    IppsFFTSpec_R_32f;

typedef struct FFTSpec_C_64fc   IppsFFTSpec_C_64fc;
typedef struct FFTSpec_C_64f    IppsFFTSpec_C_64f;
typedef struct FFTSpec_R_64f    IppsFFTSpec_R_64f;

typedef struct DFTSpec_C_32fc       IppsDFTSpec_C_32fc;
typedef struct DFTSpec_C_32f        IppsDFTSpec_C_32f;
typedef struct DFTSpec_R_32f        IppsDFTSpec_R_32f;

typedef struct DFTSpec_C_64fc       IppsDFTSpec_C_64fc;
typedef struct DFTSpec_C_64f        IppsDFTSpec_C_64f;
typedef struct DFTSpec_R_64f        IppsDFTSpec_R_64f;

typedef struct DCTFwdSpec_32f IppsDCTFwdSpec_32f;
typedef struct DCTInvSpec_32f IppsDCTInvSpec_32f;

typedef struct DCTFwdSpec_64f IppsDCTFwdSpec_64f;
typedef struct DCTInvSpec_64f IppsDCTInvSpec_64f;

typedef struct sWTFwdState_32f    IppsWTFwdState_32f;
typedef struct sWTFwdState_8u32f  IppsWTFwdState_8u32f;
typedef struct sWTFwdState_16s32f IppsWTFwdState_16s32f;
typedef struct sWTFwdState_16u32f IppsWTFwdState_16u32f;
typedef struct sWTInvState_32f    IppsWTInvState_32f;
typedef struct sWTInvState_32f8u  IppsWTInvState_32f8u;
typedef struct sWTInvState_32f16s IppsWTInvState_32f16s;
typedef struct sWTInvState_32f16u IppsWTInvState_32f16u;

typedef struct IIRState_32f      IppsIIRState_32f;
typedef struct IIRState_32fc     IppsIIRState_32fc;
typedef struct IIRState32f_16s   IppsIIRState32f_16s;
typedef struct IIRState32fc_16sc IppsIIRState32fc_16sc;
typedef struct IIRState_64f      IppsIIRState_64f;
typedef struct IIRState_64fc     IppsIIRState_64fc;
typedef struct IIRState64f_32f   IppsIIRState64f_32f;
typedef struct IIRState64fc_32fc IppsIIRState64fc_32fc;
typedef struct IIRState64f_32s   IppsIIRState64f_32s;
typedef struct IIRState64fc_32sc IppsIIRState64fc_32sc;
typedef struct IIRState64f_16s   IppsIIRState64f_16s;
typedef struct IIRState64fc_16sc IppsIIRState64fc_16sc;

typedef struct FIRSpec_32f        IppsFIRSpec_32f;
typedef struct FIRSpec_64f        IppsFIRSpec_64f;
typedef struct FIRSpec_32fc       IppsFIRSpec_32fc;
typedef struct FIRSpec_64fc       IppsFIRSpec_64fc;

typedef struct FIRLMSState_32f    IppsFIRLMSState_32f;
typedef struct FIRLMSState32f_16s IppsFIRLMSState32f_16s;

typedef struct HilbertSpec IppsHilbertSpec;

typedef struct FIRSparseState_32f IppsFIRSparseState_32f;
typedef struct IIRSparseState_32f IppsIIRSparseState_32f;

typedef struct ResamplingPolyphase_16s      IppsResamplingPolyphase_16s;
typedef struct ResamplingPolyphaseFixed_16s IppsResamplingPolyphaseFixed_16s;
typedef struct ResamplingPolyphase_32f      IppsResamplingPolyphase_32f;
typedef struct ResamplingPolyphaseFixed_32f IppsResamplingPolyphaseFixed_32f;

#endif /* _OWN_BLDPCS */

/*****************************************************************************/
/*                   Below are ippIP domain specific definitions             */
/*****************************************************************************/
#define IPP_TEMPORAL_COPY      0x0
#define IPP_NONTEMPORAL_STORE  0x01
#define IPP_NONTEMPORAL_LOAD   0x02

typedef int IppEnum;

#define IPP_DEG_TO_RAD( deg ) ( (deg)/180.0 * IPP_PI )

typedef enum {
    ippiNormNone        = 0x00000000, /* default */
    ippiNorm            = 0x00000100, /* normalized form */
    ippiNormCoefficient = 0x00000200, /* correlation coefficient in the range [-1.0 ... 1.0] */
    ippiNormMask        = 0x0000FF00
} IppiNormOp;

typedef enum {
   ippiROIFull   = 0x00000000,
   ippiROIValid  = 0x00010000,
   ippiROISame   = 0x00020000,
   ippiROIMask   = 0x00FF0000
} IppiROIShape;

typedef enum {
   ippC0    =  0,
   ippC1    =  1,
   ippC2    =  2,
   ippC3    =  3,
   ippC4    =  4,
   ippP2    =  5,
   ippP3    =  6,
   ippP4    =  7,
   ippAC1   =  8,
   ippAC4   =  9,
   ippA0C4  = 10,
   ippAP4   = 11
} IppChannels;

typedef enum _IppiBorderType {
    ippBorderRepl         =  1,
    ippBorderWrap         =  2,
    ippBorderMirror       =  3, /* left border: 012... -> 21012... */
    ippBorderMirrorR      =  4, /* left border: 012... -> 210012... */
    ippBorderDefault      =  5,
    ippBorderConst        =  6,
    ippBorderTransp       =  7,

    /* Flags to use source image memory pixels from outside of the border in particular directions */
    ippBorderInMemTop     =  0x0010,
    ippBorderInMemBottom  =  0x0020,
    ippBorderInMemLeft    =  0x0040,
    ippBorderInMemRight   =  0x0080,
    ippBorderInMem        =  ippBorderInMemLeft|ippBorderInMemTop|ippBorderInMemRight|ippBorderInMemBottom,

    /* Flags to use source image memory pixels from outside of the border for first stage only in multi-stage filters */
    ippBorderFirstStageInMemTop     =  0x0100,
    ippBorderFirstStageInMemBottom  =  0x0200,
    ippBorderFirstStageInMemLeft    =  0x0400,
    ippBorderFirstStageInMemRight   =  0x0800,
    ippBorderFirstStageInMem        = ippBorderFirstStageInMemLeft|ippBorderFirstStageInMemTop|ippBorderFirstStageInMemRight|ippBorderFirstStageInMemBottom
} IppiBorderType;

typedef enum {
    ippAxsHorizontal,
    ippAxsVertical,
    ippAxsBoth,
    ippAxs45,
    ippAxs135
} IppiAxis;

typedef struct {
    int x;
    int y;
    int width;
    int height;
} IppiRect;

typedef struct {
    int x;
    int y;
} IppiPoint;

typedef struct {
    int width;
    int height;
} IppiSize;

typedef struct {
    Ipp32f x;
    Ipp32f y;
} IppiPoint_32f;

typedef enum  _IppiMaskSize {
    ippMskSize1x3 = 13,
    ippMskSize1x5 = 15,
    ippMskSize3x1 = 31,
    ippMskSize3x3 = 33,
    ippMskSize5x1 = 51,
    ippMskSize5x5 = 55
} IppiMaskSize;

enum {
    IPPI_INTER_NN     = 1,
    IPPI_INTER_LINEAR = 2,
    IPPI_INTER_CUBIC  = 4,
    IPPI_INTER_CUBIC2P_BSPLINE,     /* two-parameter cubic filter (B=1, C=0) */
    IPPI_INTER_CUBIC2P_CATMULLROM,  /* two-parameter cubic filter (B=0, C=1/2) */
    IPPI_INTER_CUBIC2P_B05C03,      /* two-parameter cubic filter (B=1/2, C=3/10) */
    IPPI_INTER_SUPER  = 8,
    IPPI_INTER_LANCZOS = 16,
    IPPI_ANTIALIASING  = (1 << 29),
    IPPI_SUBPIXEL_EDGE = (1 << 30),
    IPPI_SMOOTH_EDGE   = (1 << 31)
};

typedef enum {
    ippNearest = IPPI_INTER_NN,
    ippLinear = IPPI_INTER_LINEAR,
    ippCubic = IPPI_INTER_CUBIC2P_CATMULLROM,
    ippLanczos = IPPI_INTER_LANCZOS,
    ippHahn = 0,
    ippSuper = IPPI_INTER_SUPER
} IppiInterpolationType;

typedef enum {
    ippPolyphase_1_2,
    ippPolyphase_3_5,
    ippPolyphase_2_3,
    ippPolyphase_7_10,
    ippPolyphase_3_4
} IppiFraction;

enum {
    IPP_FASTN_ORIENTATION = 0x0001,
    IPP_FASTN_NMS         = 0x0002,
    IPP_FASTN_CIRCLE      = 0X0004,
    IPP_FASTN_SCORE_MODE0 = 0X0020
};

#if !defined( _OWN_BLDPCS )

typedef enum {
    ippAlphaOver,
    ippAlphaIn,
    ippAlphaOut,
    ippAlphaATop,
    ippAlphaXor,
    ippAlphaPlus,
    ippAlphaOverPremul,
    ippAlphaInPremul,
    ippAlphaOutPremul,
    ippAlphaATopPremul,
    ippAlphaXorPremul,
    ippAlphaPlusPremul
} IppiAlphaType;

typedef struct DeconvFFTState_32f_C1R IppiDeconvFFTState_32f_C1R;
typedef struct DeconvFFTState_32f_C3R IppiDeconvFFTState_32f_C3R;
typedef struct DeconvLR_32f_C1R IppiDeconvLR_32f_C1R;
typedef struct DeconvLR_32f_C3R IppiDeconvLR_32f_C3R;

typedef enum {
    ippiFilterBilateralGauss = 100,
    ippiFilterBilateralGaussFast = 101
} IppiFilterBilateralType;

typedef struct FilterBilateralSpec IppiFilterBilateralSpec;

typedef enum {
    ippDistNormL1   =   0x00000002
} IppiDistanceMethodType;

typedef enum {
    ippResizeFilterHann,
    ippResizeFilterLanczos
} IppiResizeFilterType;

typedef struct ResizeFilterState IppiResizeFilterState;

typedef struct {
    Ipp32u borderLeft;
    Ipp32u borderTop;
    Ipp32u borderRight;
    Ipp32u borderBottom;
} IppiBorderSize;

typedef enum {
    ippWarpForward,
    ippWarpBackward
} IppiWarpDirection;

typedef enum {
    ippWarpAffine,
    ippWarpPerspective,
    ippWarpBilinear
} IppiWarpTransformType;


typedef struct ResizeSpec_32f   IppiResizeSpec_32f;
typedef struct ResizeYUV422Spec IppiResizeYUV422Spec;
typedef struct ResizeYUV420Spec IppiResizeYUV420Spec;

typedef struct ResizeSpec_64f   IppiResizeSpec_64f;

typedef struct IppiWarpSpec     IppiWarpSpec;

typedef struct FilterBorderSpec IppiFilterBorderSpec;

typedef struct ThresholdAdaptiveSpec IppiThresholdAdaptiveSpec;

typedef struct HistogramSpec IppiHistogramSpec;

typedef struct {
   int   cvCompatible;  /* openCV compatible output format */
   int   cellSize;      /* squre cell size (pixels) */
   int   blockSize;     /* square block size (pixels) */
   int   blockStride;   /* block displacement (the same for x- and y- directions) */
   int   nbins;         /* required number of bins */
   Ipp32f   sigma;      /* gaussian factor of HOG block weights */
   Ipp32f   l2thresh;   /* normalization factor */
   IppiSize winSize;    /* detection window size (pixels) */
} IppiHOGConfig;

typedef struct FFT2DSpec_C_32fc IppiFFTSpec_C_32fc;
typedef struct FFT2DSpec_R_32f IppiFFTSpec_R_32f;

typedef struct DFT2DSpec_C_32fc IppiDFTSpec_C_32fc;
typedef struct DFT2DSpec_R_32f IppiDFTSpec_R_32f;

typedef struct DCT2DFwdSpec_32f IppiDCTFwdSpec_32f;
typedef struct DCT2DInvSpec_32f IppiDCTInvSpec_32f;

typedef struct iWTFwdSpec_32f_C1R IppiWTFwdSpec_32f_C1R;
typedef struct iWTInvSpec_32f_C1R IppiWTInvSpec_32f_C1R;
typedef struct iWTFwdSpec_32f_C3R IppiWTFwdSpec_32f_C3R;
typedef struct iWTInvSpec_32f_C3R IppiWTInvSpec_32f_C3R;

typedef struct MomentState64f IppiMomentState_64f;
typedef Ipp64f IppiHuMoment_64f[7];

typedef struct LUT_Spec IppiLUT_Spec;

#define IPP_HOG_MAX_CELL   (16)  /* max size of CELL */
#define IPP_HOG_MAX_BLOCK  (64)  /* max size of BLOCK */
#define IPP_HOG_MAX_BINS   (16)  /* max number of BINS */

typedef struct _ipHOG   IppiHOGSpec;

#endif /* _OWN_BLDPCS */


         /**** Below are 3D Image (Volume) Processing specific definitions ****/

typedef struct {
    int width;
    int height;
    int depth;
} IpprVolume;

typedef struct {
    int x;
    int y;
    int z;
    int width;
    int height;
    int depth;
} IpprCuboid;

typedef struct {
    int x;
    int y;
    int z;
} IpprPoint;

/*****************************************************************************/
/*                   Below are ippCV domain specific definitions             */
/*****************************************************************************/

typedef enum _IppiDifferentialKernel
{
    ippFilterSobelVert,
    ippFilterSobelHoriz,
    ippFilterSobel,
    ippFilterScharrVert,
    ippFilterScharrHoriz,
    ippFilterScharr,
    ippFilterCentralDiffVert,
    ippFilterCentralDiffHoriz,
    ippFilterCentralDiff
}IppiDifferentialKernel;

#if !defined( _OWN_BLDPCS )

typedef enum _IppiKernelType {
    ippKernelSobel     =  0,
    ippKernelScharr    =  1,
    ippKernelSobelNeg  =  2
} IppiKernelType;

typedef enum _IppiNorm {
    ippiNormInf = 0,
    ippiNormL1 = 1,
    ippiNormL2 = 2,
    ippiNormFM = 3
} IppiNorm;

typedef struct ipcvMorphState IppiMorphState;
typedef struct ipcvMorphStateL IppiMorphStateL;
typedef struct ipcvMorphAdvState IppiMorphAdvState;
typedef struct ipcvMorphAdvStateL IppiMorphAdvStateL;
typedef struct ipcvMorphGrayState_8u IppiMorphGrayState_8u;
typedef struct ipcvMorphGrayState_8uL IppiMorphGrayState_8uL;
typedef struct ipcvMorphGrayState_32f IppiMorphGrayState_32f;
typedef struct ipcvMorphGrayState_32fL IppiMorphGrayState_32fL;

typedef struct ipcvConvState IppiConvState;

typedef struct _IppiConnectedComp {
    Ipp64f   area;    /*  area of the segmented component  */
    Ipp64f   value[3];/*  gray scale value of the segmented component  */
    IppiRect rect;    /*  bounding rectangle of the segmented component  */
} IppiConnectedComp;

typedef struct PyramidState IppiPyramidState;

typedef IppiPyramidState IppiPyramidDownState_8u_C1R;
typedef IppiPyramidState IppiPyramidDownState_16u_C1R;
typedef IppiPyramidState IppiPyramidDownState_32f_C1R;
typedef IppiPyramidState IppiPyramidDownState_8u_C3R;
typedef IppiPyramidState IppiPyramidDownState_16u_C3R;
typedef IppiPyramidState IppiPyramidDownState_32f_C3R;
typedef IppiPyramidState IppiPyramidUpState_8u_C1R;
typedef IppiPyramidState IppiPyramidUpState_16u_C1R;
typedef IppiPyramidState IppiPyramidUpState_32f_C1R;
typedef IppiPyramidState IppiPyramidUpState_8u_C3R;
typedef IppiPyramidState IppiPyramidUpState_16u_C3R;
typedef IppiPyramidState IppiPyramidUpState_32f_C3R;


typedef struct _IppiPyramid {
    Ipp8u         **pImage;
    IppiSize      *pRoi;
    Ipp64f        *pRate;
    int           *pStep;
    Ipp8u         *pState;
    int            level;
} IppiPyramid;

typedef struct OptFlowPyrLK IppiOptFlowPyrLK;

typedef IppiOptFlowPyrLK IppiOptFlowPyrLK_8u_C1R;
typedef IppiOptFlowPyrLK IppiOptFlowPyrLK_16u_C1R;
typedef IppiOptFlowPyrLK IppiOptFlowPyrLK_32f_C1R;

typedef struct ipcvHaarClassifier_32f IppiHaarClassifier_32f;
typedef struct ipcvHaarClassifier_32s IppiHaarClassifier_32s;

typedef struct ipcvFGHistogramState_8u_C1R IppFGHistogramState_8u_C1R;
typedef struct ipcvFGHistogramState_8u_C3R IppFGHistogramState_8u_C3R;

typedef struct ipcvFGGaussianState_8u_C1R IppFGGaussianState_8u_C1R;
typedef struct ipcvFGGaussianState_8u_C3R IppFGGaussianState_8u_C3R;

typedef enum _IppiInpaintFlag {
    IPP_INPAINT_TELEA    =  0,
    IPP_INPAINT_NS       =  1
} IppiInpaintFlag;

typedef struct ippcvFilterGaussianSpec IppFilterGaussianSpec;

typedef struct ipcvInpaintState_8u IppiInpaintState_8u_C1R;
typedef struct ipcvInpaintState_8u IppiInpaintState_8u_C3R;

typedef struct HoughProbSpec IppiHoughProbSpec;

typedef struct FastNSpec IppiFastNSpec;

typedef struct _IppiCornerFastN {
    int     x;
    int     y;
    int     cornerType;
    int     orientation;
    float   angle;
    float   score;
} IppiCornerFastN;

typedef struct FGMMState_8u_C3R IppFGMMState_8u_C3R;

typedef struct
{
  unsigned int numFrames;
  unsigned int maxNGauss;
  Ipp32f       varInit;
  Ipp32f       varMin;
  Ipp32f       varMax;
  Ipp32f       varWBRatio;
  Ipp32f       bckgThr;
  Ipp32f       varNGRatio;
  Ipp32f       reduction;
  Ipp8u        shadowValue;
  char         shadowFlag;
  Ipp32f       shadowRatio;
} IppFGMModel;

#endif /* _OWN_BLDPCS */

#define IPP_SEGMENT_QUEUE     0x01
#define IPP_SEGMENT_DISTANCE  0x02
#define IPP_SEGMENT_BORDER_4  0x40
#define IPP_SEGMENT_BORDER_8  0x80

#define IPP_TRUNC(a,b) ((a)&~((b)-1))
#define IPP_APPEND(a,b) (((a)+(b)-1)&~((b)-1))

/*****************************************************************************/
/*                   Below are ippCC domain specific definitions             */
/*****************************************************************************/
enum {
     IPP_UPPER        = 1,
     IPP_LEFT         = 2,
     IPP_CENTER       = 4,
     IPP_RIGHT        = 8,
     IPP_LOWER        = 16,
     IPP_UPPER_LEFT   = 32,
     IPP_UPPER_RIGHT  = 64,
     IPP_LOWER_LEFT   = 128,
     IPP_LOWER_RIGHT  = 256
};

#if !defined( _OWN_BLDPCS )

typedef enum {
    ippDitherNone,
    ippDitherFS,
    ippDitherJJN,
    ippDitherStucki,
    ippDitherBayer
} IppiDitherType;

#endif /* _OWN_BLDPCS */

/*****************************************************************************/
/*                   Below are ippCH domain specific definitions             */
/*****************************************************************************/

#if !defined( _OWN_BLDPCS )

typedef struct {
 void *pFind;
 int lenFind;
} IppRegExpFind;

typedef struct RegExpState IppRegExpState;

typedef enum {
    ippFmtASCII = 0,
    ippFmtUTF8
} IppRegExpFormat;

typedef struct RegExpReplaceState IppRegExpReplaceState;

#endif /* _OWN_BLDPCS */

/*****************************************************************************/
/*                   Below are ippDC domain specific definitions             */
/*****************************************************************************/

#if !defined ( _OWN_BLDPCS )

typedef struct MTFState_8u IppMTFState_8u;

typedef enum {
    ippBWTItohTanakaLimSort,
    ippBWTItohTanakaUnlimSort,
    ippBWTSuffixSort,
    ippBWTAutoSort
} IppBWTSortAlgorithmHint;

typedef struct LZSSState_8u IppLZSSState_8u;

typedef struct LZ77State_8u IppLZ77State_8u;
typedef enum{
   IppLZ77FastCompr,
   IppLZ77AverageCompr,
   IppLZ77BestCompr
} IppLZ77ComprLevel;
typedef enum{
   IppLZ77NoChcksm,
   IppLZ77Adler32,
   IppLZ77CRC32
} IppLZ77Chcksm;
typedef enum {
   IppLZ77NoFlush,
   IppLZ77SyncFlush,
   IppLZ77FullFlush,
   IppLZ77FinishFlush
} IppLZ77Flush;
typedef struct IppLZ77Pairs_16u {
   Ipp16u length;
   Ipp16u offset;
} IppLZ77Pair;
typedef enum {
   IppLZ77StatusInit,
   IppLZ77StatusLZ77Process,
   IppLZ77StatusHuffProcess,
   IppLZ77StatusFinal
} IppLZ77DeflateStatus;
typedef enum {
  IppLZ77UseFixed,
  IppLZ77UseDynamic,
  IppLZ77UseStored
} IppLZ77HuffMode;
typedef enum {
  IppLZ77InflateStatusInit,
  IppLZ77InflateStatusHuffProcess,
  IppLZ77InflateStatusLZ77Process,
  IppLZ77InflateStatusFinal
} IppLZ77InflateStatus;

typedef struct IppInflateState {
  const Ipp8u* pWindow;          /* pointer to the sliding window
                                    (the dictionary for the LZ77 algorithm) */
  unsigned int winSize;          /* size of the sliding window */
  unsigned int tableType;        /* type of Huffman code tables
                                    (for example, 0 - tables for Fixed Huffman codes) */
  unsigned int tableBufferSize;  /* (ENOUGH = 2048) * (sizeof(code) = 4) -
                                    sizeof(IppInflateState) */
} IppInflateState;

typedef enum { /* this type is used as a translator of the inflate_mode type from zlib */
  ippTYPE,
  ippLEN,
  ippLENEXT
} IppInflateMode;

typedef struct {
  Ipp16u freq;
  Ipp16u code;
} IppDeflateFreqTable;

typedef struct {
  Ipp16u code;
  Ipp16u len;
} IppDeflateHuffCode;

typedef struct RLEState_BZ2 IppRLEState_BZ2;

typedef struct EncodeHuffState_BZ2 IppEncodeHuffState_BZ2;

typedef struct DecodeHuffState_BZ2 IppDecodeHuffState_BZ2;

typedef enum {
    IppLZO1XST,      /* Single-threaded, generic LZO-compatible*/
    IppLZO1XMT      /* Multi-threaded */
} IppLZOMethod ;

typedef struct LZOState_8u IppLZOState_8u;

#endif /* _OWN_BLDPCS */

/* /////////////////////////////////////////////////////////////////////////////
//        The following enumerator defines a status of Intel(R) IPP operations
//                     negative value means error
*/
    /* errors */
#define ippStsCpuNotSupportedErr         -9999/* The target CPU is not supported. */
#define ippStsInplaceModeNotSupportedErr -9998/* The inplace operation is currently not supported. */

#define ippStsIIRIIRLengthErr             -234 /* Vector length for IIRIIR function is less than 3*(IIR order) */
#define ippStsWarpTransformTypeErr        -233 /* The warp transform type is illegal */
#define ippStsExceededSizeErr             -232 /* Requested size exceeded the maximum supported ROI size */
#define ippStsWarpDirectionErr            -231 /* The warp transform direction is illegal */
#define ippStsFilterTypeErr               -230 /* The filter type is incorrect or not supported */
#define ippStsNormErr                     -229 /* The norm is incorrect or not supported */
#define ippStsAlgTypeErr                  -228 /* Algorithm type is not supported. */
#define ippStsMisalignedOffsetErr         -227 /* The offset is not aligned with an element. */
#define ippStsBorderErr                   -225 /* Illegal value for border type.*/
#define ippStsDitherTypeErr               -224 /* Dithering type is not supported. */
#define ippStsUnknownStatusCodeErr        -216 /* Unknown status code. */
#define ippStsLzoBrokenStreamErr          -214 /* LZO safe decompression function cannot decode LZO stream.*/
#define ippStsRoundModeNotSupportedErr    -213 /* Rounding mode is not supported. */
#define ippStsDecimateFractionErr         -212 /* Fraction in Decimate is not supported. */
#define ippStsWeightErr                   -211 /* Incorrect value for weight. */
#define ippStsQualityIndexErr             -210 /* Cannot calculate the quality index for an image filled with a constant. */
#define ippStsIIRPassbandRippleErr        -209 /* Ripple in passband for Chebyshev1 design is less than zero, equal to zero, or greater than 29. */
#define ippStsFilterFrequencyErr          -208 /* Cutoff frequency of filter is less than zero, equal to zero, or greater than 0.5. */
#define ippStsIIRGenOrderErr              -206 /* Order of the IIR filter for design is less than 1, or greater than 12. */
#define ippStsConvergeErr                 -205 /* The algorithm does not converge. */
#define ippStsSizeMatchMatrixErr          -204 /* The sizes of the source matrices are unsuitable. */
#define ippStsCountMatrixErr              -203 /* Count value is less than or equal to zero. */
#define ippStsRoiShiftMatrixErr           -202 /* RoiShift value is negative or not divisible by the size of the data type. */
#define ippStsSrcDataErr                  -200 /* The source buffer contains unsupported data.*/
#define ippStsSingularErr                 -195 /* Matrix is singular. */
#define ippStsSparseErr                   -194 /* Positions of taps are not in ascending order, or are negative, or repetitive. */
#define ippStsRegExpOptionsErr            -190 /* RegExp: Options for the pattern are incorrect. */
#define ippStsRegExpErr                   -189 /* RegExp: The structure pRegExpState contains incorrect data. */
#define ippStsRegExpMatchLimitErr         -188 /* RegExp: The match limit is exhausted. */
#define ippStsRegExpQuantifierErr         -187 /* RegExp: Incorrect quantifier. */
#define ippStsRegExpGroupingErr           -186 /* RegExp: Incorrect grouping. */
#define ippStsRegExpBackRefErr            -185 /* RegExp: Incorrect back reference. */
#define ippStsRegExpChClassErr            -184 /* RegExp: Incorrect character class. */
#define ippStsRegExpMetaChErr             -183 /* RegExp: Incorrect metacharacter. */
#define ippStsStrideMatrixErr             -182 /* Stride value is not positive or not divisible by the size of the data type. */
#define ippStsNoiseRangeErr               -125 /* Noise value for Wiener Filter is out of range. */
#define ippStsNotEvenStepErr              -108 /* Step value is not pixel multiple. */
#define ippStsHistoNofLevelsErr           -107 /* Number of levels for histogram is less than 2. */
#define ippStsLUTNofLevelsErr             -106 /* Number of levels for LUT is less than 2. */
#define ippStsChannelOrderErr              -60 /* Incorrect order of the destination channels. */
#define ippStsDataTypeErr                  -59 /* Data type is incorrect or not supported. */
#define ippStsQuadErr                      -58 /* The quadrangle is nonconvex or degenerates into triangle, line, or point */
#define ippStsRectErr                      -57 /* Size of the rectangle region is less than, or equal to 1. */
#define ippStsCoeffErr                     -56 /* Incorrect values for transformation coefficients. */
#define ippStsNoiseValErr                  -55 /* Incorrect value for noise amplitude for dithering. */
#define ippStsDitherLevelsErr              -54 /* Number of dithering levels is out of range. */
#define ippStsNumChannelsErr               -53 /* Number of channels is incorrect, or not supported. */
#define ippStsCOIErr                       -52 /* COI is out of range. */
#define ippStsDivisorErr                   -51 /* Divisor is equal to zero, function is aborted. */
#define ippStsAlphaTypeErr                 -50 /* Illegal type of image compositing operation. */
#define ippStsGammaRangeErr                -49 /* Gamma range bounds is less than, or equal to zero. */
#define ippStsChannelErr                   -47 /* Illegal channel number. */
#define ippStsToneMagnErr                  -46 /* Tone magnitude is less than, or equal to zero. */
#define ippStsToneFreqErr                  -45 /* Tone frequency is negative, or greater than, or equal to 0.5. */
#define ippStsTonePhaseErr                 -44 /* Tone phase is negative, or greater than, or equal to 2*PI. */
#define ippStsTrnglMagnErr                 -43 /* Triangle magnitude is less than, or equal to zero. */
#define ippStsTrnglFreqErr                 -42 /* Triangle frequency is negative, or greater than, or equal to 0.5. */
#define ippStsTrnglPhaseErr                -41 /* Triangle phase is negative, or greater than, or equal to 2*PI. */
#define ippStsTrnglAsymErr                 -40 /* Triangle asymmetry is less than -PI, or greater than, or equal to PI. */
#define ippStsHugeWinErr                   -39 /* Kaiser window is too big. */
#define ippStsJaehneErr                    -38 /* Magnitude value is negative. */
#define ippStsStrideErr                    -37 /* Stride value is less than the length of the row. */
#define ippStsEpsValErr                    -36 /* Negative epsilon value. */
#define ippStsWtOffsetErr                  -35 /* Invalid offset value for wavelet filter. */
#define ippStsAnchorErr                    -34 /* Anchor point is outside the mask. */
#define ippStsMaskSizeErr                  -33 /* Invalid mask size. */
#define ippStsShiftErr                     -32 /* Shift value is less than zero. */
#define ippStsSampleFactorErr              -31 /* Sampling factor is less than,or equal to zero. */
#define ippStsSamplePhaseErr               -30 /* Phase value is out of range: 0 <= phase < factor. */
#define ippStsFIRMRFactorErr               -29 /* MR FIR sampling factor is less than, or equal to zero. */
#define ippStsFIRMRPhaseErr                -28 /* MR FIR sampling phase is negative, or greater than, or equal to the sampling factor. */
#define ippStsRelFreqErr                   -27 /* Relative frequency value is out of range. */
#define ippStsFIRLenErr                    -26 /* Length of a FIR filter is less than, or equal to zero. */
#define ippStsIIROrderErr                  -25 /* Order of an IIR filter is not valid. */
#define ippStsResizeFactorErr              -24 /* Resize factor(s) is less than, or equal to zero. */
#define ippStsInterpolationErr             -23 /* Invalid interpolation mode. */
#define ippStsMirrorFlipErr                -22 /* Invalid flip mode. */
#define ippStsMoment00ZeroErr              -21 /* Moment value M(0,0) is too small to continue calculations. */
#define ippStsThreshNegLevelErr            -20 /* Negative value of the level in the threshold operation. */
#define ippStsThresholdErr                 -19 /* Invalid threshold bounds. */
#define ippStsFftFlagErr                   -18 /* Invalid value for the FFT flag parameter. */
#define ippStsFftOrderErr                  -17 /* Invalid value for the FFT order parameter. */
#define ippStsStepErr                      -16 /* Step value is not valid. */

#ifndef IPPSTATUS_H__
#define IPPSTATUS_H__
typedef signed int IppStatus;
    /* start of common with ippCrypto part - any changes MUST be done in both repositories - IPP & ippCrypto */
  #define ippStsLoadDynErr                  -221 /* Error when loading the dynamic library. */
  #define ippStsLengthErr                    -15 /* Incorrect value for string length. */
  #define ippStsNotSupportedModeErr          -14 /* The requested mode is currently not supported. */
  #define ippStsContextMatchErr              -13 /* Context parameter does not match the operation. */
  #define ippStsScaleRangeErr                -12 /* Scale bounds are out of range. */
  #define ippStsOutOfRangeErr                -11 /* Argument is out of range, or point is outside the image. */
  #define ippStsDivByZeroErr                 -10 /* An attempt to divide by zero. */
  #define ippStsMemAllocErr                   -9 /* Memory allocated for the operation is not enough.*/
  #define ippStsNullPtrErr                    -8 /* Null pointer error. */
  #define ippStsRangeErr                      -7 /* Incorrect values for bounds: the lower bound is greater than the upper bound. */
  #define ippStsSizeErr                       -6 /* Incorrect value for data size. */
  #define ippStsBadArgErr                     -5 /* Incorrect arg/param of the function. */
  #define ippStsNoMemErr                      -4 /* Not enough memory for the operation. */
  #define ippStsErr                           -2 /* Unknown/unspecified error */
     /* no errors */
  #define ippStsNoErr                          0 /* No errors. */
     /* warnings  */
  #define ippStsNoOperation                    1 /* No operation has been executed. */
  #define ippStsDivByZero                      2 /* Zero value(s) for the divisor in the Div function. */
  #define ippStsWaterfall                     43 /* Cannot load required library, waterfall is used. */
    /* end of common with ippCrypto part */
#endif /* IPPSTATUS_H__ */

#define ippStsSqrtNegArg                     3 /* Negative value(s) for the argument in the Sqrt function. */
#define ippStsInvZero                        4 /* INF result. Zero value was met by InvThresh with zero level. */
#define ippStsEvenMedianMaskSize             5 /* Even size of the Median Filter mask was replaced with the odd one. */
#define ippStsLnZeroArg                      7 /* Zero value(s) for the argument in the Ln function. */
#define ippStsLnNegArg                       8 /* Negative value(s) for the argument in the Ln function. */
#define ippStsNanArg                         9 /* Argument value is not a number. */
#define ippStsOverflow                      12 /* Overflow in the operation. */
#define ippStsUnderflow                     17 /* Underflow in the operation. */
#define ippStsSingularity                   18 /* Singularity in the operation. */
#define ippStsDomain                        19 /* Argument is out of the function domain.*/
#define ippStsNonIntelCpu                   20 /* The target CPU is not Genuine Intel. */
#define ippStsCpuMismatch                   21 /* Cannot set the library for the given CPU. */
#define ippStsOvermuchStrings               26 /* Number of destination strings is more than expected. */
#define ippStsOverlongString                27 /* Length of one of the destination strings is more than expected. */
#define ippStsAffineQuadChanged             28 /* 4th vertex of destination quad is not equal to customer's one. */
#define ippStsWrongIntersectROI             29 /* ROI has no intersection with the source or destination ROI. No operation. */
#define ippStsWrongIntersectQuad            30 /* Quadrangle has no intersection with the source or destination ROI. No operation. */
#define ippStsSrcSizeLessExpected           32 /* DC: Size of the source buffer is less than the expected one.*/
#define ippStsDstSizeLessExpected           33 /* DC: Size of the destination buffer is less than the expected one. */
#define ippStsDoubleSize                    35 /* Width or height of image is odd. */
#define ippStsNotSupportedCpu               36 /* The CPU is not supported. */
#define ippStsUnknownCacheSize              37 /* The CPU is supported, but the size of the cache is unknown. */
#define ippStsSymKernelExpected             38 /* The Kernel is not symmetric. */
#define ippStsEvenMedianWeight              39 /* Even weight of the Weighted Median Filter is replaced with the odd one. */
#define ippStsWrongIntersectVOI             40 /* VOI has no intersection with the source or destination volume. No operation. */
#define ippStsNoAntialiasing                46 /* The mode does not support antialiasing. */
#define ippStsRepetitiveSrcData             47 /* DC: The source data is too repetitive.*/
#define ippStsSizeWrn                       48 /* The size does not allow to perform full operation. */
#define ippStsFeatureNotSupported           49 /* Current CPU doesn't support at least 1 of the desired features. */
#define ippStsUnknownFeature                50 /* At least one of the desired features is unknown. */
#define ippStsFeaturesCombination           51 /* Wrong combination of features. */
#define ippStsAccurateModeNotSupported      52/* Accurate mode is not supported. */

#define ippStsOk ippStsNoErr

#endif

#define ippRectInfinite ippiWarpGetRectInfinite()

#ifdef __cplusplus
}
#endif

#endif /* __IPPICV_TYPES_H__ */
