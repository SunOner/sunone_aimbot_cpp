/*******************************************************************************
* Copyright 2016 Intel Corporation.
*
*
* This software and the related documents are Intel copyrighted materials, and your use of them is governed by
* the express license under which they were provided to you ('License'). Unless the License provides otherwise,
* you may not use, modify, copy, publish, distribute, disclose or transmit this software or the related
* documents without Intel's prior written permission.
* This software and the related documents are provided as is, with no express or implied warranties, other than
* those that are expressly stated in the License.
*******************************************************************************/

#if !defined( __IPP_IW_IMAGE_FILTER__ )
#define __IPP_IW_IMAGE_FILTER__

#include "iw/iw_image.h"

#ifdef __cplusplus
extern "C" {
#endif

// Derivative operator type enumerator
typedef enum _IwiDerivativeType
{
    iwiDerivHorFirst = 0,  // Horizontal first derivative
    iwiDerivHorSecond,     // Horizontal second derivative
    iwiDerivVerFirst,      // Vertical first derivative
    iwiDerivVerSecond,     // Vertical second derivative
    iwiDerivNVerFirst      // Negative vertical first derivative
} IwiDerivativeType;

/* /////////////////////////////////////////////////////////////////////////////
//                   iwiFilter
///////////////////////////////////////////////////////////////////////////// */

// Auxiliary parameters structure
typedef struct _IwiFilterParams
{
    double              divisor;    // The value by which the computed result is divided for 16s kernel type.
                                    // If divisor is 0 and the sum of the kernel elements is not 0 then result will be normalized by the sum of the kernel elements
                                    // Only integer values are supported
    int                 offset;     // An offset which will be added to the final signed result before converting it to unsigned for 8u and 16u
    IppRoundMode        roundMode;  // Rounding mode
    IppHintAlgorithm    algoMode;   // Accuracy mode
} IwiFilterParams;

// Sets auxiliary parameters to default values
static IW_INLINE void iwiFilter_SetDefaultParams(
    IwiFilterParams *pParams      // [in,out] Pointer to the auxiliary parameters structure
)
{
    if(pParams)
    {
        pParams->divisor    = 0;
        pParams->offset     = 0;
        pParams->roundMode  = ippRndNear;
        pParams->algoMode   = ippAlgHintNone;
    }
}

// Performs filtration of an image with an arbitrary kernel
// Features support:
//      Data types:              k16s(8u,16u,16s), k32f(8u,16u,16s,32f)
//      Channel types:           C1,C3,C4
//      Inplace mode:            no
//      64-bit sizes:            no
//      Internal threading:      no
//      Manual tiling:           yes
//      IwiTile simple tiling:   yes
//      IwiTile pipeline tiling: yes
// Returns:
//      ippStsBorderErr                     border value is illegal
//      ippStsDataTypeErr                   data type is illegal
//      ippStsNumChannelsErr                channels value is illegal
//      ippStsNotEvenStepErr                step value is not divisible by size of elements
//      ippStsNotSupportedModeErr           selected function mode is not supported
//      ippStsNoMemErr                      failed to allocate memory
//      ippStsInplaceModeNotSupportedErr    doesn't support output into the source buffer
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL(IppStatus) iwiFilter(
    const IwiImage         *pSrcImage,      // [in]     Pointer to the source image
    IwiImage               *pDstImage,      // [in,out] Pointer to the destination image
    const IwiImage         *pKernel,        // [in]     Pointer to the filter kernel image. Kernel must be continuous, [16s,32f], [C1]
    const IwiFilterParams  *pAuxParams,     // [in]     Pointer to the auxiliary parameters structure. If NULL - default parameters will be used
    IwiBorderType           border,         // [in]     Extrapolation algorithm for out of image pixels: ippBorderConst, ippBorderRepl, ippBorderMirror, ippBorderInMem
    const Ipp64f           *pBorderVal,     // [in]     Pointer to array of border values for ippBorderConst. One element for each channel. Can be NULL for any other border
    const IwiTile          *pTile           // [in]     Pointer to the IwiTile structure for tiling. If NULL - the whole image will be processed in accordance to size
);

/* /////////////////////////////////////////////////////////////////////////////
//                   iwiFilterBox
///////////////////////////////////////////////////////////////////////////// */

// Auxiliary parameters structure
typedef struct _IwiFilterBoxParams
{
    IwiChDescriptor chDesc;     // Special channels processing mode
} IwiFilterBoxParams;

// Sets auxiliary parameters to default values
static IW_INLINE void iwiFilterBox_SetDefaultParams(
    IwiFilterBoxParams *pParams      // [in,out] Pointer to the auxiliary parameters structure
)
{
    if(pParams)
    {
        pParams->chDesc = iwiChDesc_None;
    }
}

// Applies box filter to the image
// Features support:
//      Data types:              8u,16u,16s,32f
//      Channel types:           C1,C3,C4,C4M1110
//      Inplace mode:            no
//      64-bit sizes:            no
//      Internal threading:      no
//      Manual tiling:           yes
//      IwiTile simple tiling:   yes
//      IwiTile pipeline tiling: yes
// Returns:
//      ippStsBorderErr                     border value is illegal
//      ippStsDataTypeErr                   data type is illegal
//      ippStsNumChannelsErr                channels value is illegal
//      ippStsNotEvenStepErr                step value is not divisible by size of elements
//      ippStsNotSupportedModeErr           selected function mode is not supported
//      ippStsNoMemErr                      failed to allocate memory
//      ippStsInplaceModeNotSupportedErr    doesn't support output into the source buffer
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL(IppStatus) iwiFilterBox(
    const IwiImage             *pSrcImage,      // [in]     Pointer to the source image
    IwiImage                   *pDstImage,      // [in,out] Pointer to the destination image
    IwiSize                     kernelSize,     // [in]     Size of the filter kernel
    const IwiFilterBoxParams   *pAuxParams,     // [in]     Pointer to the auxiliary parameters structure. If NULL - default parameters will be used
    IwiBorderType               border,         // [in]     Extrapolation algorithm for out of image pixels: ippBorderConst, ippBorderRepl, ippBorderMirror, ippBorderInMem
    const Ipp64f               *pBorderVal,     // [in]     Pointer to array of border values for ippBorderConst. One element for each channel. Can be NULL for any other border
    const IwiTile              *pTile           // [in]     Pointer to the IwiTile structure for tiling. If NULL - the whole image will be processed in accordance to size
);

/**/////////////////////////////////////////////////////////////////////////////
//                   iwiFilterSobel
///////////////////////////////////////////////////////////////////////////// */
//
//                               -1  0  1
//              SobelVert (3x3)  -2  0  2
//                               -1  0  1
//
//
//                                1  2  1
//              SobelHoriz (3x3)  0  0  0
//                               -1 -2 -1
//
//
//                                       1 -2  1
//              SobelVertSecond (3x3)    2 -4  2
//                                       1 -2  1
//
//
//                                       1  2  1
//              SobelHorizSecond (3x3)  -2 -4 -2
//                                       1  2  1
//
//                               -1  -2   0   2   1
//                               -4  -8   0   8   4
//              SobelVert (5x5)  -6 -12   0  12   6
//                               -4  -8   0   8   4
//                               -1  -2   0   2   1
//
//                                1   4   6   4   1
//                                2   8  12   8   2
//              SobelHoriz (5x5)  0   0   0   0   0
//                               -2  -8 -12  -8  -4
//                               -1  -4  -6  -4  -1
//
//                                       1   0  -2   0   1
//                                       4   0  -8   0   4
//              SobelVertSecond (5x5)    6   0 -12   0   6
//                                       4   0  -8   0   4
//                                       1   0  -2   0   1
//
//                                       1   4   6   4   1
//                                       0   0   0   0   0
//              SobelHorizSecond (5x5)  -2  -8 -12  -8  -2
//                                       0   0   0   0   0
//                                       1   4   6   4   1

// Auxiliary parameters structure
typedef struct _IwiFilterSobelParams
{
    int reserved;
} IwiFilterSobelParams;

// Sets auxiliary parameters to default values
static IW_INLINE void iwiFilterSobel_SetDefaultParams(
    IwiFilterSobelParams *pParams      // [in,out] Pointer to the auxiliary parameters structure
)
{
    (void)pParams;
}

// Applies Sobel filter of specific type to the source image
// Features support:
//      Data types:              8u16s,16s,32f
//      Channel types:           C1
//      Inplace mode:            no
//      64-bit sizes:            no
//      Internal threading:      no
//      Manual tiling:           yes
//      IwiTile simple tiling:   yes
//      IwiTile pipeline tiling: yes
// Returns:
//      ippStsMaskSizeErr                   mask value is illegal
//      ippStsDataTypeErr                   data type is illegal
//      ippStsNumChannelsErr                channels value is illegal
//      ippStsNotEvenStepErr                step value is not divisible by size of elements
//      ippStsBorderErr                     border value is illegal
//      ippStsNotSupportedModeErr           selected function mode is not supported
//      ippStsNoMemErr                      failed to allocate memory
//      ippStsSizeErr                       size fields values are illegal
//      ippStsInplaceModeNotSupportedErr    doesn't support output into the source buffer
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL(IppStatus) iwiFilterSobel(
    const IwiImage             *pSrcImage,  // [in]     Pointer to the source image
    IwiImage                   *pDstImage,  // [in,out] Pointer to the destination image
    IwiDerivativeType           opType,     // [in]     Type of derivative from IwiDerivativeType
    IppiMaskSize                kernelSize, // [in]     Size of the filter kernel: ippMskSize3x3, ippMskSize5x5
    const IwiFilterSobelParams *pAuxParams, // [in]     Pointer to the auxiliary parameters structure. If NULL - default parameters will be used
    IwiBorderType               border,     // [in]     Extrapolation algorithm for out of image pixels: ippBorderConst, ippBorderRepl, ippBorderMirror, ippBorderInMem
    const Ipp64f               *pBorderVal, // [in]     Pointer to array of border values for ippBorderConst. One element for each channel. Can be NULL for any other border
    const IwiTile              *pTile       // [in]     Pointer to the IwiTile structure for tiling. If NULL - the whole image will be processed in accordance to size
);

/**/////////////////////////////////////////////////////////////////////////////
//                   iwiFilterScharr
///////////////////////////////////////////////////////////////////////////// */
//
//                                3  0  -3
//              ScharrVert       10  0 -10
//                                3  0  -3
//
//
//                                3  10  3
//              ScharrHoriz       0   0  0
//                               -3 -10 -3

// Auxiliary parameters structure
typedef struct _IwiFilterScharrParams
{
    int reserved;
} IwiFilterScharrParams;

// Sets auxiliary parameters to default values
static IW_INLINE void iwiFilterScharr_SetDefaultParams(
    IwiFilterScharrParams *pParams      // [in,out] Pointer to the auxiliary parameters structure
)
{
    (void)pParams;
}

// Applies Scharr filter of specific type to the source image
// Features support:
//      Inplace mode:            no
//      64-bit sizes:            no
//      Internal threading:      no
//      Manual tiling:           yes
//      IwiTile simple tiling:   yes
//      IwiTile pipeline tiling: yes
// Returns:
//      ippStsMaskSizeErr                   mask value is illegal
//      ippStsDataTypeErr                   data type is illegal
//      ippStsNumChannelsErr                channels value is illegal
//      ippStsNotEvenStepErr                step value is not divisible by size of elements
//      ippStsBorderErr                     border value is illegal
//      ippStsNotSupportedModeErr           selected function mode is not supported
//      ippStsNoMemErr                      failed to allocate memory
//      ippStsSizeErr                       size fields values are illegal
//      ippStsInplaceModeNotSupportedErr    doesn't support output into the source buffer
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL(IppStatus) iwiFilterScharr(
    const IwiImage              *pSrcImage,  // [in]     Pointer to the source image
    IwiImage                    *pDstImage,  // [in,out] Pointer to the destination image
    IwiDerivativeType            opType,     // [in]     Type of derivative from IwiDerivativeType
    IppiMaskSize                 kernelSize, // [in]     Size of the filter kernel: ippMskSize3x3, ippMskSize5x5
    const IwiFilterScharrParams *pAuxParams, // [in]     Pointer to the auxiliary parameters structure. If NULL - default parameters will be used
    IwiBorderType                border,     // [in]     Extrapolation algorithm for out of image pixels: ippBorderConst, ippBorderRepl, ippBorderMirror, ippBorderInMem
    const Ipp64f                *pBorderVal, // [in]     Pointer to array of border values for ippBorderConst. One element for each channel. Can be NULL for any other border
    const IwiTile               *pTile       // [in]     Pointer to the IwiTile structure for tiling. If NULL - the whole image will be processed in accordance to size
);

/**/////////////////////////////////////////////////////////////////////////////
//                   iwiFilterLaplacian
///////////////////////////////////////////////////////////////////////////// */
//
//                                  2  0  2
//              Laplacian (3x3)     0 -8  0
//                                  2  0  2
//
//                                2   4   4   4   2
//                                4   0  -8   0   4
//              Laplacian (5x5)   4  -8 -24  -8   4
//                                4   0  -8   0   4
//                                2   4   4   4   2

// Auxiliary parameters structure
typedef struct _IwiFilterLaplacianParams
{
    int reserved;
} IwiFilterLaplacianParams;

// Sets auxiliary parameters to default values
static IW_INLINE void iwiFilterLaplacian_SetDefaultParams(
    IwiFilterLaplacianParams *pParams      // [in,out] Pointer to the auxiliary parameters structure
)
{
    (void)pParams;
}

// Applies Laplacian filter to the source image
// Features support:
//      Inplace mode:            no
//      64-bit sizes:            no
//      Internal threading:      no
//      Manual tiling:           yes
//      IwiTile simple tiling:   yes
//      IwiTile pipeline tiling: yes
// Returns:
//      ippStsMaskSizeErr                   mask value is illegal
//      ippStsDataTypeErr                   data type is illegal
//      ippStsNumChannelsErr                channels value is illegal
//      ippStsNotEvenStepErr                step value is not divisible by size of elements
//      ippStsBorderErr                     border value is illegal
//      ippStsNotSupportedModeErr           selected function mode is not supported
//      ippStsNoMemErr                      failed to allocate memory
//      ippStsSizeErr                       size fields values are illegal
//      ippStsInplaceModeNotSupportedErr    doesn't support output into the source buffer
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL(IppStatus) iwiFilterLaplacian(
    const IwiImage                 *pSrcImage,  // [in]     Pointer to the source image
    IwiImage                       *pDstImage,  // [in,out] Pointer to the destination image
    IppiMaskSize                    kernelSize, // [in]     Size of the filter kernel: ippMskSize3x3, ippMskSize5x5
    const IwiFilterLaplacianParams *pAuxParams, // [in]     Pointer to the auxiliary parameters structure. If NULL - default parameters will be used
    IwiBorderType                   border,     // [in]     Extrapolation algorithm for out of image pixels: ippBorderConst, ippBorderRepl, ippBorderMirror, ippBorderInMem
    const Ipp64f                   *pBorderVal, // [in]     Pointer to array of border values for ippBorderConst. One element for each channel. Can be NULL for any other border
    const IwiTile                  *pTile       // [in]     Pointer to the IwiTile structure for tiling. If NULL - the whole image will be processed in accordance to size
);

/* /////////////////////////////////////////////////////////////////////////////
//                   iwiFilterGaussian
///////////////////////////////////////////////////////////////////////////// */

// Internal structure for data sharing between function calls
typedef struct _IwiFilterGaussianSpec IwiFilterGaussianSpec;

// Auxiliary parameters structure
typedef struct _IwiFilterGaussianParams
{
    IwiChDescriptor chDesc; // Special channels processing mode
} IwiFilterGaussianParams;

// Sets auxiliary parameters to default values
static IW_INLINE void iwiFilterGaussian_SetDefaultParams(
    IwiFilterGaussianParams *pParams      // [in,out] Pointer to the auxiliary parameters structure
)
{
    if(pParams)
    {
        pParams->chDesc = iwiChDesc_None;
    }
}

// Applies Gaussian filter to the source image
// Features support:
//      Inplace mode:            no
//      64-bit sizes:            yes
//      Internal threading:      no
//      Manual tiling:           yes
//      IwiTile simple tiling:   yes
//      IwiTile pipeline tiling: yes
// Returns:
//      ippStsMaskSizeErr                   mask value is illegal
//      ippStsDataTypeErr                   data type is illegal
//      ippStsNumChannelsErr                channels value is illegal
//      ippStsNotEvenStepErr                step value is not divisible by size of elements
//      ippStsBorderErr                     border value is illegal
//      ippStsNotSupportedModeErr           selected function mode is not supported
//      ippStsNoMemErr                      failed to allocate memory
//      ippStsSizeErr                       size fields values are illegal
//      ippStsInplaceModeNotSupportedErr    doesn't support output into the source buffer
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL(IppStatus) iwiFilterGaussian(
    const IwiImage                 *pSrcImage,  // [in]     Pointer to the source image
    IwiImage                       *pDstImage,  // [in,out] Pointer to the destination image
    int                             kernelSize, // [in]     Size of the Gaussian kernel (odd, greater or equal to 3)
    double                          sigma,      // [in]     Standard deviation of the Gaussian kernel
    const IwiFilterGaussianParams  *pAuxParams, // [in]     Pointer to the auxiliary parameters structure. If NULL - default parameters will be used
    IwiBorderType                   border,     // [in]     Extrapolation algorithm for out of image pixels: ippBorderConst, ippBorderRepl, ippBorderMirror, ippBorderInMem
    const Ipp64f                   *pBorderVal, // [in]     Pointer to array of border values for ippBorderConst. One element for each channel. Can be NULL for any other border
    const IwiTile                  *pTile       // [in]     Pointer to the IwiTile structure for tiling. If NULL - the whole image will be processed in accordance to size
);

// Free internal data structure
IW_DECL(IppStatus) iwiFilterGaussian_Free(
    IwiFilterGaussianSpec *pSpec        // [in]  Pointer to internal spec structure
);

// Allocates and initializes internal data structure
// Returns:
//      ippStsMaskSizeErr                   mask value is illegal
//      ippStsDataTypeErr                   data type is illegal
//      ippStsNumChannelsErr                channels value is illegal
//      ippStsNotEvenStepErr                step value is not divisible by size of elements
//      ippStsBorderErr                     border value is illegal
//      ippStsNotSupportedModeErr           selected function mode is not supported
//      ippStsNoMemErr                      failed to allocate memory
//      ippStsSizeErr                       size fields values are illegal
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL(IppStatus) iwiFilterGaussian_InitAlloc(
    IwiFilterGaussianSpec         **pSpec,          // [out]    Pointer to pointer to internal spec structure. Structure will be allocated here.
    IwiSize                         size,           // [in]     Size of the image in pixels
    IppDataType                     dataType,       // [in]     Image pixel type
    int                             channels,       // [in]     Number of image channels
    int                             kernelSize,     // [in]     Size of the Gaussian kernel (odd, greater or equal to 3)
    double                          sigma,          // [in]     Standard deviation of the Gaussian kernel
    const IwiFilterGaussianParams  *pAuxParams,     // [in]     Pointer to the auxiliary parameters structure. If NULL - default parameters will be used
    IwiBorderType                   border          // [in]     Extrapolation algorithm for out of image pixels: ippBorderConst, ippBorderRepl, ippBorderMirror, ippBorderInMem
);

// Applies Gaussian filter to the source image
// Features support:
//      Inplace mode:            no
//      64-bit sizes:            yes
//      Internal threading:      no
//      Manual tiling:           yes
//      IwiTile simple tiling:   yes
//      IwiTile pipeline tiling: yes
// Returns:
//      ippStsMaskSizeErr                   mask value is illegal
//      ippStsDataTypeErr                   data type is illegal
//      ippStsNumChannelsErr                channels value is illegal
//      ippStsNotEvenStepErr                step value is not divisible by size of elements
//      ippStsBorderErr                     border value is illegal
//      ippStsContextMatchErr               internal structure is not initialized or of invalid type
//      ippStsNoMemErr                      failed to allocate memory
//      ippStsSizeErr                       size fields values are illegal
//      ippStsInplaceModeNotSupportedErr    doesn't support output into the source buffer
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL(IppStatus) iwiFilterGaussian_Process(
    const IwiFilterGaussianSpec *pSpec,      // [in]     Pointer to internal spec structure
    const IwiImage              *pSrcImage,  // [in]     Pointer to the source image
    IwiImage                    *pDstImage,  // [in,out] Pointer to the destination image
    IwiBorderType                border,     // [in]     Extrapolation algorithm for out of image pixels: ippBorderConst, ippBorderRepl, ippBorderInMem
    const Ipp64f                *pBorderVal, // [in]     Pointer to array of border values for ippBorderConst. One element for each channel. Can be NULL for any other border
    const IwiTile               *pTile       // [in]     Pointer to the IwiTile structure for tiling. If NULL - the whole image will be processed in accordance to size
);

/**/////////////////////////////////////////////////////////////////////////////
//                   iwiFilterCanny
///////////////////////////////////////////////////////////////////////////// */

// Auxiliary parameters structure
typedef struct _IwiFilterCannyParams
{
    IppiDifferentialKernel  kernel;     // Type of differential kernel: ippFilterSobel, ippFilterScharr
    IppiMaskSize            kernelSize; // Size of the filter kernel: ippFilterSobel: ippMskSize3x3, ippMskSize5x5; ippFilterScharr: ippMskSize3x3
    IppNormType             norm;       // Normalization mode: ippNormL1, ippNormL2
} IwiFilterCannyParams;

// Sets auxiliary parameters to default values
static IW_INLINE void iwiFilterCanny_SetDefaultParams(
    IwiFilterCannyParams *pParams      // [in,out] Pointer to the auxiliary parameters structure
)
{
    if(pParams)
    {
        pParams->kernel     = ippFilterSobel;
        pParams->kernelSize = ippMskSize3x3;
        pParams->norm       = ippNormL2;
    }
}

// Applies Canny edge detector to the source image
// Features support:
//      Inplace mode:            no
//      64-bit sizes:            no
//      Internal threading:      no
//      Manual tiling:           no
//      IwiTile simple tiling:   no
//      IwiTile pipeline tiling: no
// Returns:
//      ippStsMaskSizeErr                   mask value is illegal
//      ippStsDataTypeErr                   data type is illegal
//      ippStsNumChannelsErr                channels value is illegal
//      ippStsNotEvenStepErr                step value is not divisible by size of elements
//      ippStsBorderErr                     border value is illegal
//      ippStsNoMemErr                      failed to allocate memory
//      ippStsSizeErr                       size fields values are illegal
//      ippStsInplaceModeNotSupportedErr    doesn't support output into the source buffer
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL(IppStatus) iwiFilterCanny(
    const IwiImage             *pSrcImage,  // [in]     Pointer to the source image
    IwiImage                   *pDstImage,  // [in,out] Pointer to the destination image
    Ipp32f                      treshLow,   // [in]     Lower threshold for edges detection
    Ipp32f                      treshHigh,  // [in]     Upper threshold for edges detection
    const IwiFilterCannyParams *pAuxParams, // [in]     Pointer to the auxiliary parameters structure. If NULL - default parameters will be used
    IwiBorderType               border,     // [in]     Extrapolation algorithm for out of image pixels: ippBorderConst, ippBorderRepl, ippBorderMirror, ippBorderInMem
    const Ipp64f               *pBorderVal  // [in]     Pointer to array of border values for ippBorderConst. One element for each channel. Can be NULL for any other border
);

// Auxiliary parameters structure
typedef struct _IwiFilterCannyDerivParams
{
    IppNormType norm; // Normalization mode: ippNormL1, ippNormL2
} IwiFilterCannyDerivParams;

// Sets auxiliary parameters to default values
static IW_INLINE void iwiFilterCannyDeriv_SetDefaultParams(
    IwiFilterCannyDerivParams *pParams      // [in,out] Pointer to the auxiliary parameters structure
)
{
    if(pParams)
    {
        pParams->norm = ippNormL2;
    }
}

// Applies Canny edge detector to the image derivatives
// Features support:
//      Inplace mode:            no
//      64-bit sizes:            no
//      Internal threading:      no
//      Manual tiling:           no
//      IwiTile simple tiling:   no
//      IwiTile pipeline tiling: no
// Returns:
//      ippStsDataTypeErr                   data type is illegal
//      ippStsNumChannelsErr                channels value is illegal
//      ippStsNotEvenStepErr                step value is not divisible by size of elements
//      ippStsNoMemErr                      failed to allocate memory
//      ippStsSizeErr                       size fields values are illegal
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL(IppStatus) iwiFilterCannyDeriv(
    const IwiImage                  *pSrcImageDx,    // [in]     Pointer to X derivative of the source image
    const IwiImage                  *pSrcImageDy,    // [in]     Pointer to Y derivative of the source image
    IwiImage                        *pDstImage,      // [in,out] Pointer to the destination image
    Ipp32f                           treshLow,       // [in]     Lower threshold for edges detection
    Ipp32f                           treshHigh,      // [in]     Upper threshold for edges detection
    const IwiFilterCannyDerivParams *pAuxParams      // [in]     Pointer to the auxiliary parameters structure. If NULL - default parameters will be used
);

/**/////////////////////////////////////////////////////////////////////////////
//                   iwiFilterMorphology
///////////////////////////////////////////////////////////////////////////// */

// Morphology operator type enumerator
typedef enum _IwiMorphologyType
{
    iwiMorphErode  = 0, // Erode morphology operation
    iwiMorphDilate,     // Dilate morphology operation
    iwiMorphOpen,       // Open morphology operation
    iwiMorphClose,      // Close morphology operation
    iwiMorphTophat,     // Top-hat morphology operation
    iwiMorphBlackhat,   // Black-hat morphology operation
    iwiMorphGradient    // Gradient morphology operation
} IwiMorphologyType;

// Auxiliary parameters structure
typedef struct _IwiFilterMorphologyParams
{
    int iterations; // Number of repeated morphology operations to apply
} IwiFilterMorphologyParams;

// Sets auxiliary parameters to default values
static IW_INLINE void iwiFilterMorphology_SetDefaultParams(
    IwiFilterMorphologyParams *pParams      // [in,out] Pointer to the auxiliary parameters structure
)
{
    if(pParams)
    {
        pParams->iterations = 1;
    }
}

// Performs morphology filter operation
// Features support:
//      Inplace mode:            no
//      64-bit sizes:            yes
//      Internal threading:      no
//      Manual tiling:           yes
//      IwiTile simple tiling:   yes
//      IwiTile pipeline tiling: yes
// Returns:
//      ippStsBorderErr                     border value is illegal
//      ippStsDataTypeErr                   data type is illegal
//      ippStsNumChannelsErr                channels value is illegal
//      ippStsNotEvenStepErr                step value is not divisible by size of elements
//      ippStsNotSupportedModeErr           selected function mode is not supported
//      ippStsNoMemErr                      failed to allocate memory
//      ippStsInplaceModeNotSupportedErr    doesn't support output into the source buffer
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL(IppStatus) iwiFilterMorphology(
    const IwiImage                  *pSrcImage,     // [in]     Pointer to the source image
    IwiImage                        *pDstImage,     // [in,out] Pointer to the destination image
    IwiMorphologyType                morphType,     // [in]     Morphology filter type
    const IwiImage                  *pMaskImage,    // [in]     Pointer to morphology mask image. Mask must be continuous, 8-bit, 1 channel image
    const IwiFilterMorphologyParams *pAuxParams,    // [in]     Pointer to the auxiliary parameters structure. If NULL - default parameters will be used
    IwiBorderType                    border,        // [in]     Extrapolation algorithm for out of image pixels: ippBorderConst, ippBorderRepl, ippBorderMirror, ippBorderInMem
    const Ipp64f                    *pBorderVal,    // [in]     Pointer to array of border values for ippBorderConst. One element for each channel. Can be NULL for any other border
    const IwiTile                   *pTile          // [in]     Pointer to the IwiTile structure for tiling. If NULL - the whole image will be processed in accordance to size
);

// Calculates border size for morphology operation
// Returns:
//      ippStsNotSupportedModeErr           selected function mode is not supported
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL(IppStatus) iwiFilterMorphology_GetBorderSize(
    IwiMorphologyType   morphType,      // [in]  Morphology filter type
    IwiSize             maskSize,       // [in]  Size of morphology mask
    IwiBorderSize      *pBorderSize     // [out] Pointer to border size structure
);

/**/////////////////////////////////////////////////////////////////////////////
//                   iwiFilterBilateral
///////////////////////////////////////////////////////////////////////////// */

// Auxiliary parameters structure
typedef struct _IwiFilterBilateralParams
{
    IppiFilterBilateralType filter;         // Type of bilateral filter: ippiFilterBilateralGauss
    IppiDistanceMethodType  distMethod;     // Method for definition of distance between pixel intensity: ippDistNormL1
} IwiFilterBilateralParams;

// Sets auxiliary parameters to default values
static IW_INLINE void iwiFilterBilateral_SetDefaultParams(
    IwiFilterBilateralParams *pParams      // [in,out] Pointer to the auxiliary parameters structure
)
{
    if(pParams)
    {
        pParams->filter     = ippiFilterBilateralGauss;
        pParams->distMethod = ippDistNormL1;
    }
}

// Performs bilateral filtering of an image
// Features support:
//      Inplace mode:            no
//      64-bit sizes:            yes
//      Internal threading:      yes (check IW_ENABLE_THREADING_LAYER definition)
//      Manual tiling:           yes
//      IwiTile simple tiling:   yes
//      IwiTile pipeline tiling: yes
// Returns:
//      ippStsSizeErr                       size fields values are illegal
//      ippStsBadArgErr                     valSquareSigma or posSquareSigma is less or equal 0
//      ippStsMaskSizeErr                   radius is less or equal 0
//      ippStsBorderErr                     border value is illegal
//      ippStsDataTypeErr                   data type is illegal
//      ippStsNumChannelsErr                channels value is illegal
//      ippStsNotEvenStepErr                step value is not divisible by size of elements
//      ippStsNotSupportedModeErr           filter or distMethod is not supported
//      ippStsNoMemErr                      failed to allocate memory
//      ippStsInplaceModeNotSupportedErr    doesn't support output into the source buffer
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL(IppStatus) iwiFilterBilateral(
    const IwiImage                 *pSrcImage,      // [in]     Pointer to the source image
    IwiImage                       *pDstImage,      // [in,out] Pointer to the destination image
    int                             radius,         // [in]     Radius of circular neighborhood what defines pixels for calculation
    Ipp32f                          valSquareSigma, // [in]     Square of Sigma for factor function for pixel intensity
    Ipp32f                          posSquareSigma, // [in]     Square of Sigma for factor function for pixel position
    const IwiFilterBilateralParams *pAuxParams,     // [in]     Pointer to the auxiliary parameters structure. If NULL - default parameters will be used
    IwiBorderType                   border,         // [in]     Extrapolation algorithm for out of image pixels: ippBorderConst, ippBorderRepl, ippBorderMirror, ippBorderInMem
    const Ipp64f                   *pBorderVal,     // [in]     Pointer to array of border values for ippBorderConst. One element for each channel. Can be NULL for any other border
    const IwiTile                  *pTile           // [in]     Pointer to the IwiTile structure for tiling. If NULL - the whole image will be processed in accordance to size
);

#ifdef __cplusplus
}
#endif

#endif
