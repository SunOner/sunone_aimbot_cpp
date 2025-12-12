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

#if !defined( __IPP_IW_IMAGE_TRANSFORM__ )
#define __IPP_IW_IMAGE_TRANSFORM__

#include "iw/iw_image.h"

#ifdef __cplusplus
extern "C" {
#endif

/**/////////////////////////////////////////////////////////////////////////////
//                   iwiMirror
///////////////////////////////////////////////////////////////////////////// */

// Auxiliary parameters structure
typedef struct _IwiMirrorParams
{
    IwiChDescriptor chDesc;     // Special channels processing mode
} IwiMirrorParams;

// Sets auxiliary parameters to default values
static IW_INLINE void iwiMirror_SetDefaultParams(
    IwiMirrorParams *pParams      // [in,out] Pointer to the auxiliary parameters structure
)
{
    if(pParams)
    {
        pParams->chDesc = iwiChDesc_None;
    }
}

// Mirrors image around specified axis.
// For ippAxs45 and ippAxs135 destination image must have flipped size: dstWidth = srcHeight, dstHeight = srcWidth
// Features support:
//      Inplace mode:            no
//      64-bit sizes:            no
//      Internal threading:      no
//      Manual tiling:           yes
//      IwiTile simple tiling:   yes
//      IwiTile pipeline tiling: no
// Returns:
//      ippStsDataTypeErr                   data type is illegal
//      ippStsNumChannelsErr                channels value is illegal
//      ippStsNotSupportedModeErr           selected function mode is not supported
//      ippStsInplaceModeNotSupportedErr    doesn't support output into the source buffer. In tiling mode
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL(IppStatus) iwiMirror(
    const IwiImage         *pSrcImage,  // [in]     Pointer to the source image
    IwiImage               *pDstImage,  // [in,out] Pointer to the destination image
    IppiAxis                axis,       // [in]     Mirror axis
    const IwiMirrorParams  *pAuxParams, // [in]     Pointer to the auxiliary parameters structure. If NULL - default parameters will be used
    const IwiTile          *pTile       // [in]     Pointer to the IwiTile structure for tiling. If NULL - the whole image will be processed in accordance to size
);

// Calculates destination image size for iwiMirror function.
// Returns:
//      ippStsBadArgErr                     incorrect arg/param of the function
//      ippStsNoErr                         no errors
IW_DECL(IppStatus) iwiMirror_GetDstSize(
    IwiSize         srcSize,    // [in]     Size of the source image in pixels
    IppiAxis        axis,       // [in]     Angle of clockwise rotation in degrees
    IwiSize        *pDstSize    // [in,out] Destination size for the mirrored image
);

// Calculates source ROI by destination one
// Returns:
//      ippStsNotSupportedModeErr           selected function mode is not supported
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL(IppStatus) iwiMirror_GetSrcRoi(
    IppiAxis        axis,       // [in]  Mirror axis
    IwiSize         dstSize,    // [in]  Size of destination image in pixels
    IwiRoi          dstRoi,     // [in]  Destination image ROI
    IwiRoi         *pSrcRoi     // [out] Pointer to source image ROI
);

/**/////////////////////////////////////////////////////////////////////////////
//                   iwiRotate
///////////////////////////////////////////////////////////////////////////// */

// Auxiliary parameters structure
typedef struct _IwiRotateParams
{
    int reserved;
} IwiRotateParams;

// Sets auxiliary parameters to default values
static IW_INLINE void iwiRotate_SetDefaultParams(
    IwiRotateParams *pParams      // [in,out] Pointer to the auxiliary parameters structure
)
{
    (void)pParams;
}

// Performs rotation of the image around (0,0).
// This is a more simple version of iwiWarpAffine function designed specifically for rotation.
// Features support:
//      Inplace mode:            no
//      64-bit sizes:            no
//      Internal threading:      no
//      Manual tiling:           no
//      IwiTile simple tiling:   yes
//      IwiTile pipeline tiling: no
// Returns:
//      ippStsInterpolationErr              interpolation value is illegal
//      ippStsDataTypeErr                   data type is illegal
//      ippStsNumChannelsErr                channels value is illegal
//      ippStsBorderErr                     border value is illegal
//      ippStsNotSupportedModeErr           selected function mode is not supported
//      ippStsNoMemErr                      failed to allocate memory
//      ippStsSizeErr                       size fields values are illegal
//      ippStsInplaceModeNotSupportedErr    doesn't support output into the source buffer
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
//      ippStsNoOperation                   [warning] width or height of an image is zero
//      ippStsWrongIntersectQuad            [warning] transformed source image has no intersection with the destination image
IW_DECL(IppStatus) iwiRotate(
    const IwiImage         *pSrcImage,      // [in]     Pointer to the source image
    IwiImage               *pDstImage,      // [in,out] Pointer to the destination image
    double                  angle,          // [in]     Angle of clockwise rotation in degrees
    IppiInterpolationType   interpolation,  // [in]     Interpolation method: ippNearest, ippLinear, ippCubic
    const IwiRotateParams  *pAuxParams,     // [in]     Pointer to the auxiliary parameters structure. If NULL - default parameters will be used
    IwiBorderType           border,         // [in]     Extrapolation algorithm for out of image pixels: ippBorderConst, ippBorderRepl, ippBorderTransp, ippBorderInMem
    const Ipp64f           *pBorderVal,     // [in]     Pointer to array of border values for ippBorderConst. One element for each channel. Can be NULL for any other border
    const IwiTile          *pTile           // [in]     Pointer to the IwiTile structure for tiling. If NULL - the whole image will be processed in accordance to size
);

// Calculates destination image size for iwiRotate function.
// Returns:
//      ippStsBadArgErr                     incorrect arg/param of the function
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL(IppStatus) iwiRotate_GetDstSize(
    IwiSize         srcSize,    // [in]     Size of the source image in pixels
    double          angle,      // [in]     Angle of clockwise rotation in degrees
    IwiSize        *pDstSize    // [in,out] Size of rotated image boundaries
);

/**/////////////////////////////////////////////////////////////////////////////
//                   iwiResize
///////////////////////////////////////////////////////////////////////////// */

// Internal structure for data sharing between function calls
typedef struct _IwiResizeSpec IwiResizeSpec;

// Auxiliary parameters structure
typedef struct _IwiResizeParams
{
    Ipp32f cubicBVal;       // The first parameter for cubic filters
    Ipp32f cubicCVal;       // The second parameter for cubic filters
    Ipp32u lanczosLobes;    // Parameter for Lanczos filters. Possible values are 2 or 3
    Ipp32u antialiasing;    // Use resize with anti-aliasing if possible. Use this to reduce the image size with minimization of moire artifacts
} IwiResizeParams;

// Sets auxiliary parameters to default values
static IW_INLINE void iwiResize_SetDefaultParams(
    IwiResizeParams *pParams      // [in,out] Pointer to the auxiliary parameters structure
)
{
    if(pParams)
    {
        pParams->antialiasing = 0;
        pParams->cubicBVal    = 1;
        pParams->cubicCVal    = 0;
        pParams->lanczosLobes = 3;
    }
}

// Performs resize operation on the given image ROI
// Features support:
//      Data types:              8u, 16u, 16s, 32f
//      Channel types:           C1, C3, C4
//      Pre-initialization:      yes
//      Inplace mode:            no
//      64-bit sizes:            yes
//      Internal threading:      yes (check IW_ENABLE_THREADING_LAYER definition)
//      Manual tiling:           no
//      IwiTile simple tiling:   yes
//      IwiTile pipeline tiling: no
// Returns:
//      ippStsInterpolationErr              interpolation value is illegal
//      ippStsSizeErr                       size fields values are illegal
//      ippStsBorderErr                     border value is illegal
//      ippStsDataTypeErr                   data type is illegal
//      ippStsNumChannelsErr                channels value is illegal
//      ippStsNotEvenStepErr                step value is not divisible by size of elements
//      ippStsNotSupportedModeErr           selected function mode is not supported
//      ippStsNoMemErr                      failed to allocate memory
//      ippStsInplaceModeNotSupportedErr    doesn't support output into the source buffer
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
//      ippStsNoOperation                   [warning] width or height of an image is zero
IW_DECL(IppStatus) iwiResize(
    const IwiImage         *pSrcImage,      // [in]     Pointer to the source image
    IwiImage               *pDstImage,      // [in,out] Pointer to the destination image
    IppiInterpolationType   interpolation,  // [in]     Interpolation method: ippNearest, ippLinear, ippCubic, ippLanczos, ippSuper
    const IwiResizeParams  *pAuxParams,     // [in]     Pointer to the auxiliary parameters structure. If NULL - default parameters will be used
    IwiBorderType           border,         // [in]     Extrapolation algorithm for out of image pixels: ippBorderConst, ippBorderRepl, ippBorderInMem
    const Ipp64f           *pBorderVal,     // [in]     Pointer to array of border values for ippBorderConst. One element for each channel. Can be NULL for any other border
    const IwiTile          *pTile           // [in]     Pointer to the IwiTile structure for tiling. If NULL - the whole image will be processed in accordance to size
);

// Free internal data structure
IW_DECL(IppStatus) iwiResize_Free(
    IwiResizeSpec *pSpec        // [in]  Pointer to internal spec structure
);

// Allocates and initializes internal data structure
// Returns:
//      ippStsInterpolationErr              interpolation value is illegal
//      ippStsDataTypeErr                   data type is illegal
//      ippStsNumChannelsErr                channels value is illegal
//      ippStsBorderErr                     border value is illegal
//      ippStsNotSupportedModeErr           selected function mode is not supported
//      ippStsNoMemErr                      failed to allocate memory
//      ippStsSizeErr                       size fields values are illegal
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
//      ippStsNoOperation                   [warning] width or height of an image is zero
IW_DECL(IppStatus) iwiResize_InitAlloc(
    IwiResizeSpec         **pSpec,          // [out] Pointer to pointer to internal spec structure. Structure will be allocated here.
    IwiSize                 srcSize,        // [in]  Size of the source image in pixels
    IwiSize                 dstSize,        // [in]  Size of the destination image in pixels
    IppDataType             dataType,       // [in]  Image pixel type
    int                     channels,       // [in]  Number of image channels
    IppiInterpolationType   interpolation,  // [in]  Interpolation method: ippNearest, ippLinear, ippCubic, ippLanczos, ippSuper
    const IwiResizeParams  *pAuxParams,     // [in]  Pointer to the auxiliary parameters structure. If NULL - default parameters will be used
    IwiBorderType           border          // [in]  Extrapolation algorithm for out of image pixels: ippBorderConst, ippBorderRepl, ippBorderInMem
);

// Performs resize operation on given image ROI
// Features support:
//      Inplace mode            no
//      64-bit sizes            yes
//      Internal threading:     yes (check IW_ENABLE_THREADING_LAYER definition)
//      Manual tiling:          no
//      IwiRoi simple tiling:   yes
//      IwiRoi pipeline tiling: no
// Returns:
//      ippStsBorderErr                     border value is illegal
//      ippStsDataTypeErr                   data type is illegal
//      ippStsNumChannelsErr                channels value is illegal
//      ippStsNotEvenStepErr                step value is not divisible by size of elements
//      ippStsNotSupportedModeErr           selected function mode is not supported
//      ippStsNoMemErr                      failed to allocate memory
//      ippStsInplaceModeNotSupportedErr    doesn't support output into the source buffer
//      ippStsContextMatchErr               internal structure is not initialized or of invalid type
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
//      ippStsNoOperation                   [warning] width or height of an image is zero
IW_DECL(IppStatus) iwiResize_Process(
    const IwiResizeSpec    *pSpec,      // [in]     Pointer to internal spec structure
    const IwiImage         *pSrcImage,  // [in]     Pointer to the source image
    IwiImage               *pDstImage,  // [in,out] Pointer to the destination image
    IwiBorderType           border,     // [in]     Extrapolation algorithm for out of image pixels: ippBorderConst, ippBorderRepl, ippBorderInMem
    const Ipp64f           *pBorderVal, // [in]     Pointer to array of border values for ippBorderConst. One element for each channel. Can be NULL for any other border
    const IwiTile          *pTile       // [in]     Pointer to the IwiTile structure for tiling. If NULL - the whole image will be processed in accordance to size
);

// Calculates source ROI by destination one
// Returns:
//      ippStsContextMatchErr               internal structure is not initialized or of invalid type
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL(IppStatus) iwiResize_GetSrcRoi(
    const IwiResizeSpec *pSpec,         // [in]  Pointer to internal spec structure
    IwiRoi               dstRoi,        // [in]  Destination image ROI
    IwiRoi              *pSrcRoi        // [out] Pointer to source image ROI
);

// Get border size for resize
// Returns:
//      ippStsContextMatchErr               internal structure is not initialized or of invalid type
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL(IppStatus) iwiResize_GetBorderSize(
    const IwiResizeSpec *pSpec,         // [in]  Pointer to internal spec structure
    IwiBorderSize       *pBorderSize    // [out] Pointer to border size structure
);

/**/////////////////////////////////////////////////////////////////////////////
//                   iwiWarpAffine
///////////////////////////////////////////////////////////////////////////// */

// Internal structure for data sharing between function calls
typedef struct _IwiWarpAffineSpec IwiWarpAffineSpec;

// Auxiliary parameters structure
typedef struct _IwiWarpAffineParams
{
    Ipp32f cubicBVal;       // The first parameter for cubic filters
    Ipp32f cubicCVal;       // The second parameter for cubic filters
    Ipp32u smoothEdge;      // Edges smooth post-processing. Only for ippBorderTransp, ippBorderInMem
} IwiWarpAffineParams;

// Sets auxiliary parameters to default values
static IW_INLINE void iwiWarpAffine_SetDefaultParams(
    IwiWarpAffineParams *pParams    // [in,out] Pointer to the auxiliary parameters structure
)
{
    if(pParams)
    {
        pParams->smoothEdge = 0;
        pParams->cubicBVal  = 1;
        pParams->cubicCVal  = 0;
    }
}

// Simplified version of warp affine function without spec structure and initialization
// Features support:
//      Inplace mode:            no
//      64-bit sizes:            no
//      Internal threading:      no
//      Manual tiling:           no
//      IwiTile simple tiling:   yes
//      IwiTile pipeline tiling: no
// Returns:
//      ippStsWarpDirectionErr              direction value is illegal
//      ippStsCoeffErr                      affine transformation is singular
//      ippStsInterpolationErr              interpolation value is illegal
//      ippStsDataTypeErr                   data type is illegal
//      ippStsNumChannelsErr                channels value is illegal
//      ippStsBorderErr                     border value is illegal
//      ippStsNotEvenStepErr                step value is not divisible by size of elements
//      ippStsNotSupportedModeErr           selected function mode is not supported
//      ippStsNoMemErr                      failed to allocate memory
//      ippStsSizeErr                       size fields values are illegal
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
//      ippStsNoOperation                   [warning] width or height of an image is zero
//      ippStsWrongIntersectQuad            [warning] transformed source image has no intersection with the destination image
IW_DECL(IppStatus) iwiWarpAffine(
    const IwiImage             *pSrcImage,      // [in]     Pointer to the source image
    IwiImage                   *pDstImage,      // [in,out] Pointer to the destination image
    const double                coeffs[2][3],   // [in]     Coefficients for the affine transform
    IwTransDirection            direction,      // [in]     Transformation direction
    IppiInterpolationType       interpolation,  // [in]     Interpolation method: ippNearest, ippLinear, ippCubic
    const IwiWarpAffineParams  *pAuxParams,     // [in]     Pointer to the auxiliary parameters structure. If NULL - default parameters will be used
    IwiBorderType               border,         // [in]     Extrapolation algorithm for out of image pixels: ippBorderConst, ippBorderRepl, ippBorderTransp, ippBorderInMem
    const Ipp64f               *pBorderVal,     // [in]     Pointer to array of border values for ippBorderConst. One element for each channel. Can be NULL for any other border
    const IwiTile              *pTile           // [in]     Pointer to the IwiTile structure for tiling. If NULL - the whole image will be processed in accordance to size
);

// Releases internal data structure
IW_DECL(IppStatus) iwiWarpAffine_Free(
    IwiWarpAffineSpec *pSpec        // [in]  Pointer to internal spec structure
);

// Allocates and initializes internal data structure
// Returns:
//      ippStsWarpDirectionErr              direction value is illegal
//      ippStsCoeffErr                      affine transformation is singular
//      ippStsInterpolationErr              interpolation value is illegal
//      ippStsDataTypeErr                   data type is illegal
//      ippStsNumChannelsErr                channels value is illegal
//      ippStsBorderErr                     border value is illegal
//      ippStsNotSupportedModeErr           selected function mode is not supported
//      ippStsNoMemErr                      failed to allocate memory
//      ippStsSizeErr                       size fields values are illegal
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
//      ippStsNoOperation                   [warning] width or height of an image is zero
//      ippStsWrongIntersectQuad            [warning] transformed source image has no intersection with the destination image
IW_DECL(IppStatus) iwiWarpAffine_InitAlloc(
    IwiWarpAffineSpec         **pSpec,          // [out] Pointer to pointer to internal spec structure. Structure will be allocated here.
    IwiSize                     srcSize,        // [in]  Size of the source image in pixels
    IwiSize                     dstSize,        // [in]  Size of the destination image in pixels
    IppDataType                 dataType,       // [in]  Image pixel type
    int                         channels,       // [in]  Number of image channels
    const double                coeffs[2][3],   // [in]  Coefficients for the affine transform
    IwTransDirection            direction,      // [in]  Transformation direction
    IppiInterpolationType       interpolation,  // [in]  Interpolation method: ippNearest, ippLinear, ippCubic
    const IwiWarpAffineParams  *pAuxParams,     // [in]  Pointer to the auxiliary parameters structure. If NULL - default parameters will be used
    IwiBorderType               border,         // [in]  Extrapolation algorithm for out of image pixels: ippBorderConst, ippBorderRepl, ippBorderTransp, ippBorderInMem
    const Ipp64f               *pBorderVal      // [in]  Pointer to array of border values for ippBorderConst. One element for each channel. Can be NULL for any other border
);

// Performs warp affine transform of the given image ROI
// Features support:
//      Inplace mode:            no
//      64-bit sizes:            no
//      Internal threading:      no
//      Manual tiling:           no
//      IwiTile simple tiling:   yes
//      IwiTile pipeline tiling: no
// Returns:
//      ippStsInterpolationErr              interpolation value is illegal
//      ippStsDataTypeErr                   data type is illegal
//      ippStsNumChannelsErr                channels value is illegal
//      ippStsBorderErr                     border value is illegal
//      ippStsNotEvenStepErr                step value is not divisible by size of elements
//      ippStsNotSupportedModeErr           selected function mode is not supported
//      ippStsNoMemErr                      failed to allocate memory
//      ippStsSizeErr                       size fields values are illegal
//      ippStsContextMatchErr               internal structure is not initialized or of invalid type
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
//      ippStsNoOperation                   [warning] width or height of an image is zero
//      ippStsWrongIntersectQuad            [warning] transformed source image has no intersection with the destination image
IW_DECL(IppStatus) iwiWarpAffine_Process(
    const IwiWarpAffineSpec    *pSpec,      // [in]     Pointer to internal spec structure
    const IwiImage             *pSrcImage,  // [in]     Pointer to the source image
    IwiImage                   *pDstImage,  // [in,out] Pointer to the destination image
    const IwiTile              *pTile       // [in]     Pointer to the IwiTile structure for tiling. If NULL - the whole image will be processed in accordance to size
);


#ifdef __cplusplus
}
#endif

#endif
