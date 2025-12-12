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

#if !defined( __IPP_IW_IMAGE_OP__ )
#define __IPP_IW_IMAGE_OP__

#include "iw/iw_image.h"

#ifdef __cplusplus
extern "C" {
#endif

/**/////////////////////////////////////////////////////////////////////////////
//                   iwiCopy
///////////////////////////////////////////////////////////////////////////// */

// Auxiliary parameters structure
typedef struct _IwiCopyParams
{
    int reserved;
} IwiCopyParams;

// Sets auxiliary parameters to default values
static IW_INLINE void iwiCopy_SetDefaultParams(
    IwiCopyParams *pParams      // [in,out] Pointer to the auxiliary parameters structure
)
{
    (void)pParams;
}

// Copies image data to destination image with masking.
// If mask is NULL, then calls iwiCopy function.
// For masked operation, the function writes pixel values in the destination buffer only if the spatially corresponding
// mask array value is non-zero.
// Features support:
//      Inplace mode:            inapplicable
//      64-bit sizes:            no
//      Internal threading:      no
//      Manual tiling:           yes
//      IwiTile simple tiling:   yes
//      IwiTile pipeline tiling: yes
// Returns:
//      ippStsNotSupportedModeErr           selected function mode is not supported
//      ippStsSizeErr                       size fields values are illegal
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL(IppStatus) iwiCopy(
    const IwiImage      *pSrcImage,  // [in]     Pointer to the source image
    IwiImage            *pDstImage,  // [in,out] Pointer to the destination image
    const IwiImage      *pMaskImage, // [in]     Pointer to mask image. Mask must be 8-bit, 1 channel image. If NULL - no masking will be performed
    const IwiCopyParams *pAuxParams, // [in]     Pointer to the auxiliary parameters structure. If NULL - default parameters will be used
    const IwiTile       *pTile       // [in]     Pointer to the IwiTile structure for tiling. If NULL - the whole image will be processed in accordance to size
);

/**/////////////////////////////////////////////////////////////////////////////
//                   iwiCopyChannel
///////////////////////////////////////////////////////////////////////////// */

// Auxiliary parameters structure
typedef struct _IwiCopyChannelParams
{
    int reserved;
} IwiCopyChannelParams;

// Sets auxiliary parameters to default values
static IW_INLINE void iwiCopyChannel_SetDefaultParams(
    IwiCopyChannelParams *pParams      // [in,out] Pointer to the auxiliary parameters structure
)
{
    (void)pParams;
}

// Copies selected channel from one image to another.
// Features support:
//      Inplace mode:            yes
//      64-bit sizes:            no
//      Internal threading:      no
//      Manual tiling:           yes
//      IwiTile simple tiling:   yes
//      IwiTile pipeline tiling: yes
// Returns:
//      ippStsDataTypeErr                   data type is illegal
//      ippStsNumChannelsErr                channels value is illegal
//      ippStsSizeErr                       size fields values are illegal
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL(IppStatus) iwiCopyChannel(
    const IwiImage             *pSrcImage,  // [in]     Pointer to the source image
    int                         srcChannel, // [in]     Source channel to copy from (starting from 0)
    IwiImage                   *pDstImage,  // [in,out] Pointer to the destination image
    int                         dstChannel, // [in]     Destination channel to copy to (starting from 0)
    const IwiCopyChannelParams *pAuxParams, // [in]     Pointer to the auxiliary parameters structure. If NULL - default parameters will be used
    const IwiTile              *pTile       // [in]     Pointer to the IwiTile structure for tiling. If NULL - the whole image will be processed in accordance to size
);

/**/////////////////////////////////////////////////////////////////////////////
//                   iwiSplitChannels
///////////////////////////////////////////////////////////////////////////// */

// Auxiliary parameters structure
typedef struct _IwiSplitChannelsParams
{
    int reserved;
} IwiSplitChannelsParams;

// Sets auxiliary parameters to default values
static IW_INLINE void iwiSplitChannels_SetDefaultParams(
    IwiSplitChannelsParams *pParams      // [in,out] Pointer to the auxiliary parameters structure
)
{
    (void)pParams;
}

// Splits multi-channel image into array of single channel images.
// Features support:
//      Inplace mode:            no
//      64-bit sizes:            no
//      Internal threading:      no
//      Manual tiling:           yes
//      IwiTile simple tiling:   yes
//      IwiTile pipeline tiling: yes
// Returns:
//      ippStsDataTypeErr                   data type is illegal
//      ippStsNumChannelsErr                channels value is illegal
//      ippStsSizeErr                       size fields values are illegal
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL(IppStatus) iwiSplitChannels(
    const IwiImage               *pSrcImage,    // [in]     Pointer to the source image
    IwiImage* const               pDstImages[], // [in,out] Array of pointers to destination images. Size of this array must not be less than number of channels in the source image
                                                //          NULL pointers will be skipped.
    const IwiSplitChannelsParams *pAuxParams,   // [in]     Pointer to the auxiliary parameters structure. If NULL - default parameters will be used
    const IwiTile                *pTile         // [in]     Pointer to the IwiTile structure for tiling. If NULL - the whole image will be processed in accordance to size
);

/**/////////////////////////////////////////////////////////////////////////////
//                   iwiMergeChannels
///////////////////////////////////////////////////////////////////////////// */

// Auxiliary parameters structure
typedef struct _IwiMergeChannelsParams
{
    int reserved;
} IwiMergeChannelsParams;

// Sets auxiliary parameters to default values
static IW_INLINE void iwiMergeChannels_SetDefaultParams(
    IwiMergeChannelsParams *pParams      // [in,out] Pointer to the auxiliary parameters structure
)
{
    (void)pParams;
}

// Merges array of single channel images into one multi-channel image.
// Features support:
//      Inplace mode:            no
//      64-bit sizes:            no
//      Internal threading:      no
//      Manual tiling:           yes
//      IwiTile simple tiling:   yes
//      IwiTile pipeline tiling: yes
// Returns:
//      ippStsDataTypeErr                   data type is illegal
//      ippStsNumChannelsErr                channels value is illegal
//      ippStsSizeErr                       size fields values are illegal
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL(IppStatus) iwiMergeChannels(
    const IwiImage* const         pSrcImages[], // [in]     Array of pointers to source images. Size of this array must not be less than number of channels in the destination image
                                                //          NULL pointers will be skipped.
    IwiImage                     *pDstImage,    // [in,out] Pointer to the destination image
    const IwiMergeChannelsParams *pAuxParams,   // [in]     Pointer to the auxiliary parameters structure. If NULL - default parameters will be used
    const IwiTile                *pTile         // [in]     Pointer to the IwiTile structure for tiling. If NULL - the whole image will be processed in accordance to size
);

/**/////////////////////////////////////////////////////////////////////////////
//                   iwiCreateBorder
///////////////////////////////////////////////////////////////////////////// */

// Auxiliary parameters structure
typedef struct _IwiCreateBorderParams
{
    int reserved;
} IwiCreateBorderParams;

// Sets auxiliary parameters to default values
static IW_INLINE void iwiCreateBorder_SetDefaultParams(
    IwiCreateBorderParams *pParams      // [in,out] Pointer to the auxiliary parameters structure
)
{
    (void)pParams;
}

// Copies image data to destination image and constructs border of specified size.
// Destination image must have enough memory for a border according to inMemSize member.
// If border is specified with InMem flags then image will be extended by border size to include InMem pixels,
// but will not build border in this direction.
// Features support:
//      Inplace mode:            yes
//      64-bit sizes:            yes
//      Internal threading:      no
//      Manual tiling:           yes
//      IwiTile simple tiling:   yes
//      IwiTile pipeline tiling: yes
// Returns:
//      ippStsNotSupportedModeErr           selected function mode is not supported
//      ippStsSizeErr                       1) size fields values are illegal
//                                          2) border.top or border.left are greater than corresponding inMemSize values of destination image
//                                          3) dst_width  + dst_inMemSize.right  < min_width  + border.right
//                                          4) dst_height + dst_inMemSize.bottom < min_height + border.bottom
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL(IppStatus) iwiCreateBorder(
    const IwiImage                 *pSrcImage,      // [in]     Pointer to the source image
    IwiImage                       *pDstImage,      // [in,out] Pointer to the destination image which points to actual data start
    IwiBorderSize                   borderSize,     // [in]     Size of border to reconstruct. Destination image must have greater or equal inMemSize values
    IwiBorderType                   border,         // [in]     Extrapolation algorithm for out of image pixels: ippBorderConst, ippBorderRepl, ippBorderMirror, ippBorderWrap
    const Ipp64f                   *pBorderVal,     // [in]     Pointer to array of border values for ippBorderConst. One element for each channel. Can be NULL for any other border
    const IwiCreateBorderParams    *pAuxParams,     // [in]     Pointer to the auxiliary parameters structure. If NULL - default parameters will be used
    const IwiTile                  *pTile           // [in]     Pointer to the IwiTile structure for tiling. If NULL - the whole image will be processed in accordance to size
);

/**/////////////////////////////////////////////////////////////////////////////
//                   iwiSet
///////////////////////////////////////////////////////////////////////////// */

// Auxiliary parameters structure
typedef struct _IwiSetParams
{
    int reserved;
} IwiSetParams;

// Sets auxiliary parameters to default values
static IW_INLINE void iwiSet_SetDefaultParams(
    IwiSetParams *pParams      // [in,out] Pointer to the auxiliary parameters structure
)
{
    (void)pParams;
}

// Sets each channel of the image to the specified values with masking.
// If mask is NULL, then calls iwiSet function.
// For masked operation, the function writes pixel values in the destination buffer only if the spatially corresponding
// mask array value is non-zero.
// Features support:
//      Inplace mode:            inapplicable
//      64-bit sizes:            no
//      Internal threading:      no
//      Manual tiling:           yes
//      IwiTile simple tiling:   yes
//      IwiTile pipeline tiling: yes
// Returns:
//      ippStsDataTypeErr                   data type is illegal
//      ippStsNumChannelsErr                channels value is illegal
//      ippStsSizeErr                       size fields values are illegal
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL(IppStatus) iwiSet(
    const double       *pValues,        // [in]     Pointer to the array of values
    int                 valuesNum,      // [in]     Number of value elements.
                                        //          If valuesSize > number of channels then exceeding values will be ignored.
                                        //          If valuesSize < number of channels then the last value will be replicated for the remaining channels
    IwiImage           *pDstImage,      // [in,out] Pointer to the destination image
    const IwiImage     *pMaskImage,     // [in]     Pointer to the mask image. Mask must be 8-bit, 1 channel image. If NULL - no masking will be performed
    const IwiSetParams *pAuxParams,     // [in]     Pointer to the auxiliary parameters structure. If NULL - default parameters will be used
    const IwiTile      *pTile           // [in]     Pointer to the IwiTile structure for tiling. If NULL - the whole image will be processed in accordance to size
);

/**/////////////////////////////////////////////////////////////////////////////
//                   iwiSetChannel
///////////////////////////////////////////////////////////////////////////// */

// Auxiliary parameters structure
typedef struct _IwiSetChannelParams
{
    int reserved;
} IwiSetChannelParams;

// Sets auxiliary parameters to default values
static IW_INLINE void iwiSetChannel_SetDefaultParams(
    IwiSetChannelParams *pParams      // [in,out] Pointer to the auxiliary parameters structure
)
{
    (void)pParams;
}

// Sets selected channel of the multi-channel image to the specified value.
// Features support:
//      Inplace mode:            inapplicable
//      64-bit sizes:            no
//      Internal threading:      no
//      Manual tiling:           yes
//      IwiTile simple tiling:   yes
//      IwiTile pipeline tiling: yes
// Returns:
//      ippStsDataTypeErr                   data type is illegal
//      ippStsNumChannelsErr                channels value is illegal
//      ippStsSizeErr                       size fields values are illegal
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL(IppStatus) iwiSetChannel(
    double                     value,      // [in]     Value for a selected channel
    IwiImage                  *pDstImage,  // [in,out] Pointer to the destination image
    int                        channelNum, // [in]     Number of channel to be set (starting from 0)
    const IwiSetChannelParams *pAuxParams, // [in]     Pointer to the auxiliary parameters structure. If NULL - default parameters will be used
    const IwiTile             *pTile       // [in]     Pointer to the IwiTile structure for tiling. If NULL - the whole image will be processed in accordance to size
);

/**/////////////////////////////////////////////////////////////////////////////
//                   iwiAdd
///////////////////////////////////////////////////////////////////////////// */

// Auxiliary parameters structure
typedef struct _IwiAddParams
{
    int             scaleFactor;    // Scale factor
    IwiChDescriptor chDesc;         // Special channels processing mode
} IwiAddParams;

// Sets auxiliary parameters to default values
static IW_INLINE void iwiAdd_SetDefaultParams(
    IwiAddParams *pParams      // [in,out] Pointer to the auxiliary parameters structure
)
{
    if(pParams)
    {
        pParams->scaleFactor = 0;
        pParams->chDesc      = iwiChDesc_None;
    }
}

// Performs addition of one image to another and writes result to the output
// Features support:
//      Inplace mode:            yes (second addend and sum)
//      64-bit sizes:            yes
//      Internal threading:      no
//      Manual tiling:           yes
//      IwiTile simple tiling:   yes
//      IwiTile pipeline tiling: yes
// Returns:
//      ippStsDataTypeErr                   data type is illegal
//      ippStsNumChannelsErr                channels value is illegal
//      ippStsSizeErr                       size fields values are illegal
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL(IppStatus) iwiAdd(
    const IwiImage     *pAddend1,   // [in]     Pointer to the first addend image
    const IwiImage     *pAddend2,   // [in]     Pointer to the second addend image
    IwiImage           *pDstImage,  // [in,out] Pointer to the result image (can be same as the second addend)
    const IwiAddParams *pAuxParams, // [in]     Pointer to the auxiliary parameters structure. If NULL - default parameters will be used
    const IwiTile      *pTile       // [in]     Pointer to the IwiTile structure for tiling. If NULL - the whole image will be processed in accordance to size
);

/**/////////////////////////////////////////////////////////////////////////////
//                   iwiAddC
///////////////////////////////////////////////////////////////////////////// */

// Auxiliary parameters structure
typedef struct _IwiAddCParams
{
    int             scaleFactor;    // Scale factor
    IwiChDescriptor chDesc;         // Special channels processing mode
} IwiAddCParams;

// Sets auxiliary parameters to default values
static IW_INLINE void iwiAddC_SetDefaultParams(
    IwiAddCParams *pParams      // [in,out] Pointer to the auxiliary parameters structure
)
{
    if(pParams)
    {
        pParams->scaleFactor = 0;
        pParams->chDesc      = iwiChDesc_None;
    }
}

// Performs addition of the constant to the image another and writes result to the output
// Features support:
//      Inplace mode:            yes (addend image and sum)
//      64-bit sizes:            yes
//      Internal threading:      no
//      Manual tiling:           yes
//      IwiTile simple tiling:   yes
//      IwiTile pipeline tiling: yes
// Returns:
//      ippStsDataTypeErr                   data type is illegal
//      ippStsNumChannelsErr                channels value is illegal
//      ippStsSizeErr                       size fields values are illegal
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL(IppStatus) iwiAddC(
    const double        *pAddend,       // [in]     Pointer to the array of addends. One element for each channel
    int                  addendsNum,    // [in]     Number of addends.
                                        //          If addendsNum > number of channels then exceeding values will be ignored.
                                        //          If addendsNum < number of channels then the last value will be replicated for the remaining channels
    const IwiImage      *pAddendImage,  // [in]     Pointer to the addend image
    IwiImage            *pDstImage,     // [in,out] Pointer to the difference image (can be same as addend image)
    const IwiAddCParams *pAuxParams,    // [in]     Pointer to the auxiliary parameters structure. If NULL - default parameters will be used
    const IwiTile       *pTile          // [in]     Pointer to the IwiTile structure for tiling. If NULL - the whole image will be processed in accordance to size
);

/**/////////////////////////////////////////////////////////////////////////////
//                   iwiSub
///////////////////////////////////////////////////////////////////////////// */

// Auxiliary parameters structure
typedef struct _IwiSubParams
{
    int             scaleFactor;    // Scale factor
    IwiChDescriptor chDesc;         // Special channels processing mode
} IwiSubParams;

// Sets auxiliary parameters to default values
static IW_INLINE void iwiSub_SetDefaultParams(
    IwiSubParams *pParams      // [in,out] Pointer to the auxiliary parameters structure
)
{
    if(pParams)
    {
        pParams->scaleFactor = 0;
        pParams->chDesc      = iwiChDesc_None;
    }
}

// Performs subtraction of the first image from the second image and writes result to the output
// Features support:
//      Inplace mode:            yes (minuend and difference)
//      64-bit sizes:            yes
//      Internal threading:      no
//      Manual tiling:           yes
//      IwiTile simple tiling:   yes
//      IwiTile pipeline tiling: yes
// Returns:
//      ippStsDataTypeErr                   data type is illegal
//      ippStsNumChannelsErr                channels value is illegal
//      ippStsSizeErr                       size fields values are illegal
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL(IppStatus) iwiSub(
    const IwiImage      *pSubtrahend,   // [in]     Pointer to the subtrahend image
    const IwiImage      *pMinuend,      // [in]     Pointer to the minuend image
    IwiImage            *pDstImage,     // [in,out] Pointer to the difference image (can be same as minuend)
    const IwiSubParams  *pAuxParams,    // [in]     Pointer to the auxiliary parameters structure. If NULL - default parameters will be used
    const IwiTile       *pTile          // [in]     Pointer to the IwiTile structure for tiling. If NULL - the whole image will be processed in accordance to size
);

/**/////////////////////////////////////////////////////////////////////////////
//                   iwiSubC
///////////////////////////////////////////////////////////////////////////// */

// Auxiliary parameters structure
typedef struct _IwiSubCParams
{
    int             scaleFactor;    // Scale factor
    IwiChDescriptor chDesc;         // Special channels processing mode
} IwiSubCParams;

// Sets auxiliary parameters to default values
static IW_INLINE void iwiSubC_SetDefaultParams(
    IwiSubCParams *pParams      // [in,out] Pointer to the auxiliary parameters structure
)
{
    if(pParams)
    {
        pParams->scaleFactor = 0;
        pParams->chDesc      = iwiChDesc_None;
    }
}

// Performs subtraction of the constant from the image and writes result to the output
// Features support:
//      Inplace mode:            yes (minuend and difference)
//      64-bit sizes:            yes
//      Internal threading:      no
//      Manual tiling:           yes
//      IwiTile simple tiling:   yes
//      IwiTile pipeline tiling: yes
// Returns:
//      ippStsDataTypeErr                   data type is illegal
//      ippStsNumChannelsErr                channels value is illegal
//      ippStsSizeErr                       size fields values are illegal
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL(IppStatus) iwiSubC(
    const double        *pSubtrahend,       // [in]     Pointer to the array of subtrahends. One element for each channel
    int                  subtrahendsNum,    // [in]     Number of subtrahends.
                                            //          If subtrahendsNum > number of channels then exceeding values will be ignored.
                                            //          If subtrahendsNum < number of channels then the last value will be replicated for the remaining channels
    const IwiImage      *pMinuendImage,     // [in]     Pointer to the minuend image
    IwiImage            *pDstImage,         // [in,out] Pointer to the difference image (can be same as minuend)
    const IwiSubCParams *pAuxParams,        // [in]     Pointer to the auxiliary parameters structure. If NULL - default parameters will be used
    const IwiTile       *pTile              // [in]     Pointer to the IwiTile structure for tiling. If NULL - the whole image will be processed in accordance to size
);

/**/////////////////////////////////////////////////////////////////////////////
//                   iwiMul
///////////////////////////////////////////////////////////////////////////// */

// Auxiliary parameters structure
typedef struct _IwiMulParams
{
    IppHintAlgorithm algoMode;          // Accuracy mode
    int              scaleFactor;       // Scale factor
    IwiChDescriptor  chDesc;            // Special channels processing mode
} IwiMulParams;

// Sets auxiliary parameters to default values
static IW_INLINE void iwiMul_SetDefaultParams(
    IwiMulParams *pParams      // [in,out] Pointer to the auxiliary parameters structure
)
{
    if(pParams)
    {
        pParams->algoMode           = ippAlgHintNone;
        pParams->scaleFactor        = 0;
        pParams->chDesc             = iwiChDesc_None;
    }
}

// Performs multiplication of one image by another and writes result to the output
// Features support:
//      Inplace mode:            yes (second factor and product)
//      64-bit sizes:            yes
//      Internal threading:      no
//      Manual tiling:           yes
//      IwiTile simple tiling:   yes
//      IwiTile pipeline tiling: yes
// Returns:
//      ippStsDataTypeErr                   data type is illegal
//      ippStsNumChannelsErr                channels value is illegal
//      ippStsSizeErr                       size fields values are illegal
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL(IppStatus) iwiMul(
    const IwiImage     *pFactor1,   // [in]     Pointer to the first factor image
    const IwiImage     *pFactor2,   // [in]     Pointer to the second factor image
    IwiImage           *pDstImage,  // [in,out] Pointer to the product image (can be same as the second factor)
    const IwiMulParams *pAuxParams, // [in]     Pointer to the auxiliary parameters structure. If NULL - default parameters will be used
    const IwiTile      *pTile       // [in]     Pointer to the IwiTile structure for tiling. If NULL - the whole image will be processed in accordance to size
);
/**/////////////////////////////////////////////////////////////////////////////
//                   iwiMulC
///////////////////////////////////////////////////////////////////////////// */

// Auxiliary parameters structure
typedef struct _IwiMulCParams
{
    IppRoundMode     roundMode;     // Rounding mode
    IppHintAlgorithm algoMode;      // Accuracy mode
    int              scaleFactor;   // Scale factor
    IwiChDescriptor  chDesc;        // Special channels processing mode
} IwiMulCParams;

// Sets auxiliary parameters to default values
static IW_INLINE void iwiMulC_SetDefaultParams(
    IwiMulCParams *pParams      // [in,out] Pointer to the auxiliary parameters structure
)
{
    if(pParams)
    {
        pParams->roundMode   = ippRndNear;
        pParams->algoMode    = ippAlgHintNone;
        pParams->scaleFactor = 0;
        pParams->chDesc      = iwiChDesc_None;
    }
}

// Performs multiplication of one image by constant and writes result to the output
// Features support:
//      Inplace mode:            yes (factor image and product)
//      64-bit sizes:            no
//      Internal threading:      no
//      Manual tiling:           yes
//      IwiTile simple tiling:   yes
//      IwiTile pipeline tiling: yes
// Returns:
//      ippStsDataTypeErr                   data type is illegal
//      ippStsNumChannelsErr                channels value is illegal
//      ippStsSizeErr                       size fields values are illegal
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL(IppStatus) iwiMulC(
    const double        *pFactor,       // [in]     Pointer to the array of factors. One element for each channel
    int                  factorsNum,    // [in]     Number of factors.
                                        //          If factorsNum > number of channels then exceeding values will be ignored.
                                        //          If factorsNum < number of channels then the last value will be replicated for the remaining channels
    const IwiImage      *pFactorImage,  // [in]     Pointer to the factor image
    IwiImage            *pDstImage,     // [in,out] Pointer to the product image (can be same as factor image)
    const IwiMulCParams *pAuxParams,    // [in]     Pointer to the auxiliary parameters structure. If NULL - default parameters will be used
    const IwiTile       *pTile          // [in]     Pointer to the IwiTile structure for tiling. If NULL - the whole image will be processed in accordance to size
);

/**/////////////////////////////////////////////////////////////////////////////
//                   iwiDiv
///////////////////////////////////////////////////////////////////////////// */

// Auxiliary parameters structure
typedef struct _IwiDivParams
{
    IppHintAlgorithm algoMode;          // Accuracy mode
    int              scaleFactor;       // Scale factor
    IwiChDescriptor  chDesc;            // Special channels processing mode
} IwiDivParams;

// Sets auxiliary parameters to default values
static IW_INLINE void iwiDiv_SetDefaultParams(
    IwiDivParams *pParams      // [in,out] Pointer to the auxiliary parameters structure
)
{
    if(pParams)
    {
        pParams->algoMode       = ippAlgHintNone;
        pParams->scaleFactor    = 0;
        pParams->chDesc         = iwiChDesc_None;
    }
}

// Performs division of the second image by the first image and writes result to the output
// Features support:
//      Inplace mode:            yes (numerator and fraction)
//      64-bit sizes:            yes
//      Internal threading:      no
//      Manual tiling:           yes
//      IwiTile simple tiling:   yes
//      IwiTile pipeline tiling: yes
// Returns:
//      ippStsDataTypeErr                   data type is illegal
//      ippStsNumChannelsErr                channels value is illegal
//      ippStsSizeErr                       size fields values are illegal
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL(IppStatus) iwiDiv(
    const IwiImage     *pDenominator,   // [in]     Pointer to the denominator image
    const IwiImage     *pNumerator,     // [in]     Pointer to the numerator image
    IwiImage           *pDstImage,      // [in,out] Pointer to the fraction image (can be same as numerator)
    const IwiDivParams *pAuxParams,     // [in]     Pointer to the auxiliary parameters structure. If NULL - default parameters will be used
    const IwiTile      *pTile           // [in]     Pointer to the IwiTile structure for tiling. If NULL - the whole image will be processed in accordance to size
);

/**/////////////////////////////////////////////////////////////////////////////
//                   iwiDivC
///////////////////////////////////////////////////////////////////////////// */

// Auxiliary parameters structure
typedef struct _IwiDivCParams
{
    IppRoundMode     roundMode;     // Rounding mode
    IppHintAlgorithm algoMode;      // Accuracy mode
    int              scaleFactor;   // Scale factor
    IwiChDescriptor  chDesc;        // Special channels processing mode
} IwiDivCParams;

// Sets auxiliary parameters to default values
static IW_INLINE void iwiDivC_SetDefaultParams(
    IwiDivCParams *pParams      // [in,out] Pointer to the auxiliary parameters structure
)
{
    if(pParams)
    {
        pParams->roundMode   = ippRndNear;
        pParams->algoMode    = ippAlgHintNone;
        pParams->scaleFactor = 0;
        pParams->chDesc      = iwiChDesc_None;
    }
}

// Performs division of the second image by constant and writes result to the output
// Features support:
//      Inplace mode:            yes (numerator and fraction)
//      64-bit sizes:            no
//      Internal threading:      no
//      Manual tiling:           yes
//      IwiTile simple tiling:   yes
//      IwiTile pipeline tiling: yes
// Returns:
//      ippStsDataTypeErr                   data type is illegal
//      ippStsNumChannelsErr                channels value is illegal
//      ippStsSizeErr                       size fields values are illegal
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL(IppStatus) iwiDivC(
    const double        *pDenominator,      // [in]     Pointer to the array of denominators. One element for each channel
    int                  denominatorsNum,   // [in]     Number of denominators.
                                            //          If denominatorsNum > number of channels then exceeding values will be ignored.
                                            //          If denominatorsNum < number of channels then the last value will be replicated for the remaining channels
    const IwiImage      *pNumeratorImage,   // [in]     Pointer to the numerator image
    IwiImage            *pDstImage,         // [in,out] Pointer to the fraction image (can be same as numerator)
    const IwiDivCParams *pAuxParams,        // [in]     Pointer to the auxiliary parameters structure. If NULL - default parameters will be used
    const IwiTile       *pTile              // [in]     Pointer to the IwiTile structure for tiling. If NULL - the whole image will be processed in accordance to size
);

/**/////////////////////////////////////////////////////////////////////////////
//                   iwiSwapChannels
///////////////////////////////////////////////////////////////////////////// */

// Auxiliary parameters structure
typedef struct _IwiSwapChannelsParams
{
    IwiChDescriptor chDesc;     // Special channels processing mode
} IwiSwapChannelsParams;

// Sets auxiliary parameters to default values
static IW_INLINE void iwiSwapChannels_SetDefaultParams(
    IwiSwapChannelsParams *pParams      // [in,out] Pointer to the auxiliary parameters structure
)
{
    if(pParams)
    {
        pParams->chDesc = iwiChDesc_None;
    }
}

// Swaps image channels according to the destination order parameter.
// One source channel can be mapped to several destination channels.
// Special order rules:
// 1) if(dstOrder[channel] == srcChannels) dst[channel] = constValue
// 2) if(dstOrder[channel] < 0 || dstOrder[channel] > srcChannels) dst[channel] is unchanged
// Features support:
//      Inplace mode:            yes (8u only)
//      64-bit sizes:            no
//      Internal threading:      no
//      Manual tiling:           yes
//      IwiTile simple tiling:   yes
//      IwiTile pipeline tiling: yes
// Returns:
//      ippStsChannelOrderErr               destination order is out of the range
//      ippStsDataTypeErr                   data type is illegal
//      ippStsNumChannelsErr                channels value is illegal
//      ippStsSizeErr                       size fields values are illegal
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL(IppStatus) iwiSwapChannels(
    const IwiImage              *pSrcImage,  // [in]     Pointer to the source image
    IwiImage                    *pDstImage,  // [in,out] Pointer to the destination image
    const int                   *pDstOrder,  // [in]     Pointer to the destination image channels order: dst[channel] = src[dstOrder[channel]]
    double                       value,      // [in]     Value to fill the destination channel if number of destination channels is bigger than number of source channels
    const IwiSwapChannelsParams *pAuxParams, // [in]     Pointer to the auxiliary parameters structure. If NULL - default parameters will be used
    const IwiTile               *pTile       // [in]     Pointer to the IwiTile structure for tiling. If NULL - the whole image will be processed in accordance to size
);

/**/////////////////////////////////////////////////////////////////////////////
//                   iwiScale
///////////////////////////////////////////////////////////////////////////// */

// Auxiliary parameters structure
typedef struct _IwiScaleParams
{
    IppHintAlgorithm algoMode;   // Accuracy mode
} IwiScaleParams;

// Sets auxiliary parameters to default values
static IW_INLINE void iwiScale_SetDefaultParams(
    IwiScaleParams *pParams      // [in,out] Pointer to the auxiliary parameters structure
)
{
    if(pParams)
    {
        pParams->algoMode = ippAlgHintNone;
    }
}

// Converts image from one data type to another with specified scaling and shifting
// DST = saturate(SRC*mulVal + addVal)
// Features support:
//      Inplace mode:            yes (for images of same type)
//      64-bit sizes:            no
//      Internal threading:      no
//      Manual tiling:           yes
//      IwiTile simple tiling:   yes
//      IwiTile pipeline tiling: yes
// Returns:
//      ippStsDataTypeErr                   data type is illegal
//      ippStsSizeErr                       size fields values are illegal
//      ippStsInplaceModeNotSupportedErr    doesn't support output into the source buffer. If data types are different
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL(IppStatus) iwiScale(
    const IwiImage         *pSrcImage,  // [in]     Pointer to the source image
    IwiImage               *pDstImage,  // [in,out] Pointer to the destination image
    Ipp64f                  mulVal,     // [in]     Multiplier
    Ipp64f                  addVal,     // [in]     Addend
    const IwiScaleParams   *pAuxParams, // [in]     Pointer to the auxiliary parameters structure. If NULL - default parameters will be used
    const IwiTile          *pTile       // [in]     Pointer to the IwiTile structure for tiling. If NULL - the whole image will be processed in accordance to size
);

// Returns multiplication and addend values for iwiScale function to perform accurate data range scaling between two data types
// Data range for float values is considered to be from 0 to 1
// Returns:
//      ippStsDataTypeErr                   data type is illegal
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL(IppStatus) iwiScale_GetScaleVals(
    IppDataType srcType,    // [in]     Source data type
    IppDataType dstType,    // [in]     Destination data type
    Ipp64f     *pMulVal,    // [out]    Pointer to multiplier
    Ipp64f     *pAddVal     // [out]    Pointer to addend
);

#ifdef __cplusplus
}
#endif

#endif
