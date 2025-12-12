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

#if !defined( __IPP_IWPP_IMAGE_OP__ )
#define __IPP_IWPP_IMAGE_OP__

#include "iw/iw_image_op.h"
#include "iw++/iw_image.hpp"

namespace ipp
{

/**/////////////////////////////////////////////////////////////////////////////
//                   iwiCopy
///////////////////////////////////////////////////////////////////////////// */

// Auxiliary parameters class
// C API descriptions has more details
class IwiCopyParams: public ::IwiCopyParams
{
public:
    IW_BASE_PARAMS_CONSTRUCTORS(IwiCopyParams, iwiCopy_SetDefaultParams)
    IwiCopyParams() {}
};

// Copies image data to destination image with masking.
// If mask is NULL, then calls iwiCopy function.
// For masked operation, the function writes pixel values in the destination buffer only if the spatially corresponding
// mask array value is non-zero.
// C API descriptions has more details.
// Throws:
//      ippStsNotSupportedModeErr           selected function mode is not supported
//      ippStsSizeErr                       size fields values are illegal
//      ippStsInplaceModeNotSupportedErr    doesn't support output into the source buffer
//      ippStsNullPtrErr                    unexpected NULL pointer
// Returns:
//      ippStsNoErr                         no errors
IW_DECL_CPP(IppStatus) iwiCopy(
    const IwiImage      &srcImage,                      // [in]     Reference to the source image
    IwiImage            &dstImage,                      // [in,out] Reference to the destination image
    const IwiImage      &maskImage   = IwiImage(),      // [in]     Reference to the mask image. Mask must be 8-bit, 1 channel image
    const IwiCopyParams &auxParams   = IwiCopyParams(), // [in]     Reference to the auxiliary parameters structure
    const IwiTile       &tile        = IwiTile()        // [in,out] Reference to the IwiTile object for tiling. By default no tiling is used
)
{
    IppStatus ippStatus = ::iwiCopy(&srcImage, &dstImage, &maskImage, &auxParams, &tile);
    OWN_ERROR_CHECK(ippStatus)
    return ippStatus;
}

/**/////////////////////////////////////////////////////////////////////////////
//                   iwiCopyChannel
///////////////////////////////////////////////////////////////////////////// */

// Auxiliary parameters class
// C API descriptions has more details
class IwiCopyChannelParams: public ::IwiCopyChannelParams
{
public:
    IW_BASE_PARAMS_CONSTRUCTORS(IwiCopyChannelParams, iwiCopyChannel_SetDefaultParams)
    IwiCopyChannelParams() {}
};

// Copies selected channel from one image to another.
// C API descriptions has more details.
// Throws:
//      ippStsDataTypeErr                   data type is illegal
//      ippStsNumChannelsErr                channels value is illegal
//      ippStsSizeErr                       size fields values are illegal
//      ippStsInplaceModeNotSupportedErr    doesn't support output into the source buffer
//      ippStsNullPtrErr                    unexpected NULL pointer
// Returns:
//      ippStsNoErr                         no errors
IW_DECL_CPP(IppStatus) iwiCopyChannel(
    const IwiImage             &srcImage,                               // [in]     Reference to the source image
    int                         srcChannel,                             // [in]     Source channel to copy from (starting from 0)
    IwiImage                   &dstImage,                               // [in,out] Reference to the destination image
    int                         dstChannel,                             // [in]     Destination channel to copy to (starting from 0)
    const IwiCopyChannelParams &auxParams   = IwiCopyChannelParams(),   // [in]     Reference to the auxiliary parameters structure
    const IwiTile              &tile        = IwiTile()                 // [in,out] Reference to the IwiTile object for tiling. By default no tiling is used
)
{
    IppStatus ippStatus = ::iwiCopyChannel(&srcImage, srcChannel, &dstImage, dstChannel, &auxParams, &tile);
    OWN_ERROR_CHECK(ippStatus)
    return ippStatus;
}

/**/////////////////////////////////////////////////////////////////////////////
//                   iwiSplitChannels
///////////////////////////////////////////////////////////////////////////// */

// Auxiliary parameters class
// C API descriptions has more details
class IwiSplitChannelsParams: public ::IwiSplitChannelsParams
{
public:
    IW_BASE_PARAMS_CONSTRUCTORS(IwiSplitChannelsParams, iwiSplitChannels_SetDefaultParams)
    IwiSplitChannelsParams() {}
};

// Splits multi-channel image into array of single channel images.
// C API descriptions has more details.
// Throws:
//      ippStsDataTypeErr                   data type is illegal
//      ippStsNumChannelsErr                channels value is illegal
//      ippStsSizeErr                       size fields values are illegal
//      ippStsNullPtrErr                    unexpected NULL pointer
// Returns:
//      ippStsNoErr                         no errors
IW_DECL_CPP(IppStatus) iwiSplitChannels(
    const IwiImage               &srcImage,                                 // [in]     Reference to the source image
          IwiImageArray           dstImages,                                // [in,out] The array object of destination images. Uninitialized images will be skipped
    const IwiSplitChannelsParams &auxParams  = IwiSplitChannelsParams(),    // [in]     Reference to the auxiliary parameters structure
    const IwiTile                &tile       = IwiTile()                    // [in,out] Reference to the IwiTile object for tiling. By default no tiling is used
)
{
    ::IwiImage* const pDstImages[] = {&dstImages.imArray[0], &dstImages.imArray[1], &dstImages.imArray[2], &dstImages.imArray[3]};
    IppStatus ippStatus = ::iwiSplitChannels(&srcImage, pDstImages, &auxParams, &tile);
    OWN_ERROR_CHECK(ippStatus)
    return ippStatus;
}

/**/////////////////////////////////////////////////////////////////////////////
//                   iwiMergeChannels
///////////////////////////////////////////////////////////////////////////// */

// Auxiliary parameters class
// C API descriptions has more details
class IwiMergeChannelsParams: public ::IwiMergeChannelsParams
{
public:
    IW_BASE_PARAMS_CONSTRUCTORS(IwiMergeChannelsParams, iwiMergeChannels_SetDefaultParams)
    IwiMergeChannelsParams() {}
};

// Merges array of single channel images into one multi-channel image.
// C API descriptions has more details.
// Throws:
//      ippStsDataTypeErr                   data type is illegal
//      ippStsNumChannelsErr                channels value is illegal
//      ippStsSizeErr                       size fields values are illegal
//      ippStsNullPtrErr                    unexpected NULL pointer
// Returns:
//      ippStsNoErr                         no errors
IW_DECL_CPP(IppStatus) iwiMergeChannels(
    const IwiImageArray          &srcImages,                                // [in]     Reference to the array object of source images. Uninitialized images will be skipped
    IwiImage                     &dstImage,                                 // [in,out] Reference to the destination image
    const IwiMergeChannelsParams &auxParams  = IwiMergeChannelsParams(),    // [in]     Reference to the auxiliary parameters structure
    const IwiTile                &tile       = IwiTile()                    // [in,out] Reference to the IwiTile object for tiling. By default no tiling is used
)
{
    const ::IwiImage* const pSrcImages[] = {&srcImages.imArray[0], &srcImages.imArray[1], &srcImages.imArray[2], &srcImages.imArray[3]};
    IppStatus ippStatus = ::iwiMergeChannels(pSrcImages, &dstImage, &auxParams, &tile);
    OWN_ERROR_CHECK(ippStatus)
    return ippStatus;
}

/**/////////////////////////////////////////////////////////////////////////////
//                   iwiCreateBorder
///////////////////////////////////////////////////////////////////////////// */

// Auxiliary parameters class
// C API descriptions has more details
class IwiCreateBorderParams: public ::IwiCreateBorderParams
{
public:
    IW_BASE_PARAMS_CONSTRUCTORS(IwiCreateBorderParams, iwiCreateBorder_SetDefaultParams)
    IwiCreateBorderParams() {}
};

// Copies image data to destination image and constructs border of specified size.
// Destination image must have enough memory for a border according to inMemSize member.
// C API descriptions has more details.
// Throws:
//      ippStsNotSupportedModeErr           selected function mode is not supported
//      ippStsSizeErr                       1) size fields values are illegal
//                                          2) border.top or border.left are greater than corresponding inMemSize values of destination image
//                                          3) dst_width  + dst_inMemSize.right  < min_width  + border.right
//                                          4) dst_height + dst_inMemSize.bottom < min_height + border.bottom
//      ippStsNullPtrErr                    unexpected NULL pointer
// Returns:
//      ippStsNoErr                         no errors
IW_DECL_CPP(IppStatus) iwiCreateBorder(
    const IwiImage                 &srcImage,                                   // [in]     Reference to the source image
    IwiImage                       &dstImage,                                   // [in,out] Reference to the destination image actual data start
    IwiBorderSize                   borderSize,                                 // [in]     Size of border to reconstruct
    IwiBorderType                   border,                                     // [in]     Extrapolation algorithm and value for out of image pixels: ippBorderConst, ippBorderRepl, ippBorderMirror, ippBorderWrap
    const IwiCreateBorderParams    &auxParams   = IwiCreateBorderParams(),      // [in]     Reference to the auxiliary parameters structure
    const IwiTile                  &tile        = IwiTile()                     // [in,out] Reference to the IwiTile object for tiling. By default no tiling is used
)
{
    IppStatus ippStatus = ::iwiCreateBorder(&srcImage, &dstImage, borderSize, border.m_type, border.m_value, &auxParams, &tile);
    OWN_ERROR_CHECK(ippStatus)
    return ippStatus;
}

/**/////////////////////////////////////////////////////////////////////////////
//                   iwiSet
///////////////////////////////////////////////////////////////////////////// */

// Auxiliary parameters class
// C API descriptions has more details
class IwiSetParams: public ::IwiSetParams
{
public:
    IW_BASE_PARAMS_CONSTRUCTORS(IwiSetParams, iwiSet_SetDefaultParams)
    IwiSetParams() {}
};

// Sets each channel of the image to the specified values with masking.
// If mask is NULL, then calls iwiSet function.
// For masked operation, the function writes pixel values in the destination buffer only if the spatially corresponding
// mask array value is non-zero.
// C API descriptions has more details.
// Throws:
//      ippStsDataTypeErr                   data type is illegal
//      ippStsNumChannelsErr                channels value is illegal
//      ippStsSizeErr                       size fields values are illegal
//      ippStsNullPtrErr                    unexpected NULL pointer
// Returns:
//      ippStsNoErr                         no errors
IW_DECL_CPP(IppStatus) iwiSet(
    IwValueFloat        values,                         // [in]     Values to set to
    IwiImage           &dstImage,                       // [in,out] Reference to the destination image
    const IwiImage     &maskImage   = IwiImage(),       // [in]     Reference to the mask image. Mask must be 8-bit, 1 channel image
    const IwiSetParams &auxParams   = IwiSetParams(),   // [in]     Reference to the auxiliary parameters structure
    const IwiTile      &tile        = IwiTile()         // [in,out] Reference to the IwiTile object for tiling. By default no tiling is used
)
{
    IppStatus ippStatus = ::iwiSet(values, values.ValuesNum(), &dstImage, &maskImage, &auxParams, &tile);
    OWN_ERROR_CHECK(ippStatus)
    return ippStatus;
}

/**/////////////////////////////////////////////////////////////////////////////
//                   iwiSetChannel
///////////////////////////////////////////////////////////////////////////// */

// Auxiliary parameters class
// C API descriptions has more details
class IwiSetChannelParams: public ::IwiSetChannelParams
{
public:
    IW_BASE_PARAMS_CONSTRUCTORS(IwiSetChannelParams, iwiSetChannel_SetDefaultParams)
    IwiSetChannelParams() {}
};

// Sets selected channel of the multi-channel image to the specified value.
// C API descriptions has more details.
// Throws:
//      ippStsDataTypeErr                   data type is illegal
//      ippStsNumChannelsErr                channels value is illegal
//      ippStsSizeErr                       size fields values are illegal
//      ippStsNullPtrErr                    unexpected NULL pointer
// Returns:
//      ippStsNoErr                         no errors
IW_DECL_CPP(IppStatus) iwiSetChannel(
    double                      value,                                  // [in]     Value to set to
    IwiImage                   &dstImage,                               // [in,out] Reference to the destination image
    int                         channelNum,                             // [in]     Number of channel to be set (starting from 0)
    const IwiSetChannelParams  &auxParams   = IwiSetChannelParams(),    // [in]     Reference to the auxiliary parameters structure
    const IwiTile              &tile        = IwiTile()                 // [in,out] Reference to the IwiTile object for tiling. By default no tiling is used
)
{
    IppStatus ippStatus = ::iwiSetChannel(value, &dstImage, channelNum, &auxParams, &tile);
    OWN_ERROR_CHECK(ippStatus)
    return ippStatus;
}

/**/////////////////////////////////////////////////////////////////////////////
//                   iwiAdd
///////////////////////////////////////////////////////////////////////////// */

// Auxiliary parameters class
// C API descriptions has more details
class IwiAddParams: public ::IwiAddParams
{
public:
    IW_BASE_PARAMS_CONSTRUCTORS(IwiAddParams, iwiAdd_SetDefaultParams)
    IwiAddParams(int _scaleFactor = 0, IwiChDescriptor _chDesc = iwiChDesc_None)
    {
        this->scaleFactor = _scaleFactor;
        this->chDesc      = _chDesc;
    }
};

// Performs addition of one image to another and writes result to the output.
// C API descriptions has more details.
// Throws:
//      ippStsDataTypeErr                   data type is illegal
//      ippStsNumChannelsErr                channels value is illegal
//      ippStsSizeErr                       size fields values are illegal
//      ippStsNullPtrErr                    unexpected NULL pointer
// Returns:
//      ippStsNoErr                         no errors
IW_DECL_CPP(IppStatus) iwiAdd(
    const IwiImage      &addend1,                      // [in]     Reference to the first addend image
    const IwiImage      &addend2,                      // [in]     Reference to the second addend image
    IwiImage            &dstImage,                      // [in,out] Reference to the result image (can be same as the second addend)
    const IwiAddParams  &auxParams  = IwiAddParams(),   // [in]     Reference to the auxiliary parameters structure
    const IwiTile       &tile       = IwiTile()         // [in]     Reference to the IwiTile structure for tiling. By default no tiling is used
)
{
    IppStatus ippStatus = ::iwiAdd(&addend1, &addend2, &dstImage, &auxParams, &tile);
    OWN_ERROR_CHECK(ippStatus)
    return ippStatus;
}

/**/////////////////////////////////////////////////////////////////////////////
//                   iwiAddC
///////////////////////////////////////////////////////////////////////////// */

// Auxiliary parameters class
// C API descriptions has more details
class IwiAddCParams: public ::IwiAddCParams
{
public:
    IW_BASE_PARAMS_CONSTRUCTORS(IwiAddCParams, iwiAddC_SetDefaultParams)
    IwiAddCParams(int _scaleFactor = 0, IwiChDescriptor _chDesc = iwiChDesc_None)
    {
        this->scaleFactor = _scaleFactor;
        this->chDesc      = _chDesc;
    }
};

// Performs addition of the constant to the image another and writes result to the output.
// C API descriptions has more details.
// Returns:
//      ippStsDataTypeErr                   data type is illegal
//      ippStsNumChannelsErr                channels value is illegal
//      ippStsSizeErr                       size fields values are illegal
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL_CPP(IppStatus) iwiAddC(
    IwValueFloat         addend,                        // [in]     Addends. One element for each channel
    const IwiImage      &addendImage,                   // [in]     Reference to the addend image
    IwiImage            &dstImage,                      // [in,out] Reference to the difference image (can be same as addend image)
    const IwiAddCParams &auxParams  = IwiAddCParams(),  // [in]     Reference to the auxiliary parameters structure
    const IwiTile       &tile       = IwiTile()         // [in]     Reference to the IwiTile structure for tiling. By default no tiling is used
)
{
    IppStatus ippStatus = ::iwiAddC(addend, addend.ValuesNum(), &addendImage, &dstImage, &auxParams, &tile);
    OWN_ERROR_CHECK(ippStatus)
    return ippStatus;
}

/**/////////////////////////////////////////////////////////////////////////////
//                   iwiSub
///////////////////////////////////////////////////////////////////////////// */

// Auxiliary parameters class
// C API descriptions has more details
class IwiSubParams: public ::IwiSubParams
{
public:
    IW_BASE_PARAMS_CONSTRUCTORS(IwiSubParams, iwiSub_SetDefaultParams)
    IwiSubParams(int _scaleFactor = 0, IwiChDescriptor _chDesc = iwiChDesc_None)
    {
        this->scaleFactor = _scaleFactor;
        this->chDesc      = _chDesc;
    }
};

// Performs subtraction of the first image from the second image and writes result to the output.
// C API descriptions has more details.
// Throws:
//      ippStsDataTypeErr                   data type is illegal
//      ippStsNumChannelsErr                channels value is illegal
//      ippStsSizeErr                       size fields values are illegal
//      ippStsNullPtrErr                    unexpected NULL pointer
// Returns:
//      ippStsNoErr                         no errors
IW_DECL_CPP(IppStatus) iwiSub(
    const IwiImage      &subtrahend,                    // [in]     Reference to the subtrahend image
    const IwiImage      &minuend,                       // [in]     Reference to the minuend image
    IwiImage            &dstImage,                      // [in,out] Reference to the difference image (can be same as minuend)
    const IwiSubParams  &auxParams  = IwiSubParams(),   // [in]     Reference to the auxiliary parameters structure
    const IwiTile       &tile       = IwiTile()         // [in]     Reference to the IwiTile structure for tiling. By default no tiling is used
)
{
    IppStatus ippStatus = ::iwiSub(&subtrahend, &minuend, &dstImage, &auxParams, &tile);
    OWN_ERROR_CHECK(ippStatus)
    return ippStatus;
}

/**/////////////////////////////////////////////////////////////////////////////
//                   iwiSubC
///////////////////////////////////////////////////////////////////////////// */

// Auxiliary parameters class
// C API descriptions has more details
class IwiSubCParams: public ::IwiSubCParams
{
public:
    IW_BASE_PARAMS_CONSTRUCTORS(IwiSubCParams, iwiSubC_SetDefaultParams)
    IwiSubCParams(int _scaleFactor = 0, IwiChDescriptor _chDesc = iwiChDesc_None)
    {
        this->scaleFactor = _scaleFactor;
        this->chDesc      = _chDesc;
    }
};

// Performs subtraction of the constant from the image and writes result to the output.
// C API descriptions has more details.
// Returns:
//      ippStsDataTypeErr                   data type is illegal
//      ippStsNumChannelsErr                channels value is illegal
//      ippStsSizeErr                       size fields values are illegal
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL_CPP(IppStatus) iwiSubC(
    IwValueFloat         subtrahend,                    // [in]     Subtrahends. One element for each channel
    const IwiImage      &minuendImage,                  // [in]     Reference to the minuend image
    IwiImage            &dstImage,                      // [in,out] Reference to the difference image (can be same as minuend)
    const IwiSubCParams &auxParams  = IwiSubCParams(),  // [in]     Reference to the auxiliary parameters structure
    const IwiTile       &tile       = IwiTile()         // [in]     Reference to the IwiTile structure for tiling. By default no tiling is used
)
{
    IppStatus ippStatus = ::iwiSubC(subtrahend, subtrahend.ValuesNum(), &minuendImage, &dstImage, &auxParams, &tile);
    OWN_ERROR_CHECK(ippStatus)
    return ippStatus;
}

/**/////////////////////////////////////////////////////////////////////////////
//                   iwiMul
///////////////////////////////////////////////////////////////////////////// */

// Auxiliary parameters class
// C API descriptions has more details
class IwiMulParams: public ::IwiMulParams
{
public:
    IW_BASE_PARAMS_CONSTRUCTORS(IwiMulParams, iwiMul_SetDefaultParams)
    IwiMulParams(int _scaleFactor = 0, IppHintAlgorithm _algoMode = ippAlgHintNone, IwiChDescriptor _chDesc = iwiChDesc_None)
    {
        this->algoMode          = _algoMode;
        this->scaleFactor       = _scaleFactor;
        this->chDesc            = _chDesc;
    }
};

// Performs multiplication of one image by another and writes result to the output
// C API descriptions has more details.
// Throws:
//      ippStsDataTypeErr                   data type is illegal
//      ippStsNumChannelsErr                channels value is illegal
//      ippStsSizeErr                       size fields values are illegal
//      ippStsNullPtrErr                    unexpected NULL pointer
// Returns:
//      ippStsNoErr                         no errors
IW_DECL_CPP(IppStatus) iwiMul(
    const IwiImage     &factor1,                        // [in]     Reference to the first factor image
    const IwiImage     &factor2,                        // [in]     Reference to the second factor image
    IwiImage           &dstImage,                       // [in,out] Reference to the product image (can be same as the second factor)
    const IwiMulParams &auxParams   = IwiMulParams(),   // [in]     Reference to the auxiliary parameters structure
    const IwiTile      &tile        = IwiTile()         // [in]     Reference to the IwiTile structure for tiling. By default no tiling is used
)
{
    IppStatus ippStatus = ::iwiMul(&factor1, &factor2, &dstImage, &auxParams, &tile);
    OWN_ERROR_CHECK(ippStatus)
    return ippStatus;
}

/**/////////////////////////////////////////////////////////////////////////////
//                   iwiMulC
///////////////////////////////////////////////////////////////////////////// */

// Auxiliary parameters class
// C API descriptions has more details
class IwiMulCParams: public ::IwiMulCParams
{
public:
    IW_BASE_PARAMS_CONSTRUCTORS(IwiMulCParams, iwiMulC_SetDefaultParams)
    IwiMulCParams(int _scaleFactor = 0, IppHintAlgorithm _algoMode = ippAlgHintNone, IppRoundMode _round = ippRndNear, IwiChDescriptor _chDesc = iwiChDesc_None)
    {
        this->roundMode   = _round;
        this->algoMode    = _algoMode;
        this->scaleFactor = _scaleFactor;
        this->chDesc      = _chDesc;
    }
};

// Performs multiplication of one image by constant and writes result to the output.
// C API descriptions has more details.
// Returns:
//      ippStsDataTypeErr                   data type is illegal
//      ippStsNumChannelsErr                channels value is illegal
//      ippStsSizeErr                       size fields values are illegal
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL_CPP(IppStatus) iwiMulC(
    IwValueFloat         factor,                        // [in]     Factors. One element for each channel
    const IwiImage      &factorImage,                   // [in]     Reference to the factor image
    IwiImage            &dstImage,                      // [in,out] Reference to the product image (can be same as factor image)
    const IwiMulCParams &auxParams  = IwiMulCParams(),  // [in]     Reference to the auxiliary parameters structure
    const IwiTile       &tile       = IwiTile()         // [in]     Reference to the IwiTile structure for tiling. By default no tiling is used
)
{
    IppStatus ippStatus = ::iwiMulC(factor, factor.ValuesNum(), &factorImage, &dstImage, &auxParams, &tile);
    OWN_ERROR_CHECK(ippStatus)
    return ippStatus;
}

/**/////////////////////////////////////////////////////////////////////////////
//                   iwiDiv
///////////////////////////////////////////////////////////////////////////// */

// Auxiliary parameters class
// C API descriptions has more details
class IwiDivParams: public ::IwiDivParams
{
public:
    IW_BASE_PARAMS_CONSTRUCTORS(IwiDivParams, iwiDiv_SetDefaultParams)
    IwiDivParams(int _scaleFactor = 0, IppHintAlgorithm _algoMode = ippAlgHintNone, IwiChDescriptor _chDesc = iwiChDesc_None)
    {
        this->algoMode          = _algoMode;
        this->scaleFactor       = _scaleFactor;
        this->chDesc            = _chDesc;
    }
};

// Performs division of the second image by the first image and writes result to the output.
// C API descriptions has more details.
// Throws:
//      ippStsDataTypeErr                   data type is illegal
//      ippStsNumChannelsErr                channels value is illegal
//      ippStsSizeErr                       size fields values are illegal
//      ippStsNullPtrErr                    unexpected NULL pointer
// Returns:
//      ippStsNoErr                         no errors
IW_DECL_CPP(IppStatus) iwiDiv(
    const IwiImage     &denominator,                    // [in]     Reference to the denominator image
    const IwiImage     &numerator,                      // [in]     Reference to the numerator image
    IwiImage           &dstImage,                       // [in,out] Reference to the fraction image (can be same as numerator)
    const IwiDivParams &auxParams   = IwiDivParams(),   // [in]     Reference to the auxiliary parameters structure
    const IwiTile      &tile        = IwiTile()         // [in]     Reference to the IwiTile structure for tiling. By default no tiling is used
)
{
    IppStatus ippStatus = ::iwiDiv(&denominator, &numerator, &dstImage, &auxParams, &tile);
    OWN_ERROR_CHECK(ippStatus)
    return ippStatus;
}

/**/////////////////////////////////////////////////////////////////////////////
//                   iwiDivC
///////////////////////////////////////////////////////////////////////////// */

// Auxiliary parameters class
// C API descriptions has more details
class IwiDivCParams: public ::IwiDivCParams
{
public:
    IW_BASE_PARAMS_CONSTRUCTORS(IwiDivCParams, iwiDivC_SetDefaultParams)
    IwiDivCParams(int _scaleFactor = 0, IppHintAlgorithm _algoMode = ippAlgHintNone, IppRoundMode _round = ippRndNear, IwiChDescriptor _chDesc = iwiChDesc_None)
    {
        this->roundMode   = _round;
        this->algoMode    = _algoMode;
        this->scaleFactor = _scaleFactor;
        this->chDesc      = _chDesc;
    }
};

// Performs division of the second image by constant and writes result to the output.
// C API descriptions has more details.
// Returns:
//      ippStsDataTypeErr                   data type is illegal
//      ippStsNumChannelsErr                channels value is illegal
//      ippStsSizeErr                       size fields values are illegal
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL_CPP(IppStatus) iwiDivC(
    IwValueFloat         denominator,                   // [in]     Denominators. One element for each channel
    const IwiImage      &numeratorImage,                // [in]     Reference to the numerator image
    IwiImage            &dstImage,                      // [in,out] Reference to the fraction image (can be same as numerator)
    const IwiDivCParams &auxParams  = IwiDivCParams(),  // [in]     Reference to the auxiliary parameters structure
    const IwiTile       &tile       = IwiTile()         // [in]     Reference to the IwiTile structure for tiling. By default no tiling is used
)
{
    IppStatus ippStatus = ::iwiDivC(denominator, denominator.ValuesNum(), &numeratorImage, &dstImage, &auxParams, &tile);
    OWN_ERROR_CHECK(ippStatus)
    return ippStatus;
}

/**/////////////////////////////////////////////////////////////////////////////
//                   iwiSwapChannels
///////////////////////////////////////////////////////////////////////////// */

// Auxiliary parameters class
// C API descriptions has more details
class IwiSwapChannelsParams: public ::IwiSwapChannelsParams
{
public:
    IW_BASE_PARAMS_CONSTRUCTORS(IwiSwapChannelsParams, iwiSwapChannels_SetDefaultParams)
    IwiSwapChannelsParams(IwiChDescriptor _chDesc = iwiChDesc_None)
    {
        this->chDesc = _chDesc;
    }
};

// Swaps image channels according to the destination order parameter.
// C API descriptions has more details.
// Throws:
//      ippStsChannelOrderErr               destination order is out of the range
//      ippStsDataTypeErr                   data type is illegal
//      ippStsNumChannelsErr                channels value is illegal
//      ippStsSizeErr                       size fields values are illegal
//      ippStsNullPtrErr                    unexpected NULL pointer
// Returns:
//      ippStsNoErr                         no errors
IW_DECL_CPP(IppStatus) iwiSwapChannels(
    const IwiImage              &srcImage,                              // [in]     Reference to the source image
    IwiImage                    &dstImage,                              // [in,out] Reference to the destination image
    const IwValueInt            &dstOrder,                              // [in]     Destination image channels order: dst[channel] = src[dstOrder[channel]]
    double                       value      = IwValueMax,               // [in]     Value to fill the destination channel if number of destination channels is bigger than number of source channels
    const IwiSwapChannelsParams &auxParams  = IwiSwapChannelsParams(),  // [in]     Reference to the auxiliary parameters structure
    const IwiTile               &tile       = IwiTile()                 // [in]     Reference to the IwiTile structure for tiling. By default no tiling is used
)
{
    IppStatus ippStatus   = ::iwiSwapChannels(&srcImage, &dstImage, dstOrder, value, &auxParams, &tile);
    OWN_ERROR_CHECK(ippStatus)
    return ippStatus;
}

/**/////////////////////////////////////////////////////////////////////////////
//                   iwiScale
///////////////////////////////////////////////////////////////////////////// */

// Auxiliary parameters class
// C API descriptions has more details
class IwiScaleParams: public ::IwiScaleParams
{
public:
    IW_BASE_PARAMS_CONSTRUCTORS(IwiScaleParams, iwiScale_SetDefaultParams)
    IwiScaleParams(IppHintAlgorithm _algoMode = ippAlgHintNone)
    {
        this->algoMode = _algoMode;
    }
};

// Converts image from one data type to another with specified scaling and shifting
// DST = saturate(SRC*mulVal + addVal)
// C API descriptions has more details.
// Throws:
//      ippStsDataTypeErr                   data type is illegal
//      ippStsSizeErr                       size fields values are illegal
//      ippStsInplaceModeNotSupportedErr    doesn't support output into the source buffer. If data types are different
//      ippStsNullPtrErr                    unexpected NULL pointer
// Returns:
//      ippStsNoErr                         no errors
IW_DECL_CPP(IppStatus) iwiScale(
    const IwiImage         &srcImage,                       // [in]     Reference to the source image
    IwiImage               &dstImage,                       // [in,out] Reference to the destination image
    Ipp64f                  mulVal,                         // [in]     Multiplier
    Ipp64f                  addVal,                         // [in]     Addend
    const IwiScaleParams   &auxParams   = IwiScaleParams(), // [in]     Reference to the auxiliary parameters structure
    const IwiTile          &tile        = IwiTile()         // [in]     Reference to the IwiTile object for tiling. By default no tiling is used
)
{
    IppStatus ippStatus = ::iwiScale(&srcImage, &dstImage, mulVal, addVal, &auxParams, &tile);
    OWN_ERROR_CHECK(ippStatus)
    return ippStatus;
}

// Returns multiplication and addend values for iwiScale function to perform accurate data range scaling between two data types
// Data range for float values is considered to be from 0 to 1
// Throws:
//      ippStsDataTypeErr                   data type is illegal
// Returns:
//      ippStsNoErr                         no errors
IW_DECL_CPP(IppStatus) iwiScale_GetScaleVals(
    IppDataType srcType,    // [in]     Source data type
    IppDataType dstType,    // [in]     Destination data type
    Ipp64f     &mulVal,     // [out]    Pointer to multiplier
    Ipp64f     &addVal      // [out]    Pointer to addend
)
{
    IppStatus ippStatus = ::iwiScale_GetScaleVals(srcType, dstType, &mulVal, &addVal);
    OWN_ERROR_CHECK(ippStatus)
    return ippStatus;
}

}

#endif
