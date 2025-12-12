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

#if !defined( __IPP_IWPP_IMAGE_FILTER__ )
#define __IPP_IWPP_IMAGE_FILTER__

#include "iw/iw_image_filter.h"
#include "iw++/iw_image.hpp"

namespace ipp
{

/* /////////////////////////////////////////////////////////////////////////////
//                   iwiFilter
///////////////////////////////////////////////////////////////////////////// */

// Auxiliary parameters class
// C API descriptions has more details
class IwiFilterParams: public ::IwiFilterParams
{
public:
    IW_BASE_PARAMS_CONSTRUCTORS(IwiFilterParams, iwiFilter_SetDefaultParams)
    IwiFilterParams(double _divisor = 0, int _offset = 0, IppHintAlgorithm _algoMode = ippAlgHintNone, IppRoundMode _roundMode = ippRndNear)
    {
        this->divisor    = _divisor;
        this->offset     = _offset;
        this->roundMode  = _roundMode;
        this->algoMode   = _algoMode;
    }
};

// Performs filtration of an image with an arbitrary kernel
// Throws:
//      ippStsBorderErr                     border value is illegal
//      ippStsDataTypeErr                   data type is illegal
//      ippStsNumChannelsErr                channels value is illegal
//      ippStsNotEvenStepErr                step value is not divisible by size of elements
//      ippStsNotSupportedModeErr           selected function mode is not supported
//      ippStsNoMemErr                      failed to allocate memory
//      ippStsInplaceModeNotSupportedErr    doesn't support output into the source buffer
//      ippStsNullPtrErr                    unexpected NULL pointer
// Returns:
//      ippStsNoErr                         no errors
IW_DECL_CPP(IppStatus) iwiFilter(
    const IwiImage         &srcImage,                           // [in]     Reference to the source image
    IwiImage               &dstImage,                           // [in,out] Reference to the destination image
    const IwiImage         &kernel,                             // [in]     Reference to the filter kernel image. Kernel must be continuous, [16s,32f], [C1]
    const IwiFilterParams  &auxParams   = IwiFilterParams(),    // [in]     Reference to the auxiliary parameters structure
    const IwiBorderType    &border      = ippBorderRepl,        // [in]     Extrapolation algorithm and value for out of image pixels: ippBorderConst, ippBorderRepl, ippBorderMirror, ippBorderInMem
    const IwiTile          &tile        = IwiTile()             // [in]     Reference to the IwiTile structure for tiling. By default no tiling is used
)
{
    IppStatus ippStatus = ::iwiFilter(&srcImage, &dstImage, &kernel, &auxParams, border.m_type, border.m_value, &tile);
    OWN_ERROR_CHECK(ippStatus)
    return ippStatus;
}

/* /////////////////////////////////////////////////////////////////////////////
//                   iwiFilterBox
///////////////////////////////////////////////////////////////////////////// */

// Auxiliary parameters class
// C API descriptions has more details
class IwiFilterBoxParams: public ::IwiFilterBoxParams
{
public:
    IW_BASE_PARAMS_CONSTRUCTORS(IwiFilterBoxParams, iwiFilterBox_SetDefaultParams)
    IwiFilterBoxParams(IwiChDescriptor _chDesc = iwiChDesc_None)
    {
        this->chDesc = _chDesc;
    }
};

// Applies box filter to the image
// Throws:
//      ippStsBorderErr                     border value is illegal
//      ippStsDataTypeErr                   data type is illegal
//      ippStsNumChannelsErr                channels value is illegal
//      ippStsNotEvenStepErr                step value is not divisible by size of elements
//      ippStsNotSupportedModeErr           selected function mode is not supported
//      ippStsNoMemErr                      failed to allocate memory
//      ippStsInplaceModeNotSupportedErr    doesn't support output into the source buffer
//      ippStsNullPtrErr                    unexpected NULL pointer
// Returns:
//      ippStsNoErr                         no errors
IW_DECL_CPP(IppStatus) iwiFilterBox(
    const IwiImage             &srcImage,                           // [in]     Reference to the source image
    IwiImage                   &dstImage,                           // [in,out] Reference to the destination image
    IwiSize                     kernel,                             // [in]     Size of the filter kernel
    const IwiFilterBoxParams   &auxParams   = IwiFilterBoxParams(), // [in]     Reference to the auxiliary parameters structure
    const IwiBorderType        &border      = ippBorderRepl,        // [in]     Extrapolation algorithm and value for out of image pixels: ippBorderConst, ippBorderRepl, ippBorderMirror, ippBorderInMem
    const IwiTile              &tile        = IwiTile()             // [in]     Reference to the IwiTile structure for tiling. By default no tiling is used
)
{
    IppStatus ippStatus = ::iwiFilterBox(&srcImage, &dstImage, kernel, &auxParams, border.m_type, border.m_value, &tile);
    OWN_ERROR_CHECK(ippStatus)
    return ippStatus;
}

/* /////////////////////////////////////////////////////////////////////////////
//                   iwiFilterSobel
///////////////////////////////////////////////////////////////////////////// */

// Auxiliary parameters class
// C API descriptions has more details
class IwiFilterSobelParams: public ::IwiFilterSobelParams
{
public:
    IW_BASE_PARAMS_CONSTRUCTORS(IwiFilterSobelParams, iwiFilterSobel_SetDefaultParams)
    IwiFilterSobelParams() {}
};

// Applies Sobel filter of specific type to the source image
// C API descriptions has more details.
// Throws:
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
// Returns:
//      ippStsNoErr                         no errors
IW_DECL_CPP(IppStatus) iwiFilterSobel(
    const IwiImage             &srcImage,                               // [in]     Reference to the source image
    IwiImage                   &dstImage,                               // [in,out] Reference to the destination image
    IwiDerivativeType           opType,                                 // [in]     Type of derivative from IwiDerivativeType
    IppiMaskSize                kernelSize  = ippMskSize3x3,            // [in]     Size of filter kernel: ippMskSize3x3, ippMskSize5x5
    const IwiFilterSobelParams &auxParams   = IwiFilterSobelParams(),   // [in]     Reference to the auxiliary parameters structure
    const IwiBorderType        &border      = ippBorderRepl,            // [in]     Extrapolation algorithm and value for out of image pixels: ippBorderConst, ippBorderRepl, ippBorderMirror, ippBorderInMem
    const IwiTile              &tile        = IwiTile()                 // [in]     Reference to the IwiTile structure for tiling. By default no tiling is used
)
{
    IppStatus ippStatus = ::iwiFilterSobel(&srcImage, &dstImage, opType, kernelSize, &auxParams, border.m_type, border.m_value, &tile);
    OWN_ERROR_CHECK(ippStatus)
    return ippStatus;
}

/* /////////////////////////////////////////////////////////////////////////////
//                   iwiFilterScharr
///////////////////////////////////////////////////////////////////////////// */

// Auxiliary parameters class
// C API descriptions has more details
class IwiFilterScharrParams: public ::IwiFilterScharrParams
{
public:
    IW_BASE_PARAMS_CONSTRUCTORS(IwiFilterScharrParams, iwiFilterScharr_SetDefaultParams)
    IwiFilterScharrParams() {}
};

// Applies Scharr filter of specific type to the source image
// C API descriptions has more details
// Throws:
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
// Returns:
//      ippStsNoErr                         no errors
IW_DECL_CPP(IppStatus) iwiFilterScharr(
    const IwiImage              &srcImage,                              // [in]     Reference to the source image
    IwiImage                    &dstImage,                              // [in,out] Reference to the destination image
    IwiDerivativeType            opType,                                // [in]     Type of derivative from IwiDerivativeType
    IppiMaskSize                 kernelSize = ippMskSize3x3,            // [in]     Size of filter kernel: ippMskSize3x3, ippMskSize5x5
    const IwiFilterScharrParams &auxParams  = IwiFilterScharrParams(),  // [in]     Reference to the auxiliary parameters structure
    const IwiBorderType         &border     = ippBorderRepl,            // [in]     Extrapolation algorithm and value for out of image pixels: ippBorderConst, ippBorderRepl, ippBorderMirror, ippBorderInMem
    const IwiTile               &tile       = IwiTile()                 // [in]     Reference to the IwiTile structure for tiling. By default no tiling is used
)
{
    IppStatus ippStatus = ::iwiFilterScharr(&srcImage, &dstImage, opType, kernelSize, &auxParams, border.m_type, border.m_value, &tile);
    OWN_ERROR_CHECK(ippStatus)
    return ippStatus;
}

/**/////////////////////////////////////////////////////////////////////////////
//                   iwiFilterLaplacian
///////////////////////////////////////////////////////////////////////////// */

// Auxiliary parameters class
// C API descriptions has more details
class IwiFilterLaplacianParams: public ::IwiFilterLaplacianParams
{
public:
    IW_BASE_PARAMS_CONSTRUCTORS(IwiFilterLaplacianParams, iwiFilterLaplacian_SetDefaultParams)
    IwiFilterLaplacianParams() {}
};

// Applies Laplacian filter to the source image
// C API descriptions has more details.
// Throws:
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
// Returns:
//      ippStsNoErr                         no errors
IW_DECL_CPP(IppStatus) iwiFilterLaplacian(
    const IwiImage                 &srcImage,                                   // [in]     Reference to the source image
    IwiImage                       &dstImage,                                   // [in,out] Reference to the destination image
    IppiMaskSize                    kernelSize  = ippMskSize3x3,                // [in]     Size of filter kernel: ippMskSize3x3, ippMskSize5x5
    const IwiFilterLaplacianParams &auxParams   = IwiFilterLaplacianParams(),   // [in]     Reference to the auxiliary parameters structure
    const IwiBorderType            &border      = ippBorderRepl,                // [in]     Extrapolation algorithm and value for out of image pixels: ippBorderConst, ippBorderRepl, ippBorderMirror, ippBorderInMem
    const IwiTile                  &tile        = IwiTile()                     // [in]     Reference to the IwiTile structure for tiling. By default no tiling is used
)
{
    IppStatus ippStatus = ::iwiFilterLaplacian(&srcImage, &dstImage, kernelSize, &auxParams, border.m_type, border.m_value, &tile);
    OWN_ERROR_CHECK(ippStatus)
    return ippStatus;
}


/**/////////////////////////////////////////////////////////////////////////////
//                   iwiFilterGaussian
///////////////////////////////////////////////////////////////////////////// */

// Auxiliary parameters class
// C API descriptions has more details
class IwiFilterGaussianParams: public ::IwiFilterGaussianParams
{
public:
    IW_BASE_PARAMS_CONSTRUCTORS(IwiFilterGaussianParams, iwiFilterGaussian_SetDefaultParams)
    IwiFilterGaussianParams(IwiChDescriptor _chDesc = iwiChDesc_None)
    {
        this->chDesc = _chDesc;
    }
};

// Applies Gaussian filter to the source image
// C API descriptions has more details.
// Throws:
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
// Returns:
//      ippStsNoErr                         no errors
IW_DECL_CPP(IppStatus) iwiFilterGaussian(
    const IwiImage                 &srcImage,                                   // [in]     Reference to the source image
    IwiImage                       &dstImage,                                   // [in,out] Reference to the destination image
    int                             kernelSize,                                 // [in]     Size of the Gaussian kernel (odd, greater or equal to 3)
    double                          sigma,                                      // [in]     Standard deviation of the Gaussian kernel
    const IwiFilterGaussianParams  &auxParams   = IwiFilterGaussianParams(),    // [in]     Reference to the auxiliary parameters structure
    const IwiBorderType            &border      = ippBorderRepl,                // [in]     Extrapolation algorithm and value for out of image pixels: ippBorderConst, ippBorderRepl, ippBorderMirror, ippBorderInMem
    const IwiTile                  &tile        = IwiTile()                     // [in]     Reference to the IwiTile structure for tiling. By default no tiling is used
)
{
    IppStatus ippStatus = ::iwiFilterGaussian(&srcImage, &dstImage, kernelSize, sigma, &auxParams, border.m_type, border.m_value, &tile);
    OWN_ERROR_CHECK(ippStatus)
    return ippStatus;
}

// FilterGaussian operation class
// C API descriptions has more details.
class IwiFilterGaussian
{
public:
    // Default constructor
    IwiFilterGaussian()
    {
        m_bInitialized = false;
    }

    // Constructor with initialization
    // Throws:
    //      ippStsMaskSizeErr                   mask value is illegal
    //      ippStsDataTypeErr                   data type is illegal
    //      ippStsNumChannelsErr                channels value is illegal
    //      ippStsNotEvenStepErr                step value is not divisible by size of elements
    //      ippStsBorderErr                     border value is illegal
    //      ippStsNotSupportedModeErr           selected function mode is not supported
    //      ippStsNoMemErr                      failed to allocate memory
    //      ippStsSizeErr                       size fields values are illegal
    //      ippStsNullPtrErr                    unexpected NULL pointer
    IwiFilterGaussian(
        IwiSize                         size,                                       // [in]     Size of the image in pixels
        IppDataType                     dataType,                                   // [in]     Image pixel type
        int                             channels,                                   // [in]     Number of image channels
        int                             kernelSize,                                 // [in]     Size of the Gaussian kernel (odd, greater or equal to 3)
        double                          sigma,                                      // [in]     Standard deviation of the Gaussian kernel
        const IwiFilterGaussianParams  &auxParams   = IwiFilterGaussianParams(),    // [in] Reference to the auxiliary parameters structure
        const IwiBorderType            &border      = ippBorderRepl                 // [in] Extrapolation algorithm and value for out of image pixels: ippBorderConst, ippBorderRepl, ippBorderInMem
    )
    {
        m_bInitialized = false;

        IppStatus ippStatus = InitAlloc(size, dataType, channels, kernelSize, sigma, auxParams, border);
        OWN_ERROR_CHECK_THROW_ONLY(ippStatus);
    }

    // Default destructor
    ~IwiFilterGaussian()
    {
        if(m_bInitialized)
        {
            ::iwiFilterGaussian_Free(m_pSpec);
            m_bInitialized = false;
        }
    }

    // Allocates and initializes internal data structure
    // Throws:
    //      ippStsMaskSizeErr                   mask value is illegal
    //      ippStsDataTypeErr                   data type is illegal
    //      ippStsNumChannelsErr                channels value is illegal
    //      ippStsNotEvenStepErr                step value is not divisible by size of elements
    //      ippStsBorderErr                     border value is illegal
    //      ippStsNotSupportedModeErr           selected function mode is not supported
    //      ippStsNoMemErr                      failed to allocate memory
    //      ippStsSizeErr                       size fields values are illegal
    //      ippStsNullPtrErr                    unexpected NULL pointer
    // Returns:
    //      ippStsNoErr                         no errors
    IppStatus InitAlloc(
        IwiSize                         size,                                       // [in]     Size of the image in pixels
        IppDataType                     dataType,                                   // [in]     Image pixel type
        int                             channels,                                   // [in]     Number of image channels
        int                             kernelSize,                                 // [in]     Size of the Gaussian kernel (odd, greater or equal to 3)
        double                          sigma,                                      // [in]     Standard deviation of the Gaussian kernel
        const IwiFilterGaussianParams  &auxParams   = IwiFilterGaussianParams(),    // [in] Reference to the auxiliary parameters structure
        const IwiBorderType            &border      = ippBorderRepl                 // [in] Extrapolation algorithm and value for out of image pixels: ippBorderConst, ippBorderRepl, ippBorderInMem
    )
    {
        if(m_bInitialized)
        {
            ::iwiFilterGaussian_Free(m_pSpec);
            m_bInitialized = false;
        }

        IppStatus ippStatus = ::iwiFilterGaussian_InitAlloc(&m_pSpec, size, dataType, channels, kernelSize, sigma, &auxParams, border);
        OWN_ERROR_CHECK(ippStatus);

        m_bInitialized = true;
        return ippStatus;
    }

    // Applies Gaussian filter to the source image
    // Throws:
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
    // Returns:
    //      ippStsNoErr                         no errors
    IppStatus operator()(
        const IwiImage         &srcImage,                       // [in]     Reference to the source image
        IwiImage               &dstImage,                       // [in,out] Reference to the destination image
        const IwiBorderType    &border      = ippBorderRepl,    // [in]     Extrapolation algorithm and value for out of image pixels: ippBorderConst, ippBorderRepl, ippBorderInMem
        const IwiTile          &tile        = IwiTile()         // [in]     Reference to the IwiTile structure for tiling. By default no tiling is used
    ) const
    {
        if(m_bInitialized)
        {
            IppStatus ippStatus = ::iwiFilterGaussian_Process(m_pSpec, &srcImage, &dstImage, border.m_type, border.m_value, &tile);
            OWN_ERROR_CHECK(ippStatus);
            return ippStatus;
        }
        else
            OWN_ERROR_THROW(ippStsBadArgErr);
    }

private:
    IwiFilterGaussianSpec  *m_pSpec;        // Pointer to internal spec structure
    bool                    m_bInitialized; // Initialization flag
};

/* /////////////////////////////////////////////////////////////////////////////
//                   iwiFilterCanny
///////////////////////////////////////////////////////////////////////////// */

// Auxiliary parameters class
// C API descriptions has more details
class IwiFilterCannyParams: public ::IwiFilterCannyParams
{
public:
    IW_BASE_PARAMS_CONSTRUCTORS(IwiFilterCannyParams, iwiFilterCanny_SetDefaultParams)
    IwiFilterCannyParams(IppiDifferentialKernel _kernel = ippFilterSobel, IppiMaskSize _kernelSize = ippMskSize3x3, IppNormType _norm = ippNormL2)
    {
        this->kernel     = _kernel;
        this->kernelSize = _kernelSize;
        this->norm       = _norm;
    }
};

// Applies Canny edge detector to the source image
// C API descriptions has more details.
// Throws:
//      ippStsMaskSizeErr                   mask value is illegal
//      ippStsDataTypeErr                   data type is illegal
//      ippStsNumChannelsErr                channels value is illegal
//      ippStsNotEvenStepErr                step value is not divisible by size of elements
//      ippStsBorderErr                     border value is illegal
//      ippStsNoMemErr                      failed to allocate memory
//      ippStsSizeErr                       size fields values are illegal
//      ippStsInplaceModeNotSupportedErr    doesn't support output into the source buffer
//      ippStsNullPtrErr                    unexpected NULL pointer
// Returns:
//      ippStsNoErr                         no errors
IW_DECL_CPP(IppStatus) iwiFilterCanny(
    const IwiImage             &srcImage,                               // [in]     Reference to the source image
    IwiImage                   &dstImage,                               // [in,out] Reference to the destination image
    Ipp32f                      treshLow    = 50,                       // [in]     Lower threshold for edges detection
    Ipp32f                      treshHigh   = 150,                      // [in]     Upper threshold for edges detection
    const IwiFilterCannyParams &auxParams   = IwiFilterCannyParams(),   // [in]     Reference to the auxiliary parameters structure
    const IwiBorderType        &border      = ippBorderRepl             // [in]     Extrapolation algorithm and value for out of image pixels: ippBorderConst, ippBorderRepl, ippBorderMirror, ippBorderInMem
)
{
    IppStatus ippStatus = ::iwiFilterCanny(&srcImage, &dstImage, treshLow, treshHigh, &auxParams, border.m_type, border.m_value);
    OWN_ERROR_CHECK(ippStatus)
    return ippStatus;
}

/* /////////////////////////////////////////////////////////////////////////////
//                   iwiFilterCannyDeriv
///////////////////////////////////////////////////////////////////////////// */

// Auxiliary parameters class
// C API descriptions has more details
class IwiFilterCannyDerivParams: public ::IwiFilterCannyDerivParams
{
public:
    IW_BASE_PARAMS_CONSTRUCTORS(IwiFilterCannyDerivParams, iwiFilterCannyDeriv_SetDefaultParams)
    IwiFilterCannyDerivParams(IppNormType _norm = ippNormL2)
    {
        this->norm = _norm;
    }
};

// Applies Canny edge detector to the image derivatives
// C API descriptions has more details.
// Throws:
//      ippStsDataTypeErr                   data type is illegal
//      ippStsNumChannelsErr                channels value is illegal
//      ippStsNotEvenStepErr                step value is not divisible by size of elements
//      ippStsNoMemErr                      failed to allocate memory
//      ippStsSizeErr                       size fields values are illegal
//      ippStsNullPtrErr                    unexpected NULL pointer
// Returns:
//      ippStsNoErr                         no errors
IW_DECL_CPP(IppStatus) iwiFilterCannyDeriv(
    const IwiImage                  &srcImageDx,                                // [in]     Reference to X derivative of the source image
    const IwiImage                  &srcImageDy,                                // [in]     Reference to Y derivative of the source image
    IwiImage                        &dstImage,                                  // [in,out] Reference to the destination image
    Ipp32f                           treshLow   = 50,                           // [in]     Lower threshold for edges detection
    Ipp32f                           treshHigh  = 150,                          // [in]     Upper threshold for edges detection
    const IwiFilterCannyDerivParams &auxParams  = IwiFilterCannyDerivParams()   // [in]     Reference to the auxiliary parameters structure
)
{
    IppStatus ippStatus = ::iwiFilterCannyDeriv(&srcImageDx, &srcImageDy, &dstImage, treshLow, treshHigh, &auxParams);
    OWN_ERROR_CHECK(ippStatus)
    return ippStatus;
}

/**/////////////////////////////////////////////////////////////////////////////
//                   iwiFilterMorphology
///////////////////////////////////////////////////////////////////////////// */

using ::IwiMorphologyType;
using ::iwiMorphErode;
using ::iwiMorphDilate;
using ::iwiMorphOpen;
using ::iwiMorphClose;
using ::iwiMorphTophat;
using ::iwiMorphBlackhat;
using ::iwiMorphGradient;

// Auxiliary parameters class
// C API descriptions has more details
class IwiFilterMorphologyParams: public ::IwiFilterMorphologyParams
{
public:
    IW_BASE_PARAMS_CONSTRUCTORS(IwiFilterMorphologyParams, iwiFilterMorphology_SetDefaultParams)
    IwiFilterMorphologyParams(int _iterations = 1)
    {
        this->iterations = _iterations;
    }
};

// Performs morphology filter operation on given image ROI
// C API descriptions has more details.
// Throws:
//      ippStsBorderErr                     border value is illegal
//      ippStsDataTypeErr                   data type is illegal
//      ippStsNumChannelsErr                channels value is illegal
//      ippStsNotEvenStepErr                step value is not divisible by size of elements
//      ippStsNotSupportedModeErr           selected function mode is not supported
//      ippStsNoMemErr                      failed to allocate memory
//      ippStsInplaceModeNotSupportedErr    doesn't support output into the source buffer
//      ippStsNullPtrErr                    unexpected NULL pointer
// Returns:
//      ippStsNoErr                         no errors
IW_DECL_CPP(IppStatus) iwiFilterMorphology(
    const IwiImage                  &srcImage,                                  // [in]     Reference to the source image
    IwiImage                        &dstImage,                                  // [in,out] Reference to the destination image
    IwiMorphologyType                morphType,                                 // [in]     Morphology filter type
    const IwiImage                  &maskImage,                                 // [in]     Reference to the morphology mask image
    const IwiFilterMorphologyParams &auxParams  = IwiFilterMorphologyParams(),  // [in]     Reference to the auxiliary parameters structure
    const IwiBorderType             &border     = ippBorderDefault,             // [in]     Extrapolation algorithm and value for out of image pixels: ippBorderDefault, ippBorderConst, ippBorderRepl, ippBorderMirror, ippBorderInMem
    const IwiTile                   &tile       = IwiTile()                     // [in]     Reference to the IwiTile structure for tiling. By default no tiling is used
)
{
    IppStatus ippStatus = ::iwiFilterMorphology(&srcImage, &dstImage, morphType, &maskImage, &auxParams, border.m_type, border.m_value, &tile);
    OWN_ERROR_CHECK(ippStatus)
    return ippStatus;
}

// Calculates border size for morphology operation
// Throws:
//      ippStsNotSupportedModeErr           selected function mode is not supported
//      ippStsNullPtrErr                    unexpected NULL pointer
// Returns:
//      Size of border
IW_DECL_CPP(IwiBorderSize) iwiFilterMorphology_GetBorderSize(
    IwiMorphologyType   morphType,      // [in]  Morphology filter type
    IwiSize             maskSize        // [in]  Size of morphology mask
)
{
    IwiBorderSize borderSize;
    IppStatus ippStatus = ::iwiFilterMorphology_GetBorderSize(morphType, maskSize, &borderSize);
    OWN_ERROR_CHECK_THROW_ONLY(ippStatus)
    return borderSize;
}

/**/////////////////////////////////////////////////////////////////////////////
//                   iwiFilterBilateral
///////////////////////////////////////////////////////////////////////////// */

// Auxiliary parameters class
// C API descriptions has more details
class IwiFilterBilateralParams: public ::IwiFilterBilateralParams
{
public:
    IW_BASE_PARAMS_CONSTRUCTORS(IwiFilterBilateralParams, iwiFilterBilateral_SetDefaultParams)
    IwiFilterBilateralParams(IppiFilterBilateralType _filter = ippiFilterBilateralGauss,
        IppiDistanceMethodType _distMethod = ippDistNormL1)
    {
        this->distMethod = _distMethod;
        this->filter     = _filter;
    }
};

// Performs bilateral filtering of an image
// C API descriptions has more details.
// Throws:
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
// Returns:
//      ippStsNoErr                         no errors
IW_DECL_CPP(IppStatus) iwiFilterBilateral(
    const IwiImage                 &srcImage,                                   // [in]     Reference to the source image
    IwiImage                       &dstImage,                                   // [in,out] Reference to the destination image
    int                             radius,                                     // [in]     Radius of circular neighborhood what defines pixels for calculation
    Ipp32f                          valSquareSigma,                             // [in]     Square of Sigma for factor function for pixel intensity
    Ipp32f                          posSquareSigma,                             // [in]     Square of Sigma for factor function for pixel position
    const IwiFilterBilateralParams &auxParams   = IwiFilterBilateralParams(),   // [in]     Reference to the auxiliary parameters structure
    const IwiBorderType            &border      = ippBorderRepl,                // [in]     Extrapolation algorithm and value for out of image pixels: ippBorderConst, ippBorderRepl, ippBorderMirror, ippBorderInMem
    const IwiTile                  &tile        = IwiTile()                     // [in]     Reference to the IwiTile structure for tiling. By default no tiling is used
)
{
    IppStatus ippStatus = ::iwiFilterBilateral(&srcImage, &dstImage, radius, valSquareSigma, posSquareSigma, &auxParams, border.m_type, border.m_value, &tile);
    OWN_ERROR_CHECK(ippStatus)
    return ippStatus;
}

}

#endif
