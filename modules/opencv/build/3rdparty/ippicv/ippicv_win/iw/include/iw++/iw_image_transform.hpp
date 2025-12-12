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

#if !defined( __IPP_IWPP_IMAGE_TRANSFORM__ )
#define __IPP_IWPP_IMAGE_TRANSFORM__

#include "iw/iw_image_transform.h"
#include "iw++/iw_image.hpp"

namespace ipp
{

/**/////////////////////////////////////////////////////////////////////////////
//                   iwiMirror
///////////////////////////////////////////////////////////////////////////// */

// Auxiliary parameters class
// C API descriptions has more details
class IwiMirrorParams: public ::IwiMirrorParams
{
public:
    IW_BASE_PARAMS_CONSTRUCTORS(IwiMirrorParams, iwiMirror_SetDefaultParams)
    IwiMirrorParams(IwiChDescriptor _chDesc = iwiChDesc_None)
    {
        this->chDesc = _chDesc;
    }
};

// Mirrors image around specified axis.
// For ippAxs45 and ippAxs135 destination image must have flipped size: dstWidth = srcHeight, dstHeight = srcWidth.
// C API descriptions has more details.
// Throws:
//      ippStsDataTypeErr                   data type is illegal
//      ippStsNumChannelsErr                channels value is illegal
//      ippStsNotSupportedModeErr           selected function mode is not supported
//      ippStsInplaceModeNotSupportedErr    doesn't support output into the source buffer. In tiling mode
//      ippStsNullPtrErr                    unexpected NULL pointer
// Returns:
//      ippStsNoErr                         no errors
IW_DECL_CPP(IppStatus) iwiMirror(
    const IwiImage         &srcImage,                           // [in]     Reference to the source image
    IwiImage               &dstImage,                           // [in,out] Reference to the destination image
    IppiAxis                axis,                               // [in]     Mirror axis
    const IwiMirrorParams  &auxParams   = IwiMirrorParams(),    // [in]     Reference to the auxiliary parameters structure
    const IwiTile          &tile        = IwiTile()             // [in]     Reference to the IwiTile structure for tiling. By default no tiling is used
)
{
    IppStatus ippStatus = ::iwiMirror(&srcImage, &dstImage, axis, &auxParams, &tile);
    OWN_ERROR_CHECK(ippStatus)
    return ippStatus;
}

// Calculates source ROI by destination one
// Throws:
//      ippStsNotSupportedModeErr           selected function mode is not supported
//      ippStsNullPtrErr                    unexpected NULL pointer
// Returns:
//      Source image ROI
IW_DECL_CPP(IwiRoi) iwiMirror_GetSrcRoi(
    IppiAxis        axis,       // [in]  Mirror axis
    IwiSize         dstSize,    // [in]  Size of destination image in pixels
    IwiRoi          dstRoi      // [in]  Destination image ROI
)
{
    IwiRoi    srcRoi;
    IppStatus ippStatus = ::iwiMirror_GetSrcRoi(axis, dstSize, dstRoi, &srcRoi);
    OWN_ERROR_CHECK_THROW_ONLY(ippStatus)
    return srcRoi;
}

// Calculates destination image size for iwiMirror function.
// Throws:
//      ippStsBadArgErr                     incorrect arg/param of the function
// Returns:
//      Destination size
IW_DECL_CPP(IwiSize) iwiMirror_GetDstSize(
    IwiSize         srcSize,    // [in]  Size of the source image in pixels
    IppiAxis        axis        // [in]  Angle of clockwise rotation in degrees
)
{
    IwiSize   dstSize;
    IppStatus ippStatus = ::iwiMirror_GetDstSize(srcSize, axis, &dstSize);
    OWN_ERROR_CHECK_THROW_ONLY(ippStatus);
    return dstSize;
}

/**/////////////////////////////////////////////////////////////////////////////
//                   iwiRotate
///////////////////////////////////////////////////////////////////////////// */

// Auxiliary parameters class
// C API descriptions has more details
class IwiRotateParams: public ::IwiRotateParams
{
public:
    IW_BASE_PARAMS_CONSTRUCTORS(IwiRotateParams, iwiRotate_SetDefaultParams)
    IwiRotateParams() {}
};

// Performs rotation of the image around (0,0).
// This is a simplified version of iwiWarpAffine function designed specifically for rotation.
// C API descriptions has more details.
// Throws:
//      ippStsInterpolationErr              interpolation value is illegal
//      ippStsDataTypeErr                   data type is illegal
//      ippStsNumChannelsErr                channels value is illegal
//      ippStsBorderErr                     border value is illegal
//      ippStsNotSupportedModeErr           selected function mode is not supported
//      ippStsNoMemErr                      failed to allocate memory
//      ippStsSizeErr                       size fields values are illegal
//      ippStsInplaceModeNotSupportedErr    doesn't support output into the source buffer
//      ippStsNullPtrErr                    unexpected NULL pointer
// Returns:
//      ippStsNoErr                         no errors
//      ippStsNoOperation                   [warning] width or height of an image is zero
//      ippStsWrongIntersectQuad            [warning] transformed source image has no intersection with the destination image
IW_DECL_CPP(IppStatus) iwiRotate(
    const IwiImage         &srcImage,                           // [in]     Reference to the source image
    IwiImage               &dstImage,                           // [in,out] Reference to the destination image
    double                  angle,                              // [in]     Angle of clockwise rotation in degrees
    IppiInterpolationType   interpolation,                      // [in]     Interpolation method: ippNearest, ippLinear, ippCubic
    const IwiRotateParams  &auxParams   = IwiRotateParams(),    // [in]     Reference to the auxiliary parameters structure
    const IwiBorderType    &border      = ippBorderTransp,      // [in]     Extrapolation algorithm and value for out of image pixels: ippBorderConst, ippBorderRepl, ippBorderTransp, ippBorderInMem
    const IwiTile          &tile        = IwiTile()             // [in]     Reference to the IwiTile object for tiling. By default no tiling is used
)
{
    IppStatus ippStatus = ::iwiRotate(&srcImage, &dstImage, angle, interpolation, &auxParams, border.m_type, border.m_value, &tile);
    OWN_ERROR_CHECK(ippStatus)
    return ippStatus;
}

// Calculates destination image size for iwiRotate function.
// This is a more simple version of iwiWarpAffine function designed specifically for rotation.
// Throws:
//      ippStsErr                           size calculation error
// Returns:
//      Size of rotated image boundaries
IW_DECL_CPP(IwiSize) iwiRotate_GetDstSize(
    IwiSize         srcSize,    // [in]  Size of the source image in pixels
    double          angle       // [in]  Angle of clockwise rotation in degrees
)
{
    IwiSize   size;
    IppStatus ippStatus = ::iwiRotate_GetDstSize(srcSize, angle, &size);
    OWN_ERROR_CHECK_THROW_ONLY(ippStatus);
    return size;
}

/* /////////////////////////////////////////////////////////////////////////////
//                   IwiResize
///////////////////////////////////////////////////////////////////////////// */

// Auxiliary parameters class
// C API descriptions has more details
class IwiResizeParams: public ::IwiResizeParams
{
public:
    IW_BASE_PARAMS_CONSTRUCTORS(IwiResizeParams, iwiResize_SetDefaultParams)
    IwiResizeParams(Ipp32u _antialiasing = 0, Ipp32f _cubicBVal = 1, Ipp32f _cubicCVal = 0, Ipp32u _lanczosLobes = 3)
    {
        this->antialiasing = _antialiasing;
        this->cubicBVal    = _cubicBVal;
        this->cubicCVal    = _cubicCVal;
        this->lanczosLobes = _lanczosLobes;
    }

    // Constructor for ippCubic
    IwiResizeParams(Ipp32f _cubicBVal, Ipp32f _cubicCVal, Ipp32u _antialiasing)
    {
        iwiResize_SetDefaultParams(this);
        this->antialiasing = _antialiasing;
        this->cubicBVal    = _cubicBVal;
        this->cubicCVal    = _cubicCVal;
    }

    // Constructor for ippLanczos
    IwiResizeParams(Ipp32u _lanczosLobes, Ipp32u _antialiasing)
    {
        iwiResize_SetDefaultParams(this);
        this->antialiasing = _antialiasing;
        this->lanczosLobes = _lanczosLobes;
    }
};

// Simplified version of resize function without spec structure and initialization
// C API descriptions has more details.
// Throws:
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
// Returns:
//      ippStsNoErr                         no errors
//      ippStsNoOperation                   [warning] width or height of an image is zero
IW_DECL_CPP(IppStatus) iwiResize(
    const IwiImage         &srcImage,                           // [in]     Reference to the source image
    IwiImage               &dstImage,                           // [in,out] Reference to the destination image
    IppiInterpolationType   interpolation,                      // [in]     Interpolation method: ippNearest, ippLinear, ippCubic, ippLanczos, ippSuper
    const IwiResizeParams  &auxParams   = IwiResizeParams(),    // [in]     Reference to the auxiliary parameters structure
    const IwiBorderType    &border      = ippBorderRepl,        // [in]     Extrapolation algorithm and value for out of image pixels: ippBorderConst, ippBorderRepl, ippBorderMirror, ippBorderInMem
    const IwiTile          &tile        = IwiTile()             // [in]     Reference to the IwiTile structure for tiling. By default no tiling is used
)
{
    IppStatus ippStatus = ::iwiResize(&srcImage, &dstImage, interpolation, &auxParams, border.m_type, border.m_value, &tile);
    OWN_ERROR_CHECK(ippStatus)
    return ippStatus;
}

// Resize operation class
// C API descriptions has more details.
class IwiResize
{
public:
    // Default constructor
    IwiResize()
    {
        m_initialized = false;
    }

    // Constructor with initialization
    // Throws:
    //      ippStsInterpolationErr              interpolation value is illegal
    //      ippStsDataTypeErr                   data type is illegal
    //      ippStsNumChannelsErr                channels value is illegal
    //      ippStsBorderErr                     border value is illegal
    //      ippStsNotSupportedModeErr           selected function mode is not supported
    //      ippStsNoMemErr                      failed to allocate memory
    //      ippStsSizeErr                       size fields values are illegal
    //      ippStsNullPtrErr                    unexpected NULL pointer
    IwiResize(
        IwiSize                 srcSize,                            // [in] Size of the source image in pixels
        IwiSize                 dstSize,                            // [in] Size of the destination image in pixels
        IppDataType             dataType,                           // [in] Image pixel type
        int                     channels,                           // [in] Number of image channels
        IppiInterpolationType   interpolation,                      // [in] Interpolation method: ippNearest, ippLinear, ippCubic, ippLanczos, ippSuper
        const IwiResizeParams  &auxParams   = IwiResizeParams(),    // [in] Reference to the auxiliary parameters structure
        const IwiBorderType    &border      = ippBorderRepl         // [in] Extrapolation algorithm and value for out of image pixels: ippBorderConst, ippBorderRepl, ippBorderInMem
    )
    {
        m_initialized = false;

        IppStatus ippStatus = InitAlloc(srcSize, dstSize, dataType, channels, interpolation, auxParams, border);
        OWN_ERROR_CHECK_THROW_ONLY(ippStatus);
    }

    // Default destructor
    ~IwiResize()
    {
        if(m_initialized)
        {
            ::iwiResize_Free(m_pSpec);
            m_initialized = false;
        }
    }

    // Allocates and initializes internal data structure
    // Throws:
    //      ippStsInterpolationErr              interpolation value is illegal
    //      ippStsDataTypeErr                   data type is illegal
    //      ippStsNumChannelsErr                channels value is illegal
    //      ippStsBorderErr                     border value is illegal
    //      ippStsNotSupportedModeErr           selected function mode is not supported
    //      ippStsNoMemErr                      failed to allocate memory
    //      ippStsSizeErr                       size fields values are illegal
    //      ippStsNullPtrErr                    unexpected NULL pointer
    // Returns:
    //      ippStsNoErr                         no errors
    //      ippStsNoOperation                   [warning] width or height of an image is zero
    IppStatus InitAlloc(
        IwiSize                 srcSize,                            // [in] Size of the source image in pixels
        IwiSize                 dstSize,                            // [in] Size of the destination image in pixels
        IppDataType             dataType,                           // [in] Image pixel type
        int                     channels,                           // [in] Number of image channels
        IppiInterpolationType   interpolation,                      // [in] Interpolation method: ippNearest, ippLinear, ippCubic, ippLanczos, ippSuper
        const IwiResizeParams  &auxParams   = IwiResizeParams(),    // [in] Reference to the auxiliary parameters structure
        const IwiBorderType    &border      = ippBorderRepl         // [in] Extrapolation algorithm and value for out of image pixels: ippBorderConst, ippBorderRepl, ippBorderInMem
    )
    {
        if(m_initialized)
        {
            ::iwiResize_Free(m_pSpec);
            m_initialized = false;
        }

        IppStatus ippStatus = ::iwiResize_InitAlloc(&m_pSpec, srcSize, dstSize, dataType, channels, interpolation, &auxParams, border);
        OWN_ERROR_CHECK(ippStatus);

        m_initialized = true;
        return ippStatus;
    }

    // Performs resize operation on given image ROI
    // Throws:
    //      ippStsBorderErr                     border value is illegal
    //      ippStsDataTypeErr                   data type is illegal
    //      ippStsNumChannelsErr                channels value is illegal
    //      ippStsNotEvenStepErr                step value is not divisible by size of elements
    //      ippStsNotSupportedModeErr           selected function mode is not supported
    //      ippStsNoMemErr                      failed to allocate memory
    //      ippStsInplaceModeNotSupportedErr    doesn't support output into the source buffer
    //      ippStsContextMatchErr               internal structure is not initialized or of invalid type
    //      ippStsNullPtrErr                    unexpected NULL pointer
    // Returns:
    //      ippStsNoErr                         no errors
    //      ippStsNoOperation                   [warning] width or height of an image is zero
    IppStatus operator()(
        const IwiImage         &srcImage,                       // [in]     Reference to the source image
        IwiImage               &dstImage,                       // [in,out] Reference to the destination image
        const IwiBorderType    &border      = ippBorderRepl,    // [in]     Extrapolation algorithm and value for out of image pixels: ippBorderConst, ippBorderRepl, ippBorderInMem
        const IwiTile          &tile        = IwiTile()         // [in]     Reference to the IwiTile structure for tiling. By default no tiling is used
    ) const
    {
        if(m_initialized)
        {
            IppStatus ippStatus = ::iwiResize_Process(m_pSpec, &srcImage, &dstImage, border.m_type, border.m_value, &tile);
            OWN_ERROR_CHECK(ippStatus)
            return ippStatus;
        }
        else
            OWN_ERROR_THROW(ippStsContextMatchErr);
    }

    // Calculates source ROI by destination one
    // Throws:
    //      ippStsContextMatchErr               internal structure is not initialized or of invalid type
    //      ippStsNullPtrErr                    unexpected NULL pointer
    // Returns:
    //      Source image ROI
    IwiRoi GetSrcRoi(
        IwiRoi  dstRoi             // [in]  Destination image ROI
    ) const
    {
        if(m_initialized)
        {
            IwiRoi    srcRoi;
            IppStatus ippStatus = ::iwiResize_GetSrcRoi(m_pSpec, dstRoi, &srcRoi);
            OWN_ERROR_CHECK_THROW_ONLY(ippStatus)
            return srcRoi;
        }
        else
            OWN_ERROR_THROW_ONLY(ippStsContextMatchErr)
    }

    // Get border size for resize
    // Throws:
    //      ippStsContextMatchErr               internal structure is not initialized or of invalid type
    //      ippStsNullPtrErr                    unexpected NULL pointer
    // Returns:
    //      Border size
    IwiBorderSize GetBorderSize() const
    {
        if(m_initialized)
        {
            IwiBorderSize borderSize;
            IppStatus     ippStatus = ::iwiResize_GetBorderSize(m_pSpec, &borderSize);
            OWN_ERROR_CHECK_THROW_ONLY(ippStatus)
            return borderSize;
        }
        else
            OWN_ERROR_THROW_ONLY(ippStsContextMatchErr)
    }

private:
    IwiResizeSpec *m_pSpec;         // Pointer to internal spec structure
    bool           m_initialized;   // Initialization flag
};

/* /////////////////////////////////////////////////////////////////////////////
//                   IwiWarpAffine
///////////////////////////////////////////////////////////////////////////// */

// Auxiliary parameters class
// C API descriptions has more details
class IwiWarpAffineParams: public ::IwiWarpAffineParams
{
public:
    IW_BASE_PARAMS_CONSTRUCTORS(IwiWarpAffineParams, iwiWarpAffine_SetDefaultParams)
    IwiWarpAffineParams(Ipp32u _smoothEdge = 0, Ipp32f _cubicBVal = 1, Ipp32f _cubicCVal = 0)
    {
        this->cubicBVal    = _cubicBVal;
        this->cubicCVal    = _cubicCVal;
        this->smoothEdge   = _smoothEdge;
    }
};

// Simplified version of warp affine function without spec structure and initialization
// C API descriptions has more details.
// Throws:
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
// Returns:
//      ippStsNoErr                         no errors
//      ippStsNoOperation                   [warning] width or height of an image is zero
//      ippStsWrongIntersectQuad            [warning] transformed source image has no intersection with the destination image
IW_DECL_CPP(IppStatus) iwiWarpAffine(
    const IwiImage             &srcImage,                               // [in]     Reference to the source image
    IwiImage                   &dstImage,                               // [in,out] Reference to the destination image
    const double                coeffs[2][3],                           // [in]     Coefficients for the affine transform
    IwTransDirection            direction,                              // [in]     Transformation direction
    IppiInterpolationType       interpolation,                          // [in]     Interpolation method: ippNearest, ippLinear, ippCubic
    const IwiWarpAffineParams  &auxParams   = IwiWarpAffineParams(),    // [in]     Reference to the auxiliary parameters structure
    const IwiBorderType        &border      = ippBorderTransp,          // [in]     Extrapolation algorithm and value for out of image pixels: ippBorderConst, ippBorderRepl, ippBorderTransp, ippBorderInMem
    const IwiTile              &tile        = IwiTile()                 // [in]     Reference to the IwiTile object for tiling. By default no tiling is used
)
{
    IppStatus ippStatus = ::iwiWarpAffine(&srcImage, &dstImage, coeffs, direction, interpolation, &auxParams, border.m_type, border.m_value, &tile);
    OWN_ERROR_CHECK(ippStatus)
    return ippStatus;
}

// WarpAffine operation class
// C API descriptions has more details.
class IwiWarpAffine
{
public:
    // Default constructor
    IwiWarpAffine()
    {
        m_bInitialized = false;
    }

    // Constructor with initialization
    // Throws:
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
    IwiWarpAffine(
        IwiSize                     srcSize,                                // [in] Size of the source image in pixels
        IwiSize                     dstSize,                                // [in] Size of the destination image in pixels
        IppDataType                 dataType,                               // [in] Image pixel type
        int                         channels,                               // [in] Number of image channels
        const double                coeffs[2][3],                           // [in] Coefficients for the affine transform
        IwTransDirection            direction,                              // [in] Transformation direction
        IppiInterpolationType       interpolation,                          // [in] Interpolation method: ippNearest, ippLinear, ippCubic
        const IwiWarpAffineParams  &auxParams   = IwiWarpAffineParams(),    // [in] Reference to the auxiliary parameters structure
        const IwiBorderType        &border      = ippBorderTransp           // [in] Extrapolation algorithm and value for out of image pixels: ippBorderConst, ippBorderRepl, ippBorderTransp, ippBorderInMem
    )
    {
        m_bInitialized = false;

        IppStatus ippStatus = InitAlloc(srcSize, dstSize, dataType, channels, coeffs, direction, interpolation, auxParams, border);
        OWN_ERROR_CHECK_THROW_ONLY(ippStatus);
    }

    // Default destructor
    ~IwiWarpAffine()
    {
        if(m_bInitialized)
        {
            ::iwiWarpAffine_Free(m_pSpec);
            m_bInitialized = false;
        }
    }

    // Allocates and initializes internal data structure
    // Throws:
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
    // Returns:
    //      ippStsNoErr                         no errors
    //      ippStsNoOperation                   [warning] width or height of an image is zero
    //      ippStsWrongIntersectQuad            [warning] transformed source image has no intersection with the destination image
    IppStatus InitAlloc(
        IwiSize                     srcSize,                                // [in] Size of the source image in pixels
        IwiSize                     dstSize,                                // [in] Size of the destination image in pixels
        IppDataType                 dataType,                               // [in] Image pixel type
        int                         channels,                               // [in] Number of image channels
        const double                coeffs[2][3],                           // [in] Coefficients for the affine transform
        IwTransDirection            direction,                              // [in] Transformation direction
        IppiInterpolationType       interpolation,                          // [in] Interpolation method: ippNearest, ippLinear, ippCubic
        const IwiWarpAffineParams  &auxParams   = IwiWarpAffineParams(),    // [in] Reference to the auxiliary parameters structure
        const IwiBorderType        &border      = ippBorderTransp           // [in] Extrapolation algorithm and value for out of image pixels: ippBorderConst, ippBorderRepl, ippBorderTransp, ippBorderInMem
    )
    {
        if(m_bInitialized)
        {
            ::iwiWarpAffine_Free(m_pSpec);
            m_bInitialized = false;
        }

        IppStatus ippStatus = ::iwiWarpAffine_InitAlloc(&m_pSpec, srcSize, dstSize, dataType, channels, coeffs, direction, interpolation, &auxParams, border.m_type, border.m_value);
        OWN_ERROR_CHECK(ippStatus);

        m_bInitialized = true;
        return ippStatus;
    }

    // Performs warp affine operation on given image ROI
    // Throws:
    //      ippStsInterpolationErr              interpolation value is illegal
    //      ippStsDataTypeErr                   data type is illegal
    //      ippStsNumChannelsErr                channels value is illegal
    //      ippStsBorderErr                     border value is illegal
    //      ippStsNotEvenStepErr                step value is not divisible by size of elements
    //      ippStsNotSupportedModeErr           selected function mode is not supported
    //      ippStsNoMemErr                      failed to allocate memory
    //      ippStsContextMatchErr               internal structure is not initialized or of invalid type
    //      ippStsSizeErr                       size fields values are illegal
    //      ippStsNullPtrErr                    unexpected NULL pointer
    // Returns:
    //      ippStsNoErr                         no errors
    //      ippStsNoOperation                   [warning] width or height of an image is zero
    //      ippStsWrongIntersectQuad            [warning] transformed source image has no intersection with the destination image
    IppStatus operator()(
        const IwiImage &srcImage,           // [in]     Reference to the source image
        IwiImage       &dstImage,           // [in,out] Reference to the destination image
        const IwiTile  &tile    = IwiTile() // [in]     Reference to the IwiTile structure for tiling. By default no tiling is used
    ) const
    {
        if(m_bInitialized)
        {
            IppStatus ippStatus = ::iwiWarpAffine_Process(m_pSpec, &srcImage, &dstImage, &tile);
            OWN_ERROR_CHECK(ippStatus);
            return ippStatus;
        }
        else
            OWN_ERROR_THROW(ippStsBadArgErr);
    }

private:
    IwiWarpAffineSpec  *m_pSpec;        // Pointer to internal spec structure
    bool                m_bInitialized; // Initialization flag
};

}

#endif
