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

#if !defined( __IPP_IW_IMAGE__ )
#define __IPP_IW_IMAGE__

#include "iw/iw_core.h"

#ifdef __cplusplus
extern "C" {
#endif

/* /////////////////////////////////////////////////////////////////////////////
//                   Image IW definitions
///////////////////////////////////////////////////////////////////////////// */

typedef IppiSizeL      IwiSize;
typedef IppiRectL      IwiRoi;
typedef IppiPointL     IwiPoint;
typedef IppiBorderType IwiBorderType;

typedef struct {
    IwSize left;
    IwSize top;
    IwSize right;
    IwSize bottom;
} IwiBorderSize;

// Special channels descriptor
// These codes specify how to process non-standard channels configurations
// If the descriptor is not supported, then error will be returned.
// If the descriptor is for different channels number, then it will be ignored.
typedef enum _IwiChDescriptor
{
    iwiChDesc_None    = 0,          // Process all channels

    // C4 descriptors
    iwiChDesc_C4M1110 = 0x00004007, // Process only the first 3 channels as RGB. Equivalent of AC4 functions from the main Intel(R) IPP library.
    iwiChDesc_C4M1000 = 0x00004001, // Process only the first channel as Gray-scale.
    iwiChDesc_C4M1001 = 0x00004009, // Process only the first channel and the last channel as Gray-scale with Alpha.
    iwiChDesc_C4M1XX0 = 0x00064001, // Process only the first channel as Gray-scale and replicate it to remaining color channels.
    iwiChDesc_C4M1XX1 = 0x00064009  // Process only the first channel and the last channel as Gray-scale with Alpha and replicate Gray-scale to remaining color channels.

} IwiChDescriptor;


/* /////////////////////////////////////////////////////////////////////////////
//                   Image IW utility functions
///////////////////////////////////////////////////////////////////////////// */

// Convert IppiMaskSize enumerator to actual IwiSize size
// Returns:
//      Width and height of IppiMaskSize in pixels
IW_DECL(IwiSize) iwiMaskToSize(
    IppiMaskSize mask    // Kernel or mask size enumerator
);

// Convert kernel or mask size to border size
// Returns:
//      Border required for a filter with specified kernel size
static IW_INLINE IwiBorderSize iwiSizeToBorderSize(
    IwiSize kernelSize   // Size of kernel as from iwiMaskToSize() or arbitrary
)
{
    IwiBorderSize bordSize;
    bordSize.left = bordSize.right  = kernelSize.width/2;
    bordSize.top  = bordSize.bottom = kernelSize.height/2;
    return bordSize;
}

// Converts symmetric kernel or mask length to border size
// Returns:
//      Border required for a filter with specified kernel length
static IW_INLINE IwiBorderSize iwiSizeSymToBorderSize(
    IwSize kernelSize   // Size of symmetric kernel
)
{
    IwiBorderSize bordSize;
    bordSize.left = bordSize.right  = kernelSize/2;
    bordSize.top  = bordSize.bottom = kernelSize/2;
    return bordSize;
}

// Shift pointer to specific pixel coordinates
// Returns:
//      Shifted pointer
static IW_INLINE void* iwiShiftPtr(
    const void *pPtr,       // Original pointer
    IwSize      step,       // Image step
    int         typeSize,   // Size of image type as from iwTypeToLen()
    int         channels,   // Number of channels in image
    IwSize      y,          // y shift, as rows
    IwSize      x           // x shift, as columns
)
{
    return (((Ipp8u*)pPtr) + step*y + typeSize*channels*x);
}

// Shift pointer to specific pixel coordinates for read-only image
// Returns:
//      Shifted pointer
static IW_INLINE const void* iwiShiftPtrConst(
    const void *pPtr,       // Original pointer
    IwSize      step,       // Image step
    int         typeSize,   // Size of image type as from iwTypeToLen()
    int         channels,   // Number of channels in image
    IwSize      y,          // y shift, as rows
    IwSize      x           // x shift, as columns
)
{
    return (((const Ipp8u*)pPtr) + step*y + typeSize*channels*x);
}


/* /////////////////////////////////////////////////////////////////////////////
//                   IwiImage - Image structure
///////////////////////////////////////////////////////////////////////////// */

// IwiImage is a base structure for IW image processing functions to store input and output data.
typedef struct _IwiImage
{
// public:
    void           *m_ptr;          // Pointer to the start of actual image data. This pointer must be NULL for read-only image.
    const void     *m_ptrConst;     // Pointer to the start of actual read-only image data. This pointer is valid for any image.
    IwSize          m_step;         // Distance, in bytes, between the starting points of consecutive lines in the source image memory
    IwiSize         m_size;         // Image size, in pixels
    IppDataType     m_dataType;     // Image pixel type
    int             m_typeSize;     // Size of image pixel type in bytes
    int             m_channels;     // Number of image channels
    IwiBorderSize   m_inMemSize;    // Memory border size around image data

// private:
    void           *m_pBuffer;      // Pointer to the allocated image buffer. This variable must be NULL for any external buffer.

} IwiImage;

// Initializes image structure with external buffer
// Returns:
//      ippStsDataTypeErr                   data type is illegal
//      ippStsNumChannelsErr                channels value is illegal
//      ippStsSizeErr                       size fields values are illegal
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL(IppStatus) iwiImage_InitExternal(
    IwiImage               *pImage,         // Pointer to IwiImage structure
    IwiSize                 size,           // Image size, in pixels, without border
    IppDataType             dataType,       // Image pixel type
    int                     channels,       // Number of image channels
    const IwiBorderSize    *pInMemBorder,   // Size of border around image or NULL if there is no border
    void                   *pBuffer,        // Pointer to the external buffer image buffer
    IwSize                  step            // Distance, in bytes, between the starting points of consecutive lines in the external buffer
);

// Initializes image structure with external read-only buffer
// Returns:
//      ippStsDataTypeErr                   data type is illegal
//      ippStsNumChannelsErr                channels value is illegal
//      ippStsSizeErr                       size fields values are illegal
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL(IppStatus) iwiImage_InitExternalConst(
    IwiImage               *pImage,         // Pointer to IwiImage structure
    IwiSize                 size,           // Image size, in pixels, without border
    IppDataType             dataType,       // Image pixel type
    int                     channels,       // Number of image channels
    const IwiBorderSize    *pInMemBorder,   // Size of border around image or NULL if there is no border
    const void             *pBuffer,        // Pointer to the external buffer image buffer
    IwSize                  step            // Distance, in bytes, between the starting points of consecutive lines in the external buffer
);

// Resets image structure values. This functions doesn't release data!
IW_DECL(void) iwiImage_Init(
    IwiImage *pImage
);

// Allocates image data for initialized structure. iwiImage_Init must be called once before.
// Returns:
//      ippStsDataTypeErr                   data type is illegal
//      ippStsNumChannelsErr                channels value is illegal
//      ippStsSizeErr                       size fields values are illegal
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL(IppStatus) iwiImage_Alloc(
    IwiImage               *pImage,         // Pointer to IwiImage structure
    IwiSize                 size,           // Image size, in pixels, without border
    IppDataType             dataType,       // Image pixel type
    int                     channels,       // Number of image channels
    const IwiBorderSize    *pInMemBorder    // Size of border around image or NULL if there is no border
);

// Releases image data if it was allocated by iwiImage_Alloc
IW_DECL(void) iwiImage_Release(
    IwiImage   *pImage      // Pointer to IwiImage structure
);

// Returns pointer to specified pixel position in image buffer
// Returns:
//      Pointer to the image data
IW_DECL(void*)     iwiImage_GetPtr(
    const IwiImage *pImage, // Pointer to IwiImage structure
    IwSize          y,      // y shift, as rows
    IwSize          x,      // x shift, as columns
    int             ch      // channels shift
);

// Returns pointer to specified pixel position in read-only image buffer
// Returns:
//      Pointer to the image data
IW_DECL(const void*) iwiImage_GetPtrConst(
    const IwiImage *pImage, // Pointer to IwiImage structure
    IwSize          y,      // y shift, as rows
    IwSize          x,      // x shift, as columns
    int             ch      // channels shift
);

// Add border size to current inMem image border, making image size smaller. Resulted image cannot be smaller than 1x1 pixels
// Returns:
//      ippStsSizeErr                       ROI size is illegal
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL(IppStatus) iwiImage_BorderAdd(
    IwiImage       *pImage,     // Pointer to IwiImage structure
    IwiBorderSize   borderSize  // Size of border
);

// Subtracts border size from current inMem image border, making image size bigger. Resulted border cannot be lesser than 0
// Returns:
//      ippStsOutOfRangeErr                 ROI is out of image
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL(IppStatus) iwiImage_BorderSub(
    IwiImage       *pImage,     // Pointer to IwiImage structure
    IwiBorderSize   borderSize  // Size of border
);

// Set border size to current inMem image border, adjusting image size. Resulted image cannot be smaller than 1x1 pixels
// Returns:
//      ippStsSizeErr                       ROI size is illegal
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL(IppStatus) iwiImage_BorderSet(
    IwiImage       *pImage,     // Pointer to IwiImage structure
    IwiBorderSize   borderSize  // Size of border
);

// Applies ROI to the current image by adjusting size and starting point of the image. Can be applied recursively.
// This function saturates ROIs which step outside of the image border.
// If ROI has no intersection with the image then resulted image size will be 0x0
// Returns:
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL(IppStatus) iwiImage_RoiSet(
    IwiImage       *pImage, // Pointer to IwiImage structure
    IwiRoi          roi     // ROI rectangle of the required sub-image
);

// Returns sub-image with size and starting point of the specified ROI. Can be applied recursively.
// See iwiImage_RoiSet
// Returns:
//      IwiImage structure of sub-image
IW_DECL(IwiImage) iwiImage_GetRoiImage(
    const IwiImage *pImage, // Pointer to IwiImage structure
    IwiRoi          roi     // ROI rectangle of the required sub-image
);

/* /////////////////////////////////////////////////////////////////////////////
//                   IW Tiling
///////////////////////////////////////////////////////////////////////////// */

/* /////////////////////////////////////////////////////////////////////////////
//                   Manual tiling control
///////////////////////////////////////////////////////////////////////////// */

// Returns border with proper ippBorderInMem flags for current tile position, image size and border size
// Returns:
//      Border type with proper ippBorderInMem flags
IW_DECL(IwiBorderType) iwiTile_GetTileBorder(
    IwiRoi          roi,            // [in]     Tile position and size
    IwiBorderType   border,         // [in]     Initial border type
    IwiBorderSize   borderSize,     // [in]     Border size
    IwiSize         srcImageSize    // [in]     Source image size
);

// Returns minimal acceptable tile size for the current border size and type
// Returns:
//      Minimal tile size
IW_DECL(IwiSize) iwiTile_GetMinTileSize(
    IwiBorderType   border,     // [in]     Border type
    IwiBorderSize   borderSize  // [in]     Border size
);

/* Function corrects ROI position and size to prevent overlapping between filtering function border and image border in
// case of border reconstruction. If image already has a right or a bottom border in memory and border type flags
// ippBorderInMemRight or ippBorderInMemBottom were specified accordingly then no correction is required.
//
// Overlapping example:
//                      image border
//                      /
// |-------------------|
// | image {  [      ]~|~}     ~ - pixels of the tile border which overlaps the image border.
// |       {  [      ]~|~}         One pixel of the tile border is inside the image, the other is outside
// |       {  [ tile ]~|~}
// |       {  [      ]~|~}
// |-------------------|  \
//                        tile border (2px)
//
// Assumption 1: processing of some pixels can be delayed. If your program expects EXACT same result as specified
// tile parameters demand then you should not use this function.
// Assumption 2: tile size for a function is not less than the maximum border size (use function iwiTile_GetMinTileSize)
//
// To prevent borders overlapping this function changes the tile according to following logic (keeping assumptions in mind):
// 1. If the "right" tile border overlaps "right" image border, then function decreases tile size to move
//    whole border inside the image.
//
// Corrected overlapping:
//                       image border
//                       /
// |--------------------|
// | image {  [     ]  }|
// |       {  [     ]  }|
// |       {  [ tile]  }|
// |       {  [     ]  }|
// |--------------------\
//                       tile border
//
// 2. Now we need to compensate right adjacent tile. So if the "left" tile border is located in the overlapping zone of
//    the "right" image boundary, then the function assumes that the previous step was taken and changes tile position and
//    size to process all remaining input
//
// Before compensation:                     After compensation (now missing pixels are inside tile ROI):
//                      image border                              image border
//                      /                                         /
// |--------------------|                   |--------------------|
// | image        { ~[ ]|  }                | image       {  [~ ]|  }
// |              { ~[ ]|  }                |             {  [~ ]|  }
// |              { ~[ ]|  }                |             {  [~ ]|  }
// |              { ~[ ]|  }                |             {  [~ ]|  }
// |---------------\----|                   |--------------\-----|
//                 tile border                             tile border
//  ~ - missing pixels after step 1
*/
// Returns:
//      Corrected ROI
IW_DECL(IwiRoi) iwiTile_CorrectBordersOverlap(
    IwiRoi          roi,            // [in]     Tile position and size to be checked and corrected
    IwiBorderType   border,         // [in]     Border type
    IwiBorderSize   borderSize,     // [in]     Border size
    IwiSize         srcImageSize    // [in]     Source image size
);

/* /////////////////////////////////////////////////////////////////////////////
//                   IwiTile - tiling structure
///////////////////////////////////////////////////////////////////////////// */

// Function pointer type for return of src ROI by dst ROI
// Returns:
//      0 if operation is successful, any other if failed
typedef int (IPP_STDCALL *IwiTile_GetSrcRoiFunPtr)(
    IwiRoi     dstRoi,  // [in]     Destination ROI for transform operation
    IwiRoi    *pSrcRoi, // [in,out] Output source ROI for transform operation
    void*      pParams  // [in]     Parameters of transform operation
);

// Tile geometric transform structure
// This structure contains function pointers and parameters which are necessary for tile geometric transformations inside the pipeline
typedef struct _IwiTileTransform
{
// public:
    IwiTile_GetSrcRoiFunPtr  getSrcRoiFun;  // Pointer to IwiRoiRectTransformFunctionPtr function which returns source ROI for the current destination one
    void                    *pParams;       // Pointer to user parameters for transform functions

    IwiSize                  srcImageSize;  // Image size before transformation

} IwiTileTransform;

// Main structure for semi-automatic ROI operations
// This structure provides main context for tiling across IW API
// Mainly it contains values for complex pipelines tiling
typedef struct _IwiTile
{
// private:
    IwiRoi            m_srcRoi;            // Absolute ROI for the source image
    IwiRoi            m_dstRoi;            // Absolute ROI for the destination image

    IwiPoint          m_untaintSrcPos;     // Absolute unaligned source ROI position
    IwiPoint          m_untaintDstPos;     // Absolute unaligned destination ROI position

    IwiRoi            m_boundSrcRoi;       // Relative ROI for the source image bounded to the buffer
    IwiRoi            m_boundDstRoi;       // Relative ROI for the destination image bounded to the buffer

    IwiSize           m_srcBufferSize;     // Actual source buffer size
    IwiSize           m_dstBufferSize;     // Actual destination buffer size

    IwiSize           m_srcImageSize;      // Full source image size
    IwiSize           m_dstImageSize;      // Full destination image size

    IwiSize           m_srcExImageSize;    // Source image size extended on parent InMem border size
    IwiSize           m_dstExImageSize;    // Destination image size extended on parent InMem border size

    IwiSize           m_maxTileSize;       // Maximum tile size

    IwiBorderType     m_borderType;        // Type of source image border
    IwiBorderSize     m_borderSize;        // Border required for the current operation
    IwiBorderSize     m_borderSizeAcc;     // Accumulated border size for current and parent operations
    IwiBorderSize     m_externalMem;       // Amount of memory required to process InMem border for current edge tile
    IwiBorderSize     m_externalMemAcc;    // Amount of memory required to process InMem border for all edge tiles

    IwiTileTransform  m_transformStruct;   // Transformation proxy functions and data structure

    int               m_initialized;       // Internal initialization states

    struct _IwiTile  *m_pChild;            // Next Tile in the pipeline
    struct _IwiTile  *m_pParent;           // Previous Tile in the pipeline

} IwiTile;

/* /////////////////////////////////////////////////////////////////////////////
//                   IwiTile-based basic tiling
///////////////////////////////////////////////////////////////////////////// */

// Basic tiling initializer for IwiTile structure.
// Use this method to set up single function tiling or tiling for pipelines with border-less functions.
// For functions which operate with different sizes for source and destination images use destination size as a base
// for tile parameters.
// Returns:
//      Valid IwiTile structure for simple tiling
IW_DECL(IwiTile) iwiTile_SetRoi(
    IwiRoi   tileRoi     // [in] Tile offset and size
);

/* /////////////////////////////////////////////////////////////////////////////
//                   IwiTile-based pipeline tiling
///////////////////////////////////////////////////////////////////////////// */

// Important notice:
// This tiling API is created for tiling of complex pipelines with functions which use borders.
// Tiling of pipelines instead of isolated functions can increase scalability of threading or performance of
// non-threaded functions by performing all operations inside CPU cache.
//
// This is advanced tiling method, so you better know what you are doing.
// 1. Pipeline tiling operates in reverse order: from destination to source.
//    a. Use tile size based on final destination image size
//    b. Initialize IwiTile structure with iwiTilePipeline_Init for the last operation
//    c. Initialize IwiTile structure for other operations from last to first with iwiTilePipeline_InitChild
// 2. Derive border size for each operation from its mask size, kernel size or specific border size getter if any
// 3. If you have geometric transform inside pipeline, fill IwiTileTransform structure for IwiTile for this transform operation
// 4. In case of threading don't forget to copy initialized IwiTile structures to local thread or initialize them on
//    per-thread basis. Access to structures is not thread safe!
// 5. Do not exceed maximum tile size specified during initialization. This can lead to buffers overflow!
//
// There is a set of examples covering usage of tiling. Please refer to them for help.
//
// Pipeline tiling with scaling is not supported in this version.

// Pipeline tiling root node initializer for IwiTile structure.
// This initializer should be used first and for IwiTile structure of the final operation.
// Returns:
//      ippStsBadArgErr                     incorrect arg/param of the function
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL(IppStatus) iwiTilePipeline_Init(
    IwiTile                *pTile,             // [in] Pointer to IwiTile structure
    IwiSize                 tileSizeMax,       // [in] Maximum tile size for intermediate buffers size calculation
    IwiSize                 dstImageSize,      // [in] Destination image size for current operation
    const IwiBorderType    *pBorderType,       // [in] Border type for the current operation. NULL if operation doesn't have a border
    const IwiBorderSize    *pBorderSize,       // [in] Border size for the current operation. NULL if operation doesn't have a border
    const IwiTileTransform *pTransformStruct   // [in] Initialized transform structure if operation performs geometric transformation. NULL if operation doesn't perform transformation
);

// Pipeline tiling child node initializer for IwiTile structure.
// This initializer should be called for any operation preceding the last operation in reverse order.
// Returns:
//      ippStsBadArgErr                     incorrect arg/param of the function
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL(IppStatus) iwiTilePipeline_InitChild(
    IwiTile                *pTile,             // [in] Pointer to IwiTile structure
    IwiTile                *pParent,           // [in] Pointer to IwiTile structure of previous operation
    const IwiBorderType    *pBorderType,       // [in] Border type for the current operation. NULL if operation doesn't have a border
    const IwiBorderSize    *pBorderSize,       // [in] Border size for the current operation. NULL if operation doesn't have a border
    const IwiTileTransform *pTransformStruct   // [in] Initialized transform structure if operation performs geometric transformation. NULL if operation doesn't perform transformation
);

// Releases allocated data from the pipeline tiling structure.
IW_DECL(void) iwiTilePipeline_Release(
    IwiTile *pTile  // [in] Pointer to IwiTile structure
);

// Returns buffer size required to store destination intermediate image for the current pipeline element.
// Returns:
//      ippStsContextMatchErr               internal structure is not initialized or of invalid type
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL(IppStatus) iwiTilePipeline_GetDstBufferSize(
    const IwiTile   *pTile,         // [in]     Pointer to IwiTile structure
    IwiSize         *pDstSize       // [out]    Minimal required size of destination intermediate buffer
);

// Returns full size of source image for the child pipeline element which includes required InMem borders.
// This function is required to supply correct image size for geometric transform functions.
// Returns:
//      ippStsContextMatchErr               internal structure is not initialized or of invalid type
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL(IppStatus) iwiTilePipeline_GetChildSrcImageSize(
    const IwiTile   *pTile,         // [in]     Pointer to IwiTile structure
    IwiSize          srcOrigSize,   // [in]     Original source image size
    IwiSize         *pSrcFullSize   // [out]    Pointer to IwiSize structure to write full image size
);

// Returns full size of destination image for the child pipeline element which includes required InMem borders.
// This function is required to supply correct image size for geometric transform functions.
// Returns:
//      ippStsContextMatchErr               internal structure is not initialized or of invalid type
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL(IppStatus) iwiTilePipeline_GetChildDstImageSize(
    const IwiTile   *pTile,         // [in]     Pointer to IwiTile structure
    IwiSize          dstOrigSize,   // [in]     Original destination image size
    IwiSize         *pDstFullSize   // [out]    Pointer to IwiSize structure to write full image size
);

// Sets current tile rectangle for the pipeline to process
// Returns:
//      ippStsContextMatchErr               internal structure is not initialized or of invalid type
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL(IppStatus) iwiTilePipeline_SetRoi(
    IwiTile         *pTile,         // [in] Pointer to IwiTile structure
    IwiRoi           tileRoi        // [in] Tile offset and size
);

// This function builds border for the current tile source buffer.
// This allows to feed function with InMem borders only thus reducing possiblity of borders conflicts on image boundary.
// By default this function is not applied to the first image in the pipeline, only to intermediate buffers, but
// it can be used manually to construct border for it too.
// Returns:
//      ippStsContextMatchErr               internal structure is not initialized or of invalid type
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL(IppStatus) iwiTilePipeline_BuildBorder(
    const IwiTile   *pTile,          // [in]     Pointer to IwiTile structure for current tile
    IwiImage        *pSrcImage,      // [in,out] Pointer to the source image for which to build border
    IwiBorderType   *pBorder,        // [in,out] Extrapolation algorithm for out of image pixels. Updated InMem flags will be returned here
    const Ipp64f    *pBorderVal      // [in]     Pointer to array of border values for ippBorderConst. One element for each channel. Can be NULL for any other border
);

// Calculates actual border parameter with InMem flags for the current tile absolute and relative offsets and sizes
// Returns:
//      ippStsContextMatchErr               internal structure is not initialized or of invalid type
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL(IppStatus) iwiTilePipeline_GetTileBorder(
    const IwiTile   *pTile,         // [in]     Pointer to IwiTile structure
    IwiBorderType   *pBorder        // [in,out] Pointer to border type, actual tile border will be written here
);

// Checks for image and buffer boundaries for the source buffer and limits tile rectangle
// Returns:
//      ippStsContextMatchErr               internal structure is not initialized or of invalid type
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL(IppStatus) iwiTilePipeline_GetBoundedSrcRoi(
    const IwiTile   *pTile,         // [in]     Pointer to IwiTile structure
    IwiRoi          *pBoundedRoi    // [out]    Pointer to ROI adjusted to source buffer boundaries
);

// Checks for image and buffer boundaries for the destination buffer and limits tile rectangle
// Returns:
//      ippStsContextMatchErr               internal structure is not initialized or of invalid type
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL(IppStatus) iwiTilePipeline_GetBoundedDstRoi(
    const IwiTile   *pTile,         // [in]     Pointer to IwiTile structure
    IwiRoi          *pBoundedRoi    // [out]    Pointer to ROI adjusted to destination buffer boundaries
);

// Returns minimal acceptable tile size for current pipeline
// Returns:
//      ippStsContextMatchErr               internal structure is not initialized or of invalid type
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL(IppStatus) iwiTilePipeline_GetMinTileSize(
    const IwiTile   *pTile,        // [in]      Pointer to IwiTile structure
    IwiSize         *pMinTileSize  // [out]     Pointer to the minimal tile size for current pipeline
);

#ifdef __cplusplus
}
#endif

#endif
