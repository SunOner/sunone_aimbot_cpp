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

#if !defined( __IPP_IW_IMAGE_COLOR__ )
#define __IPP_IW_IMAGE_COLOR__

#include "iw/iw_image.h"

#ifdef __cplusplus
extern "C" {
#endif

// The purpose of a color model is to facilitate the specification of colors in some standard generally accepted way.
// In essence, a color model is a specification of a 3-D coordinate system and a subspace within that system where each
// color is represented by a single point.
//
// Each industry that uses color employs the most suitable color model. For example, the RGB color model is used in
// computer graphics, YUV or YCbCr are used in video systems, PhotoYCC* is used in PhotoCD* production and so on.
// Transferring color information from one industry to another requires transformation from one set of values to
// another.

#define IWI_COLOR_IS_PLANAR(V)        ((V)&0x80000000)
#define IWI_COLOR_GET_PLANES(V)       (((V)>>24)&0xFF)
#define IWI_COLOR_GET_CHANNELS(V)     (((V)>>24)&0xFF)
#define IWI_COLOR_FORMAT(N, C)        ((N)|((C)<<24))
#define IWI_COLOR_FORMAT_PLANAR(N, P) ((N)|((P)<<24)(1<<31))

// Color formats enumerator
typedef enum _IwiColorFmt
{
    iwiColorUndefined   = IWI_COLOR_FORMAT(0x00,0),

    // Basic formats
    iwiColorGray        = IWI_COLOR_FORMAT(0x00,1),
    iwiColorRGB         = IWI_COLOR_FORMAT(0x01,3),
    iwiColorRGBA        = IWI_COLOR_FORMAT(0x02,4),
    iwiColorBGR         = IWI_COLOR_FORMAT(0x03,3),
    iwiColorBGRA        = IWI_COLOR_FORMAT(0x04,4)
} IwiColorFmt;

// Converts IwiColorFmt to number of channels
// Returns:
//      If color format is of interleaved type then returns number of channels.
//      If color format is of planar type then returns number channels for the specified plane.
//      If color format is undefined or number of planes is incorrect then returns 0.
IW_DECL(int) iwiColorToChannels(
    IwiColorFmt color,   // Color format
    int         planeNum // Number of plane for planar format, from 0. Ignored for interleaved formats
);

// Converts IwiColorFmt to number of planes required to store the image
// Returns:
//      Number of planes in the color format
//      If color format is undefined then returns 0
IW_DECL(int) iwiColorToPlanes(
    IwiColorFmt color    // Color format
);

// Calculates image size required to store particular plane for a color format
// Returns:
//      If color format is of interleaved type then returns origSize.
//      If color format is of planar type then returns size for the specified plane.
//      If color format is undefined or number of planes is incorrect then returns {0,0}.
IW_DECL(IwiSize) iwiColorGetPlaneSize(
    IwiColorFmt color,    // Color format
    IwiSize     origSize, // Full plane image size
    int         planeNum  // Number of plane for planar format, from 0. Ignored for interleaved formats
);

/**/////////////////////////////////////////////////////////////////////////////
//                   iwiColorConvert
///////////////////////////////////////////////////////////////////////////// */

// Auxiliary parameters structure
typedef struct _IwiColorConvertParams
{
    int reserved;
} IwiColorConvertParams;

// Sets auxiliary parameters to default values
static IW_INLINE void iwiColorConvert_SetDefaultParams(
    IwiColorConvertParams *pParams      // [in,out] Pointer to the auxiliary parameters structure
)
{
    (void)pParams;
}

// Color conversions aggregation. Performs conversions between every supported color types
// Features support:
//      Inplace mode:            no
//      64-bit sizes:            no
//      Internal threading:      no
//      Manual tiling:           yes
//      IwiTile simple tiling:   yes
//      IwiTile pipeline tiling: yes
// Returns:
//      ippStsSizeErr                       size fields values are illegal
//      ippStsInplaceModeNotSupportedErr    doesn't support output into the source buffer
//      ippStsNullPtrErr                    unexpected NULL pointer
//      ippStsNoErr                         no errors
IW_DECL(IppStatus) iwiColorConvert(
    const IwiImage* const        pSrcImage[],  // [in]     Pointer to an array of pointers to source images
    IwiColorFmt                  srcFormat,    // [in]     Source image color format
    IwiImage* const              pDstImage[],  // [in,out] Pointer to an array of pointers to destination images
    IwiColorFmt                  dstFormat,    // [in]     Destination image color format
    Ipp64f                       alphaVal,     // [in]     Value to set for alpha channel if new alpha channels is to be created
    const IwiColorConvertParams *pAuxParams,   // [in]     Pointer to the auxiliary parameters structure. If NULL - default parameters will be used
    const IwiTile               *pTile         // [in,out] Pointer to IwiTile structure for tiling. If NULL - the whole image will be processed in accordance to size
);

#ifdef __cplusplus
}
#endif

#endif
