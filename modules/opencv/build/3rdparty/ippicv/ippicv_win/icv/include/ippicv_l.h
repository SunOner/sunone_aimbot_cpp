/*
// Copyright 2014 Intel Corporation All Rights Reserved.
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

#if !defined( __IPPICV_L_H__ )
#define __IPPICV_L_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "ippicv_defs_l.h"
#include "ippicv_types_l.h"
#include "ippicv_redefs.h"
#include "ippversion.h"


/* =============================================================================
							ippCore
============================================================================= */


/* /////////////////////////////////////////////////////////////////////////////
//                   Functions to allocate memory
///////////////////////////////////////////////////////////////////////////// */
/* /////////////////////////////////////////////////////////////////////////////
//  Name:       ippMalloc_L
//  Purpose:    64-byte aligned memory allocation
//  Parameter:
//    len       number of bytes
//  Returns:    pointer to allocated memory
//
//  Notes:      the memory allocated by ippMalloc has to be free by ippFree
//              function only.
*/
IPPAPI(void*, ippMalloc_L, (IppSizeL length))



/* =============================================================================
							ippVM
============================================================================= */




/* =============================================================================
							ippSP
============================================================================= */

/* /////////////////////////////////////////////////////////////////////////////
//  Name:       ippsMalloc*_L
//  Purpose:    64-byte aligned memory allocation
//  Parameter:
//    len       number of elements (according to their type)
//  Returns:    pointer to allocated memory
//
//  Notes:      the memory allocated by ippsMalloc has to be free by ippsFree
//              function only.
*/

IPPAPI(Ipp8u*, ippsMalloc_8u_L, (IppSizeL len))


/* =============================================================================
							ippIP
============================================================================= */

/* ////////////////////////////////////////////////////////////////////////////
//  Name:       ippiCopy..L
//
//  Purpose:  copy pixel values from the source image to the destination  image
//
//
//  Returns:
//    ippStsNullPtrErr  One of the pointers is NULL
//    ippStsSizeErr     roiSize has a field with zero or negative value
//    ippStsNoErr       OK
//
//  Parameters:
//    pSrc              Pointer  to the source image buffer
//    srcStep           Step in bytes through the source image buffer
//    pDst              Pointer to the  destination image buffer
//    dstStep           Step in bytes through the destination image buffer
//    roiSize           Size of the ROI
//    pMask             Pointer to the mask image buffer
//    maskStep          Step in bytes through the mask image buffer
*/

IPPAPI(IppStatus, ippiCopy_8u_C1R_L, (const Ipp8u* pSrc, IppSizeL srcStep, Ipp8u* pDst, IppSizeL dstStep, IppiSizeL roiSize))

/* ////////////////////////////////////////////////////////////////////////////
//  Name:       ippiCopyReplicateBorder
//
//  Purpose:   Copies pixel values between two buffers and adds
//             the replicated border pixels.
//
//  Returns:
//    ippStsNullPtrErr    One of the pointers is NULL
//    ippStsSizeErr       1). srcRoiSize or dstRoiSize has a field with negative or zero value
//                        2). topBorderHeight or leftBorderWidth is less than zero
//                        3). dstRoiSize.width < srcRoiSize.width + leftBorderWidth
//                        4). dstRoiSize.height < srcRoiSize.height + topBorderHeight
//    ippStsStepErr       srcStep or dstStep is less than or equal to zero
//    ippStsNoErr         OK
//
//  Parameters:
//    pSrc                Pointer  to the source image buffer
//    srcStep             Step in bytes through the source image
//    pDst                Pointer to the  destination image buffer
//    dstStep             Step in bytes through the destination image
//    scrRoiSize          Size of the source ROI in pixels
//    dstRoiSize          Size of the destination ROI in pixels
//    topBorderHeight     Height of the top border in pixels
//    leftBorderWidth     Width of the left border in pixels
*/
IPPAPI(IppStatus, ippiCopyReplicateBorder_8u_C1R_L,  (const Ipp8u* pSrc, IppSizeL srcStep, IppiSizeL srcRoiSize, Ipp8u* pDst, IppSizeL dstStep, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftBorderWidth))
IPPAPI(IppStatus, ippiCopyReplicateBorder_8u_C3R_L,  (const Ipp8u* pSrc, IppSizeL srcStep, IppiSizeL srcRoiSize, Ipp8u* pDst, IppSizeL dstStep, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftBorderWidth))
IPPAPI(IppStatus, ippiCopyReplicateBorder_8u_C4R_L,  (const Ipp8u* pSrc, IppSizeL srcStep, IppiSizeL srcRoiSize, Ipp8u* pDst, IppSizeL dstStep, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftBorderWidth))
IPPAPI(IppStatus, ippiCopyReplicateBorder_16s_C1R_L, (const Ipp16s* pSrc, IppSizeL srcStep, IppiSizeL srcRoiSize, Ipp16s* pDst, IppSizeL dstStep, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftBorderWidth))
IPPAPI(IppStatus, ippiCopyReplicateBorder_16s_C3R_L, (const Ipp16s* pSrc, IppSizeL srcStep, IppiSizeL srcRoiSize, Ipp16s* pDst, IppSizeL dstStep, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftBorderWidth))
IPPAPI(IppStatus, ippiCopyReplicateBorder_16s_C4R_L, (const Ipp16s* pSrc, IppSizeL srcStep, IppiSizeL srcRoiSize, Ipp16s* pDst, IppSizeL dstStep, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftBorderWidth))
IPPAPI(IppStatus, ippiCopyReplicateBorder_16u_C1R_L, (const Ipp16u* pSrc, IppSizeL srcStep, IppiSizeL srcRoiSize, Ipp16u* pDst, IppSizeL dstStep, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftBorderWidth))
IPPAPI(IppStatus, ippiCopyReplicateBorder_16u_C3R_L, (const Ipp16u* pSrc, IppSizeL srcStep, IppiSizeL srcRoiSize, Ipp16u* pDst, IppSizeL dstStep, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftBorderWidth))
IPPAPI(IppStatus, ippiCopyReplicateBorder_16u_C4R_L, (const Ipp16u* pSrc, IppSizeL srcStep, IppiSizeL srcRoiSize, Ipp16u* pDst, IppSizeL dstStep, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftBorderWidth))
IPPAPI(IppStatus, ippiCopyReplicateBorder_32s_C1R_L, (const Ipp32s* pSrc, IppSizeL srcStep, IppiSizeL srcRoiSize, Ipp32s* pDst, IppSizeL dstStep, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftBorderWidth))
IPPAPI(IppStatus, ippiCopyReplicateBorder_32s_C3R_L, (const Ipp32s* pSrc, IppSizeL srcStep, IppiSizeL srcRoiSize, Ipp32s* pDst, IppSizeL dstStep, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftBorderWidth))
IPPAPI(IppStatus, ippiCopyReplicateBorder_32s_C4R_L, (const Ipp32s* pSrc, IppSizeL srcStep, IppiSizeL srcRoiSize, Ipp32s* pDst, IppSizeL dstStep, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftBorderWidth))
IPPAPI(IppStatus, ippiCopyReplicateBorder_32s_C1IR_L,(const Ipp32s* pSrc, IppSizeL srcDstStep,    IppiSizeL srcRoiSize, IppiSizeL dstRoiSize,    IppSizeL topBorderHeight, IppSizeL leftborderwidth))
IPPAPI(IppStatus, ippiCopyReplicateBorder_32s_C3IR_L, (const Ipp32s* pSrc, IppSizeL srcDstStep, IppiSizeL srcRoiSize, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftborderwidth))
IPPAPI(IppStatus, ippiCopyReplicateBorder_32s_C4IR_L, (const Ipp32s* pSrc, IppSizeL srcDstStep, IppiSizeL srcRoiSize, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftborderwidth))
IPPAPI(IppStatus, ippiCopyReplicateBorder_32f_C1R_L, (const Ipp32f* pSrc, IppSizeL srcStep, IppiSizeL srcRoiSize, Ipp32f* pDst, IppSizeL dstStep, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftBorderWidth))
IPPAPI(IppStatus, ippiCopyReplicateBorder_32f_C3R_L, (const Ipp32f* pSrc, IppSizeL srcStep, IppiSizeL srcRoiSize, Ipp32f* pDst, IppSizeL dstStep, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftBorderWidth))
IPPAPI(IppStatus, ippiCopyReplicateBorder_32f_C4R_L, (const Ipp32f* pSrc, IppSizeL srcStep, IppiSizeL srcRoiSize, Ipp32f* pDst, IppSizeL dstStep, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftBorderWidth))
IPPAPI(IppStatus, ippiCopyReplicateBorder_32f_C1IR_L, (const Ipp32f* pSrc, IppSizeL srcDstStep, IppiSizeL srcRoiSize, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftborderwidth))
IPPAPI(IppStatus, ippiCopyReplicateBorder_32f_C3IR_L, (const Ipp32f* pSrc, IppSizeL srcDstStep, IppiSizeL srcRoiSize, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftborderwidth))
IPPAPI(IppStatus, ippiCopyReplicateBorder_32f_C4IR_L, (const Ipp32f* pSrc, IppSizeL srcDstStep, IppiSizeL srcRoiSize, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftborderwidth))
IPPAPI(IppStatus, ippiCopyReplicateBorder_8u_C1IR_L, (const Ipp8u* pSrc, IppSizeL srcDstStep, IppiSizeL srcRoiSize, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftborderwidth))
IPPAPI(IppStatus, ippiCopyReplicateBorder_8u_C3IR_L, (const Ipp8u* pSrc, IppSizeL srcDstStep, IppiSizeL srcRoiSize, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftborderwidth))
IPPAPI(IppStatus, ippiCopyReplicateBorder_8u_C4IR_L, (const Ipp8u* pSrc, IppSizeL srcDstStep, IppiSizeL srcRoiSize, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftborderwidth))
IPPAPI(IppStatus, ippiCopyReplicateBorder_16u_C1IR_L, (const Ipp16u* pSrc, IppSizeL srcDstStep, IppiSizeL srcRoiSize, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftborderwidth))
IPPAPI(IppStatus, ippiCopyReplicateBorder_16u_C3IR_L, (const Ipp16u* pSrc, IppSizeL srcDstStep, IppiSizeL srcRoiSize, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftborderwidth))
IPPAPI(IppStatus, ippiCopyReplicateBorder_16u_C4IR_L, (const Ipp16u* pSrc, IppSizeL srcDstStep, IppiSizeL srcRoiSize, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftborderwidth))
IPPAPI(IppStatus, ippiCopyReplicateBorder_16s_C1IR_L, (const Ipp16s* pSrc, IppSizeL srcDstStep, IppiSizeL srcRoiSize, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftborderwidth))
IPPAPI(IppStatus, ippiCopyReplicateBorder_16s_C3IR_L, (const Ipp16s* pSrc, IppSizeL srcDstStep, IppiSizeL srcRoiSize, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftborderwidth))
IPPAPI(IppStatus, ippiCopyReplicateBorder_16s_C4IR_L, (const Ipp16s* pSrc, IppSizeL srcDstStep, IppiSizeL srcRoiSize, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftborderwidth))

/* ////////////////////////////////////////////////////////////////////////////
//  Name:       ippiCopyWrapBorder
//
//  Purpose:    Copies pixel values between two buffers and adds the border pixels.
//
//  Returns:
//    ippStsNullPtrErr    One of the pointers is NULL
//    ippStsSizeErr       1). srcRoiSize or dstRoiSize has a field with negative or zero value
//                        2). topBorderHeight or leftBorderWidth is less than zero
//                        3). dstRoiSize.width < srcRoiSize.width + leftBorderWidth
//                        4). dstRoiSize.height < srcRoiSize.height + topBorderHeight
//    ippStsStepErr       srcStep or dstStep is less than or equal to zero
//    ippStsNoErr         OK
//
//  Parameters:
//    pSrc                Pointer  to the source image buffer
//    srcStep             Step in bytes through the source image
//    pDst                Pointer to the  destination image buffer
//    dstStep             Step in bytes through the destination image
//    scrRoiSize          Size of the source ROI in pixels
//    dstRoiSize          Size of the destination ROI in pixels
//    topBorderHeight     Height of the top border in pixels
//    leftBorderWidth     Width of the left border in pixels
*/
IPPAPI(IppStatus, ippiCopyWrapBorder_32s_C1R_L, (const Ipp32s* pSrc, IppSizeL srcStep, IppiSizeL srcRoiSize, Ipp32s* pDst, IppSizeL dstStep, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftBorderWidth))
IPPAPI(IppStatus, ippiCopyWrapBorder_32s_C1IR_L, (const Ipp32s* pSrc, IppSizeL srcDstStep, IppiSizeL srcRoiSize, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftborderwidth))
IPPAPI(IppStatus, ippiCopyWrapBorder_32f_C1R_L, (const Ipp32f* pSrc, IppSizeL srcStep, IppiSizeL srcRoiSize, Ipp32f* pDst, IppSizeL dstStep, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftBorderWidth))
IPPAPI(IppStatus, ippiCopyWrapBorder_32f_C1IR_L, (const Ipp32f* pSrc, IppSizeL srcDstStep, IppiSizeL srcRoiSize, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftborderwidth))

/* ////////////////////////////////////////////////////////////////////////////
//  Name:       ippiCopyConstBorder
//
//  Purpose:    Copies pixel values between two buffers and adds
//              the border pixels with constant value.
//
//  Returns:
//    ippStsNullPtrErr   One of the pointers is NULL
//    ippStsSizeErr      1). srcRoiSize or dstRoiSize has a field with negative or zero value
//                       2). topBorderHeight or leftBorderWidth is less than zero
//                       3). dstRoiSize.width < srcRoiSize.width + leftBorderWidth
//                       4). dstRoiSize.height < srcRoiSize.height + topBorderHeight
//    ippStsStepErr      srcStep or dstStep is less than or equal to zero
//    ippStsNoErr        OK
//
//  Parameters:
//    pSrc               Pointer  to the source image buffer
//    srcStep            Step in bytes through the source image
//    pDst               Pointer to the  destination image buffer
//    dstStep            Step in bytes through the destination image
//    srcRoiSize         Size of the source ROI in pixels
//    dstRoiSize         Size of the destination ROI in pixels
//    topBorderHeight    Height of the top border in pixels
//    leftBorderWidth    Width of the left border in pixels
//    value              Constant value to assign to the border pixels
*/
IPPAPI(IppStatus, ippiCopyConstBorder_8u_C1R_L, (const Ipp8u* pSrc, IppSizeL srcStep, IppiSizeL srcRoiSize, Ipp8u* pDst, IppSizeL dstStep, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftBorderWidth, Ipp8u value))
IPPAPI(IppStatus, ippiCopyConstBorder_8u_C3R_L, (const Ipp8u* pSrc, IppSizeL srcStep, IppiSizeL srcRoiSize, Ipp8u* pDst, IppSizeL dstStep, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftBorderWidth, const Ipp8u value[3]))
IPPAPI(IppStatus, ippiCopyConstBorder_8u_C4R_L, (const Ipp8u* pSrc, IppSizeL srcStep, IppiSizeL srcRoiSize, Ipp8u* pDst, IppSizeL dstStep, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftBorderWidth, const Ipp8u value[4]))
IPPAPI(IppStatus, ippiCopyConstBorder_16s_C1R_L, (const Ipp16s* pSrc, IppSizeL srcStep, IppiSizeL srcRoiSize, Ipp16s* pDst, IppSizeL dstStep, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftBorderWidth, Ipp16s value))
IPPAPI(IppStatus, ippiCopyConstBorder_16s_C3R_L, (const Ipp16s* pSrc, IppSizeL srcStep, IppiSizeL srcRoiSize, Ipp16s* pDst, IppSizeL dstStep, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftBorderWidth, const Ipp16s value[3]))
IPPAPI(IppStatus, ippiCopyConstBorder_16s_C4R_L, (const Ipp16s* pSrc, IppSizeL srcStep, IppiSizeL srcRoiSize, Ipp16s* pDst, IppSizeL dstStep, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftBorderWidth, const Ipp16s value[4]))
IPPAPI(IppStatus, ippiCopyConstBorder_32s_C1R_L, (const Ipp32s* pSrc, IppSizeL srcStep, IppiSizeL srcRoiSize, Ipp32s* pDst, IppSizeL dstStep, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftBorderWidth, Ipp32s value))
IPPAPI(IppStatus, ippiCopyConstBorder_32s_C3R_L, (const Ipp32s* pSrc, IppSizeL srcStep, IppiSizeL srcRoiSize, Ipp32s* pDst, IppSizeL dstStep, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftBorderWidth, const Ipp32s value[3]))
IPPAPI(IppStatus, ippiCopyConstBorder_32s_C4R_L, (const Ipp32s* pSrc, IppSizeL srcStep, IppiSizeL srcRoiSize, Ipp32s* pDst, IppSizeL dstStep, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftBorderWidth, const Ipp32s value[4]))

IPPAPI(IppStatus, ippiCopyConstBorder_16u_C1R_L, (const Ipp16u* pSrc, IppSizeL srcStep, IppiSizeL srcRoiSize, Ipp16u* pDst, IppSizeL dstStep, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftBorderWidth, Ipp16u value))
IPPAPI(IppStatus, ippiCopyConstBorder_16u_C3R_L, (const Ipp16u* pSrc, IppSizeL srcStep, IppiSizeL srcRoiSize, Ipp16u* pDst, IppSizeL dstStep, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftBorderWidth, const Ipp16u value[3]))
IPPAPI(IppStatus, ippiCopyConstBorder_16u_C4R_L, (const Ipp16u* pSrc, IppSizeL srcStep, IppiSizeL srcRoiSize, Ipp16u* pDst, IppSizeL dstStep, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftBorderWidth, const Ipp16u value[4]))

IPPAPI(IppStatus, ippiCopyConstBorder_32f_C1R_L, (const Ipp32f* pSrc, IppSizeL srcStep, IppiSizeL srcRoiSize, Ipp32f* pDst, IppSizeL dstStep, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftBorderWidth, Ipp32f value))
IPPAPI(IppStatus, ippiCopyConstBorder_32f_C3R_L, (const Ipp32f* pSrc, IppSizeL srcStep, IppiSizeL srcRoiSize, Ipp32f* pDst, IppSizeL dstStep, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftBorderWidth, const Ipp32f value[3]))
IPPAPI(IppStatus, ippiCopyConstBorder_32f_C4R_L, (const Ipp32f* pSrc, IppSizeL srcStep, IppiSizeL srcRoiSize, Ipp32f* pDst, IppSizeL dstStep, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftBorderWidth, const Ipp32f value[4]))


IPPAPI(IppStatus, ippiCopyConstBorder_8u_C1IR_L, (Ipp8u* pSrcDst, IppSizeL srcDstStep, IppiSizeL srcRoiSize, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftborderwidth, const Ipp8u value))
IPPAPI(IppStatus, ippiCopyConstBorder_8u_C3IR_L, (Ipp8u* pSrcDst, IppSizeL srcDstStep, IppiSizeL srcRoiSize, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftborderwidth, const Ipp8u value[3]))
IPPAPI(IppStatus, ippiCopyConstBorder_8u_C4IR_L, (Ipp8u* pSrcDst, IppSizeL srcDstStep, IppiSizeL srcRoiSize, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftborderwidth, const Ipp8u value[4]))

IPPAPI(IppStatus, ippiCopyConstBorder_16u_C1IR_L, (Ipp16u* pSrcDst, IppSizeL srcDstStep, IppiSizeL srcRoiSize, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftborderwidth, const Ipp16u value))
IPPAPI(IppStatus, ippiCopyConstBorder_16u_C3IR_L, (Ipp16u* pSrcDst, IppSizeL srcDstStep, IppiSizeL srcRoiSize, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftborderwidth, const Ipp16u value[3]))
IPPAPI(IppStatus, ippiCopyConstBorder_16u_C4IR_L, (Ipp16u* pSrcDst, IppSizeL srcDstStep, IppiSizeL srcRoiSize, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftborderwidth, const Ipp16u value[4]))

IPPAPI(IppStatus, ippiCopyConstBorder_16s_C1IR_L, (Ipp16s* pSrcDst, IppSizeL srcDstStep, IppiSizeL srcRoiSize, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftborderwidth, const Ipp16s value))
IPPAPI(IppStatus, ippiCopyConstBorder_16s_C3IR_L, (Ipp16s* pSrcDst, IppSizeL srcDstStep, IppiSizeL srcRoiSize, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftborderwidth, const Ipp16s value[3]))
IPPAPI(IppStatus, ippiCopyConstBorder_16s_C4IR_L, (Ipp16s* pSrcDst, IppSizeL srcDstStep, IppiSizeL srcRoiSize, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftborderwidth, const Ipp16s value[4]))

IPPAPI(IppStatus, ippiCopyConstBorder_32s_C1IR_L, (Ipp32s* pSrcDst, IppSizeL srcDstStep, IppiSizeL srcRoiSize, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftborderwidth, const Ipp32s value))
IPPAPI(IppStatus, ippiCopyConstBorder_32s_C3IR_L, (Ipp32s* pSrcDst, IppSizeL srcDstStep, IppiSizeL srcRoiSize, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftborderwidth, const Ipp32s value[3]))
IPPAPI(IppStatus, ippiCopyConstBorder_32s_C4IR_L, (Ipp32s* pSrcDst, IppSizeL srcDstStep, IppiSizeL srcRoiSize, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftborderwidth, const Ipp32s value[4]))

IPPAPI(IppStatus, ippiCopyConstBorder_32f_C1IR_L, (Ipp32f* pSrcDst, IppSizeL srcDstStep, IppiSizeL srcRoiSize, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftborderwidth, const Ipp32f value))
IPPAPI(IppStatus, ippiCopyConstBorder_32f_C3IR_L, (Ipp32f* pSrcDst, IppSizeL srcDstStep, IppiSizeL srcRoiSize, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftborderwidth, const Ipp32f value[3]))
IPPAPI(IppStatus, ippiCopyConstBorder_32f_C4IR_L, (Ipp32f* pSrcDst, IppSizeL srcDstStep, IppiSizeL srcRoiSize, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftborderwidth, const Ipp32f value[4]))


/* ////////////////////////////////////////////////////////////////////////////
//  Name:       ippiCopyMirrorBorder
//
//  Purpose:   Copies pixel values between two buffers and adds
//             the mirror border pixels.
//
//  Returns:
//    ippStsNullPtrErr    One of the pointers is NULL
//    ippStsSizeErr       1). srcRoiSize or dstRoiSize has a field with negative or zero value
//                        2). topBorderHeight or leftBorderWidth is less than zero
//                        3). dstRoiSize.width < srcRoiSize.width + leftBorderWidth
//                        4). dstRoiSize.height < srcRoiSize.height + topBorderHeight
//    ippStsStepErr       srcStep or dstStep is less than or equal to zero
//    ippStsNoErr         OK
//
//  Parameters:
//    pSrc                Pointer  to the source image buffer
//    srcStep             Step in bytes through the source image
//    pDst                Pointer to the  destination image buffer
//    dstStep             Step in bytes through the destination image
//    scrRoiSize          Size of the source ROI in pixels
//    dstRoiSize          Size of the destination ROI in pixels
//    topBorderHeight     Height of the top border in pixels
//    leftBorderWidth     Width of the left border in pixels
*/
IPPAPI(IppStatus, ippiCopyMirrorBorder_8u_C1R_L, (const Ipp8u* pSrc, IppSizeL srcStep, IppiSizeL srcRoiSize, Ipp8u* pDst, IppSizeL dstStep, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftBorderWidth))
IPPAPI(IppStatus, ippiCopyMirrorBorder_8u_C3R_L, (const Ipp8u* pSrc, IppSizeL srcStep, IppiSizeL srcRoiSize, Ipp8u* pDst, IppSizeL dstStep, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftBorderWidth))
IPPAPI(IppStatus, ippiCopyMirrorBorder_8u_C4R_L, (const Ipp8u* pSrc, IppSizeL srcStep, IppiSizeL srcRoiSize, Ipp8u* pDst, IppSizeL dstStep, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftBorderWidth))
IPPAPI(IppStatus, ippiCopyMirrorBorder_16s_C1R_L, (const Ipp16s* pSrc, IppSizeL srcStep, IppiSizeL srcRoiSize, Ipp16s* pDst, IppSizeL dstStep, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftBorderWidth))
IPPAPI(IppStatus, ippiCopyMirrorBorder_16s_C3R_L, (const Ipp16s* pSrc, IppSizeL srcStep, IppiSizeL srcRoiSize, Ipp16s* pDst, IppSizeL dstStep, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftBorderWidth))
IPPAPI(IppStatus, ippiCopyMirrorBorder_16s_C4R_L, (const Ipp16s* pSrc, IppSizeL srcStep, IppiSizeL srcRoiSize, Ipp16s* pDst, IppSizeL dstStep, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftBorderWidth))
IPPAPI(IppStatus, ippiCopyMirrorBorder_32s_C1R_L, (const Ipp32s* pSrc, IppSizeL srcStep, IppiSizeL srcRoiSize, Ipp32s* pDst, IppSizeL dstStep, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftBorderWidth))
IPPAPI(IppStatus, ippiCopyMirrorBorder_32s_C3R_L, (const Ipp32s* pSrc, IppSizeL srcStep, IppiSizeL srcRoiSize, Ipp32s* pDst, IppSizeL dstStep, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftBorderWidth))
IPPAPI(IppStatus, ippiCopyMirrorBorder_32s_C4R_L, (const Ipp32s* pSrc, IppSizeL srcStep, IppiSizeL srcRoiSize, Ipp32s* pDst, IppSizeL dstStep, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftBorderWidth))
IPPAPI(IppStatus, ippiCopyMirrorBorder_8u_C1IR_L, (const Ipp8u* pSrc, IppSizeL srcDstStep, IppiSizeL srcRoiSize, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftborderwidth))
IPPAPI(IppStatus, ippiCopyMirrorBorder_8u_C3IR_L, (const Ipp8u* pSrc, IppSizeL srcDstStep, IppiSizeL srcRoiSize, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftborderwidth))
IPPAPI(IppStatus, ippiCopyMirrorBorder_8u_C4IR_L, (const Ipp8u* pSrc, IppSizeL srcDstStep, IppiSizeL srcRoiSize, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftborderwidth))
IPPAPI(IppStatus, ippiCopyMirrorBorder_16s_C1IR_L, (const Ipp16s* pSrc, IppSizeL srcDstStep, IppiSizeL srcRoiSize, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftborderwidth))
IPPAPI(IppStatus, ippiCopyMirrorBorder_16s_C3IR_L, (const Ipp16s* pSrc, IppSizeL srcDstStep, IppiSizeL srcRoiSize, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftborderwidth))
IPPAPI(IppStatus, ippiCopyMirrorBorder_16s_C4IR_L, (const Ipp16s* pSrc, IppSizeL srcDstStep, IppiSizeL srcRoiSize, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftborderwidth))
IPPAPI(IppStatus, ippiCopyMirrorBorder_32s_C1IR_L, (const Ipp32s* pSrc, IppSizeL srcDstStep, IppiSizeL srcRoiSize, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftborderwidth))
IPPAPI(IppStatus, ippiCopyMirrorBorder_32s_C3IR_L, (const Ipp32s* pSrc, IppSizeL srcDstStep, IppiSizeL srcRoiSize, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftborderwidth))
IPPAPI(IppStatus, ippiCopyMirrorBorder_32s_C4IR_L, (const Ipp32s* pSrc, IppSizeL srcDstStep, IppiSizeL srcRoiSize, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftborderwidth))
IPPAPI(IppStatus, ippiCopyMirrorBorder_16u_C1IR_L, (const Ipp16u* pSrc, IppSizeL srcDstStep, IppiSizeL srcRoiSize, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftborderwidth))
IPPAPI(IppStatus, ippiCopyMirrorBorder_16u_C3IR_L, (const Ipp16u* pSrc, IppSizeL srcDstStep, IppiSizeL srcRoiSize, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftborderwidth))
IPPAPI(IppStatus, ippiCopyMirrorBorder_16u_C4IR_L, (const Ipp16u* pSrc, IppSizeL srcDstStep, IppiSizeL srcRoiSize, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftborderwidth))
IPPAPI(IppStatus, ippiCopyMirrorBorder_16u_C1R_L, (const Ipp16u* pSrc, IppSizeL srcStep, IppiSizeL srcRoiSize, Ipp16u* pDst, IppSizeL dstStep, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftBorderWidth))
IPPAPI(IppStatus, ippiCopyMirrorBorder_16u_C3R_L, (const Ipp16u* pSrc, IppSizeL srcStep, IppiSizeL srcRoiSize, Ipp16u* pDst, IppSizeL dstStep, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftBorderWidth))
IPPAPI(IppStatus, ippiCopyMirrorBorder_16u_C4R_L, (const Ipp16u* pSrc, IppSizeL srcStep, IppiSizeL srcRoiSize, Ipp16u* pDst, IppSizeL dstStep, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftBorderWidth))
IPPAPI(IppStatus, ippiCopyMirrorBorder_32f_C1R_L, (const Ipp32f* pSrc, IppSizeL srcStep, IppiSizeL srcRoiSize, Ipp32f* pDst, IppSizeL dstStep, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftBorderWidth))
IPPAPI(IppStatus, ippiCopyMirrorBorder_32f_C3R_L, (const Ipp32f* pSrc, IppSizeL srcStep, IppiSizeL srcRoiSize, Ipp32f* pDst, IppSizeL dstStep, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftBorderWidth))
IPPAPI(IppStatus, ippiCopyMirrorBorder_32f_C4R_L, (const Ipp32f* pSrc, IppSizeL srcStep, IppiSizeL srcRoiSize, Ipp32f* pDst, IppSizeL dstStep, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftBorderWidth))
IPPAPI(IppStatus, ippiCopyMirrorBorder_32f_C1IR_L, (const Ipp32f* pSrc, IppSizeL srcDstStep, IppiSizeL srcRoiSize, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftborderwidth))
IPPAPI(IppStatus, ippiCopyMirrorBorder_32f_C3IR_L, (const Ipp32f* pSrc, IppSizeL srcDstStep, IppiSizeL srcRoiSize, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftborderwidth))
IPPAPI(IppStatus, ippiCopyMirrorBorder_32f_C4IR_L, (const Ipp32f* pSrc, IppSizeL srcDstStep, IppiSizeL srcRoiSize, IppiSizeL dstRoiSize, IppSizeL topBorderHeight, IppSizeL leftborderwidth))

/* /////////////////////////////////////////////////////////////////////////////
//                     Bilateral filter functions with Border
/////////////////////////////////////////////////////////////////////////////
//  Name:       ippiFilterBilateralBorderGetBufferSize_L
//  Purpose:    to define buffer size for bilateral filter
//  Parameters:
//   filter        Type of bilateral filter. Possible value is ippiFilterBilateralGauss.
//   dstRoiSize    Roi size (in pixels) of destination image what will be applied
//                 for processing.
//   radius        Radius of circular neighborhood what defines pixels for calculation.
//   dataType      Data type of the source and destination images. Possible values
//                 are Ipp8u and Ipp32f.
//   numChannels   Number of channels in the images. Possible values are 1 and 3.
//   distMethod    The type of method for definition of distance beetween pixel untensity.
//                 Possible value is ippDistNormL1.
//   pSpecSize     Pointer to the size (in bytes) of the spec.
//   pBufferSize   Pointer to the size (in bytes) of the external work buffer.
//  Return:
//    ippStsNoErr               OK
//    ippStsNullPtrErr          any pointer is NULL
//    ippStsSizeErr             size of dstRoiSize is less or equal 0
//    ippStsMaskSizeErr         radius is less or equal 0
//    ippStsNotSupportedModeErr filter or distMethod is not supported
//    ippStsDataTypeErr         Indicates an error when dataType has an illegal value.
//    ippStsNumChannelsErr      Indicates an error when numChannels has an illegal value.
*/
IPPAPI(IppStatus, ippiFilterBilateralBorderGetBufferSize_L ,  (IppiFilterBilateralType filter, IppiSizeL dstRoiSize, int radius, IppDataType dataType, int numChannels, IppiDistanceMethodType distMethodType, IppSizeL *pSpecSize, IppSizeL *pBufferSize))

/* /////////////////////////////////////////////////////////////////////////////
//  Name:       ippiFilterBilateralBorderInit_L
//  Purpose:    initialization of Spec for bilateral filter with border
//  Parameters:
//   filter           Type of bilateral filter. Possible value is ippiFilterBilateralGauss.
//   dstRoiSize       Roi size (in pixels) of destination image what will be applied
//                    for processing.
//   radius           Radius of circular neighborhood what defines pixels for calculation.
//   dataType         Data type of the source and destination images. Possible values
//                    are Ipp8u and Ipp32f.
//   numChannels      Number of channels in the images. Possible values are 1 and 3.
//   distMethodType   The type of method for definition of distance between pixel intensity.
//                    Possible value is ippDistNormL1.
//   valSquareSigma   square of Sigma for factor function for pixel intensity
//   posSquareSigma   square of Sigma for factor function for pixel position
//    pSpec           pointer to Spec
//  Return:
//    ippStsNoErr               OK
//    ippStsNullPtrErr          pointer ro Spec is NULL
//    ippStsSizeErr             size of dstRoiSize is less or equal 0
//    ippStsMaskSizeErr         radius is less or equal 0
//    ippStsNotSupportedModeErr filter or distMethod is not supported
//    ippStsDataTypeErr         Indicates an error when dataType has an illegal value.
//    ippStsNumChannelsErr      Indicates an error when numChannels has an illegal value.
//    ippStsBadArgErr           valSquareSigma or posSquareSigma is less or equal 0
*/
IPPAPI(IppStatus, ippiFilterBilateralBorderInit_L ,(IppiFilterBilateralType filter, IppiSizeL dstRoiSize, int radius, IppDataType dataType, int numChannels, IppiDistanceMethodType distMethod, Ipp32f valSquareSigma, Ipp32f posSquareSigma, IppiFilterBilateralSpec *pSpec))

/* /////////////////////////////////////////////////////////////////////////////
//  Name:       ippiFilterBilateralBorder_8u_C1R
//              ippiFilterBilateralBorder_8u_C3R
//  Purpose:    bilateral filter
//  Parameters:
//    pSrc         Pointer to the source image
//    srcStep      Step through the source image
//    pDst         Pointer to the destination image
//    dstStep      Step through the destination image
//    dstRoiSize   Size of the destination ROI
//    borderType   Type of border.
//    borderValue  Pointer to constant value to assign to pixels of the constant border. This parameter is applicable
//                 only to the ippBorderConst border type. If this pointer is NULL than the constant value is equal 0.
//    pSpec        Pointer to filter spec
//    pBuffer      Pointer to work buffer
//  Return:
//    ippStsNoErr           OK
//    ippStsNullPtrErr      pointer to Src, Dst, Spec or Buffer is NULL
//    ippStsSizeErr         size of dstRoiSize is less or equal 0
//    ippStsContextMatchErr filter Spec is not match
//    ippStsNotEvenStepErr  Indicated an error when one of the step values is not divisible by 4
//                          for floating-point images.
//    ippStsBorderErr       Indicates an error when borderType has illegal value.
*/
IPPAPI(IppStatus, ippiFilterBilateralBorder_8u_C1R_L, (const Ipp8u *pSrc, IppSizeL srcStep, Ipp8u *pDst, IppSizeL dstStep, IppiSizeL dstRoiSize, IppiBorderType borderType, Ipp8u *pBorderValue, const IppiFilterBilateralSpec *pSpec, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiFilterBilateralBorder_8u_C3R_L, (const Ipp8u *pSrc, IppSizeL srcStep, Ipp8u *pDst, IppSizeL dstStep, IppiSizeL dstRoiSize, IppiBorderType borderType, Ipp8u *pBorderValue, const IppiFilterBilateralSpec *pSpec, Ipp8u* pBuffer))


/* /////////////////////////////////////////////////////////////////////////////
//                      Resize Transform Functions
///////////////////////////////////////////////////////////////////////////// */

/* /////////////////////////////////////////////////////////////////////////////
//  Name:               ippiResizeGetSize_L
//  Purpose:            Computes the size of Spec structure and temporal buffer for Resize transform
//
//  Parameters:
//    srcSize           Size of the input image (in pixels)
//    dstSize           Size of the output image (in pixels)
//    dataType          Data type {ipp8u|ipp16u|ipp16s|ipp32f} and ipp64f only for Linear interpolation
//    numChannels       Number of channels, possible values are 1 or 3 or 4
//    interpolation     Interpolation method
//    antialiasing      Supported values:
//                        0 - resizing without antialiasing
//                        1 - resizing with antialiasing
//    pSpecSize         Pointer to the size (in bytes) of the Spec structure
//    pInitBufSize      Pointer to the size (in bytes) of the temporal buffer
//
//  Return Values:
//    ippStsNoErr               Indicates no error
//    ippStsNullPtrErr          Indicates an error if one of the specified pointers is NULL
//    ippStsNoOperation         Indicates a warning if width or height of any image is zero
//    ippStsSizeErr             Indicates an error in the following cases:
//                              -  if width or height of the source or destination image is negative,
//                              -  if the source image size is less than a filter size of the chosen
//                                 interpolation method (except ippSuper)
//                              -  if one of the specified dimensions of the source image is less than
//                                 the corresponding dimension of the destination image (for ippSuper method only)
//    ippStsExceededSizeErr     Indicates an error in the following cases:
//                              -  if one of the calculated sizes exceeds maximum of IppSizeL type positive value
//                                 (the size of the one of the processed images is too large)
//                              -  if one of width or height of the destination image or the source image with borders
//                                 exceeds 536870911 (0x1FFFFFFF)
//    ippStsInterpolationErr    Indicates an error if interpolation has an illegal value
//    ippStsDataTypeErr         Indicates an error when dataType has an illegal value
//    ippStsNoAntialiasing      Indicates a warning if specified interpolation does not support antialiasing
//    ippStsNotSupportedModeErr Indicates an error if requested mode is currently not supported
//
//  Notes:
//    1. Supported interpolation methods are ippNearest, ippLinear, ippCubic, ippLanczos and ippSuper.
//    2. If antialiasing value is equal to 1, use the ippResizeAntialiasing<Filter>Init functions, otherwise, use ippResize<Filter>Init
//    3. The implemented interpolation algorithms have the following filter sizes: Nearest Neighbor 1x1,
//       Linear 2x2, Cubic 4x4, 2-lobed Lanczos 4x4.
*/
IPPAPI(IppStatus, ippiResizeGetSize_L, (IppiSizeL srcSize, IppiSizeL dstSize, IppDataType dataType, IppiInterpolationType interpolation, Ipp32u antialiasing, IppSizeL* pSpecSize, IppSizeL* pInitBufSize))

/* /////////////////////////////////////////////////////////////////////////////
//  Name:               ippiResizeGetBufferSize_L
//  Purpose:            Computes the size of external buffer for Resize transform
//
//  Parameters:
//    pSpec             Pointer to the Spec structure for resize filter
//    dstSize           Size of the output image (in pixels)
//    numChannels       Number of channels, possible values are 1 or 3 or 4
//    pBufSize          Pointer to the size (in bytes) of the external buffer
//
//  Return Values:
//    ippStsNoErr               Indicates no error
//    ippStsNullPtrErr          Indicates an error if one of the specified pointers is NULL
//    ippStsContextMatchErr     Indicates an error if pointer to an invalid pSpec structure is passed
//    ippStsNumChannelsErr      Indicates an error if numChannels has illegal value
//    ippStsExceededSizeErr     Indicates an error if one of the calculated sizes exceeds maximum of IppSizeL type
//                              positive value (the size of the one of the processed images is too large)
//    ippStsSizeWrn             Indicates a warning if the destination image size is more than
//                              the destination image origin size
*/
IPPAPI(IppStatus, ippiResizeGetBufferSize_L, (const IppiResizeSpec* pSpec, IppiSizeL dstSize, Ipp32u numChannels, IppSizeL*  pBufSize))

/* /////////////////////////////////////////////////////////////////////////////
//  Name:               ippiResizeGetBorderSize_L
//  Purpose:            Computes the size of possible borders for Resize transform
//
//  Parameters:
//    pSpec             Pointer to the Spec structure for resize filter
//    borderSize        Size of necessary borders (for memory allocation)
//
//  Return Values:
//    ippStsNoErr           Indicates no error
//    ippStsNullPtrErr      Indicates an error if one of the specified pointers is NULL
//    ippStsContextMatchErr Indicates an error if pointer to an invalid pSpec structure is passed
*/
IPPAPI(IppStatus, ippiResizeGetBorderSize_L, (const IppiResizeSpec* pSpec, IppiBorderSize* pBorderSize))

/* /////////////////////////////////////////////////////////////////////////////
//  Name:               ippiResizeGetSrcOffset
//  Purpose:            Computes the offset of input image for Resize transform by tile processing
//
//  Parameters:
//    pSpec             Pointer to the Spec structure for resize filter
//    dstOffset         Offset of the tiled destination image respective
//                      to the destination image origin
//    srcOffset         Pointer to the computed offset of input image
//
//  Return Values:
//    ippStsNoErr           Indicates no error
//    ippStsNullPtrErr      Indicates an error if one of the specified pointers is NULL
//    ippStsContextMatchErr Indicates an error if pointer to an invalid pSpec structure is passed
//    ippStsOutOfRangeErr   Indicates an error if the destination image offset point is outside the
//                          destination image origin
*/
IPPAPI (IppStatus, ippiResizeGetSrcOffset_L, (const IppiResizeSpec* pSpec, IppiPointL dstOffset, IppiPointL* srcOffset))

/* /////////////////////////////////////////////////////////////////////////////
//  Name:               ippiResizeGetSrcRoi
//  Purpose:            Computes the ROI of input image
//                      for Resize transform by tile processing
//
//  Parameters:
//    pSpec             Pointer to the Spec structure for resize filter
//    dstRoiOffset      Offset of the destination image ROI
//    dstRoiSize        Size of the ROI of destination image
//    srcRoiOffset      Pointer to the computed offset of source image ROI
//    srcRoiSize        Pointer to the computed ROI size of source image
//
//  Return Values:
//    ippStsNoErr           Indicates no error
//    ippStsNullPtrErr      Indicates an error if one of the specified pointers is NULL
//    ippStsContextMatchErr Indicates an error if pointer to an invalid pSpec structure is passed
//    ippStsOutOfRangeErr   Indicates an error if the destination image offset point is outside
//                          the destination image origin
//    ippStsSizeErr         Indicates an error in the following cases:
//                           -  if width or height of the destination image ROI size 
//                              is negative or equal to 0,
//    IppStsSizeWrn         Indicates a warning if the destination ROI exceeds with
//                          the destination image origin
*/
IPPAPI (IppStatus, ippiResizeGetSrcRoi_L, (const IppiResizeSpec* pSpec, IppiPointL dstRoiOffset, IppiSizeL dstRoiSize, IppiPointL* srcRoiOffset, IppiSizeL* srcRoiSize))

/* /////////////////////////////////////////////////////////////////////////////
//  Name:               ippiResizeNearestInit_L
//                      ippiResizeLinearInit_L
//                      ippiResizeCubicInit_L
//                      ippiResizeLanczosInit_L
//                      ippiResizeSuperInit_L
//
//  Purpose:            Initializes the Spec structure for the Resize transform
//                      by different interpolation methods
//
//  Parameters:
//    srcSize           Size of the input image (in pixels)
//    dstSize           Size of the output image (in pixels)
//    dataType          Data type {ipp8u|ipp16u|ipp16s|ipp32f} and ipp64f only for Linear interpolation
//    numChannels       Number of channels, possible values are 1 or 3 or 4
//    valueB            The first parameter (B) for specifying Cubic filters
//    valueC            The second parameter (C) for specifying Cubic filters
//    numLobes          The parameter for specifying Lanczos (2 or 3) or Hahn (3 or 4) filters
//    pInitBuf          Pointer to the temporal buffer for several filter initialization
//    pSpec             Pointer to the Spec structure for resize filter
//
//  Return Values:
//    ippStsNoErr               Indicates no error
//    ippStsNullPtrErr          Indicates an error if one of the specified pointers is NULL
//    ippStsNoOperation         Indicates a warning if width or height of any image is zero
//    ippStsSizeErr             Indicates an error in the following cases:
//                              -  if width or height of the source or destination image is negative,
//                              -  if the source image size is less than a filter size of the chosen
//                                 interpolation method (except ippiResizeSuperInit).
//                              -  if one of the specified dimensions of the source image is less than
//                                 the corresponding dimension of the destination image
//                                 (for ippiResizeSuperInit only).
//    ippStsExceededSizeErr     Indicates an error if one of width or height of the destination image or
//                              the source image with borders exceeds 536870911 (0x1FFFFFFF)
//    ippStsDataTypeErr         Indicates an error when dataType has an illegal value.
//    ippStsNotSupportedModeErr Indicates an error if the requested mode is not supported.
//
//  Notes/References:
//    1. The equation shows the family of cubic filters:
//           ((12-9B-6C)*|x|^3 + (-18+12B+6C)*|x|^2                  + (6-2B)  ) / 6   for |x| < 1
//    K(x) = ((   -B-6C)*|x|^3 + (    6B+30C)*|x|^2 + (-12B-48C)*|x| + (8B+24C)) / 6   for 1 <= |x| < 2
//           0   elsewhere
//    Some values of (B,C) correspond to known cubic splines: Catmull-Rom (B=0,C=0.5), B-Spline (B=1,C=0) and other.
//      Mitchell, Don P.; Netravali, Arun N. (Aug. 1988). "Reconstruction filters in computer graphics"
//      http://www.mentallandscape.com/Papers_siggraph88.pdf
//
//    2. Hahn filter does not supported now.
//    3. The implemented interpolation algorithms have the following filter sizes: Nearest Neighbor 1x1,
//       Linear 2x2, Cubic 4x4, 2-lobed Lanczos 4x4, 3-lobed Lanczos 6x6.
*/
IPPAPI(IppStatus, ippiResizeNearestInit_L, (IppiSizeL srcSize, IppiSizeL dstSize, IppDataType dataType, IppiResizeSpec* pSpec))
IPPAPI(IppStatus, ippiResizeLinearInit_L,  (IppiSizeL srcSize, IppiSizeL dstSize, IppDataType dataType, IppiResizeSpec* pSpec))
IPPAPI(IppStatus, ippiResizeCubicInit_L,   (IppiSizeL srcSize, IppiSizeL dstSize, IppDataType dataType, Ipp32f valueB, Ipp32f valueC, IppiResizeSpec* pSpec, Ipp8u* pInitBuf))
IPPAPI(IppStatus, ippiResizeLanczosInit_L, (IppiSizeL srcSize, IppiSizeL dstSize, IppDataType dataType, Ipp32u numLobes, IppiResizeSpec* pSpec, Ipp8u* pInitBuf))
IPPAPI(IppStatus, ippiResizeSuperInit_L,   (IppiSizeL srcSize, IppiSizeL dstSize, IppDataType dataType, IppiResizeSpec* pSpec))

/* /////////////////////////////////////////////////////////////////////////////
//  Name:               ippiResizeNearest
//                      ippiResizeLinear
//                      ippiResizeCubic
//                      ippiResizeLanczos
//                      ippiResizeSuper
//
//  Purpose:            Changes an image size by different interpolation methods
//
//  Parameters:
//    pSrc              Pointer to the source image
//    srcStep           Distance (in bytes) between of consecutive lines in the source image
//    pDst              Pointer to the destination image
//    dstStep           Distance (in bytes) between of consecutive lines in the destination image
//    border            Type of the border
//    borderValue       Pointer to the constant value(s) if border type equals ippBorderConstant
//    pSpec             Pointer to the Spec structure for resize filter
//    pBuffer           Pointer to the work buffer
//
//  Return Values:
//    ippStsNoErr               Indicates no error
//    ippStsNullPtrErr          Indicates an error if one of the specified pointers is NULL
//    ippStsNoOperation         Indicates a warning if width or height of output image is zero
//    ippStsBorderErr           Indicates an error if border type has an illegal value
//    ippStsContextMatchErr     Indicates an error if pointer to an invalid pSpec structure is passed
//    ippStsNotSupportedModeErr Indicates an error if requested mode is currently not supported
//    ippStsSizeErr             Indicates an error if width or height of the destination image
//                              is negative
//    ippStsStepErr             Indicates an error if the step value is not data type multiple
//    ippStsSizeWrn             Indicates a warning if the destination image size is more than
//                              the destination image origin size
//
//  Notes:
//    1. Supported border types are ippBorderInMem and ippBorderRepl
//       (except Nearest Neighbor and Super Sampling methods).
//    2. Hahn filter does not supported now.
*/
IPPAPI(IppStatus, ippiResizeNearest_8u_C1R_L, (const Ipp8u* pSrc, IppSizeL srcStep, Ipp8u* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiResizeNearest_8u_C3R_L, (const Ipp8u* pSrc, IppSizeL srcStep, Ipp8u* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiResizeNearest_8u_C4R_L, (const Ipp8u* pSrc, IppSizeL srcStep, Ipp8u* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiResizeNearest_16u_C1R_L, (const Ipp16u* pSrc, IppSizeL srcStep, Ipp16u* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiResizeNearest_16u_C3R_L, (const Ipp16u* pSrc, IppSizeL srcStep, Ipp16u* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiResizeNearest_16u_C4R_L, (const Ipp16u* pSrc, IppSizeL srcStep, Ipp16u* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiResizeNearest_16s_C1R_L, (const Ipp16s* pSrc, IppSizeL srcStep, Ipp16s* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiResizeNearest_16s_C3R_L, (const Ipp16s* pSrc, IppSizeL srcStep, Ipp16s* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiResizeNearest_16s_C4R_L, (const Ipp16s* pSrc, IppSizeL srcStep, Ipp16s* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiResizeNearest_32f_C1R_L, (const Ipp32f* pSrc, IppSizeL srcStep, Ipp32f* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiResizeNearest_32f_C3R_L, (const Ipp32f* pSrc, IppSizeL srcStep, Ipp32f* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiResizeNearest_32f_C4R_L, (const Ipp32f* pSrc, IppSizeL srcStep, Ipp32f* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))

IPPAPI (IppStatus, ippiResizeLinear_8u_C1R_L,  (const Ipp8u* pSrc, IppSizeL srcStep, Ipp8u* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, IppiBorderType border, const Ipp8u* pBorderValue, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiResizeLinear_8u_C3R_L,  (const Ipp8u* pSrc, IppSizeL srcStep, Ipp8u* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, IppiBorderType border, const Ipp8u* pBorderValue, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiResizeLinear_8u_C4R_L,  (const Ipp8u* pSrc, IppSizeL srcStep, Ipp8u* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, IppiBorderType border, const Ipp8u* pBorderValue, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiResizeLinear_16u_C1R_L, (const Ipp16u* pSrc, IppSizeL srcStep, Ipp16u* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, IppiBorderType border, const Ipp16u* pBorderValue, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiResizeLinear_16u_C3R_L, (const Ipp16u* pSrc, IppSizeL srcStep, Ipp16u* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, IppiBorderType border, const Ipp16u* pBorderValue, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiResizeLinear_16u_C4R_L, (const Ipp16u* pSrc, IppSizeL srcStep, Ipp16u* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, IppiBorderType border, const Ipp16u* pBorderValue, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiResizeLinear_16s_C1R_L, (const Ipp16s* pSrc, IppSizeL srcStep, Ipp16s* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, IppiBorderType border, const Ipp16s* pBorderValue, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiResizeLinear_16s_C3R_L, (const Ipp16s* pSrc, IppSizeL srcStep, Ipp16s* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, IppiBorderType border, const Ipp16s* pBorderValue, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiResizeLinear_16s_C4R_L, (const Ipp16s* pSrc, IppSizeL srcStep, Ipp16s* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, IppiBorderType border, const Ipp16s* pBorderValue, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiResizeLinear_32f_C1R_L, (const Ipp32f* pSrc, IppSizeL srcStep, Ipp32f* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, IppiBorderType border, const Ipp32f* pBorderValue, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiResizeLinear_32f_C3R_L, (const Ipp32f* pSrc, IppSizeL srcStep, Ipp32f* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, IppiBorderType border, const Ipp32f* pBorderValue, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiResizeLinear_32f_C4R_L, (const Ipp32f* pSrc, IppSizeL srcStep, Ipp32f* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, IppiBorderType border, const Ipp32f* pBorderValue, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiResizeLinear_64f_C1R_L, (const Ipp64f* pSrc, IppSizeL srcStep, Ipp64f* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, IppiBorderType border, const Ipp64f* pBorderValue, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiResizeLinear_64f_C3R_L, (const Ipp64f* pSrc, IppSizeL srcStep, Ipp64f* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, IppiBorderType border, const Ipp64f* pBorderValue, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiResizeLinear_64f_C4R_L, (const Ipp64f* pSrc, IppSizeL srcStep, Ipp64f* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, IppiBorderType border, const Ipp64f* pBorderValue, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))

IPPAPI (IppStatus, ippiResizeCubic_8u_C1R_L,  (const Ipp8u* pSrc, IppSizeL srcStep, Ipp8u* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, IppiBorderType border, const Ipp8u* pBorderValue, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiResizeCubic_8u_C3R_L,  (const Ipp8u* pSrc, IppSizeL srcStep, Ipp8u* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, IppiBorderType border, const Ipp8u* pBorderValue, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiResizeCubic_8u_C4R_L,  (const Ipp8u* pSrc, IppSizeL srcStep, Ipp8u* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, IppiBorderType border, const Ipp8u* pBorderValue, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiResizeCubic_16u_C1R_L, (const Ipp16u* pSrc, IppSizeL srcStep, Ipp16u* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, IppiBorderType border, const Ipp16u* pBorderValue, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiResizeCubic_16u_C3R_L, (const Ipp16u* pSrc, IppSizeL srcStep, Ipp16u* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, IppiBorderType border, const Ipp16u* pBorderValue, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiResizeCubic_16u_C4R_L, (const Ipp16u* pSrc, IppSizeL srcStep, Ipp16u* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, IppiBorderType border, const Ipp16u* pBorderValue, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiResizeCubic_16s_C1R_L, (const Ipp16s* pSrc, IppSizeL srcStep, Ipp16s* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, IppiBorderType border, const Ipp16s* pBorderValue, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiResizeCubic_16s_C3R_L, (const Ipp16s* pSrc, IppSizeL srcStep, Ipp16s* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, IppiBorderType border, const Ipp16s* pBorderValue, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiResizeCubic_16s_C4R_L, (const Ipp16s* pSrc, IppSizeL srcStep, Ipp16s* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, IppiBorderType border, const Ipp16s* pBorderValue, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiResizeCubic_32f_C1R_L, (const Ipp32f* pSrc, IppSizeL srcStep, Ipp32f* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, IppiBorderType border, const Ipp32f* pBorderValue, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiResizeCubic_32f_C3R_L, (const Ipp32f* pSrc, IppSizeL srcStep, Ipp32f* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, IppiBorderType border, const Ipp32f* pBorderValue, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiResizeCubic_32f_C4R_L, (const Ipp32f* pSrc, IppSizeL srcStep, Ipp32f* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, IppiBorderType border, const Ipp32f* pBorderValue, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))

IPPAPI (IppStatus, ippiResizeLanczos_8u_C1R_L, (const Ipp8u* pSrc, IppSizeL srcStep, Ipp8u* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, IppiBorderType border, const Ipp8u* pBorderValue, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiResizeLanczos_8u_C3R_L, (const Ipp8u* pSrc, IppSizeL srcStep, Ipp8u* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, IppiBorderType border, const Ipp8u* pBorderValue, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiResizeLanczos_8u_C4R_L, (const Ipp8u* pSrc, IppSizeL srcStep, Ipp8u* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, IppiBorderType border, const Ipp8u* pBorderValue, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiResizeLanczos_16u_C1R_L, (const Ipp16u* pSrc, IppSizeL srcStep, Ipp16u* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, IppiBorderType border, const Ipp16u* pBorderValue, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiResizeLanczos_16u_C3R_L, (const Ipp16u* pSrc, IppSizeL srcStep, Ipp16u* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, IppiBorderType border, const Ipp16u* pBorderValue, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiResizeLanczos_16u_C4R_L, (const Ipp16u* pSrc, IppSizeL srcStep, Ipp16u* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, IppiBorderType border, const Ipp16u* pBorderValue, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiResizeLanczos_16s_C1R_L, (const Ipp16s* pSrc, IppSizeL srcStep, Ipp16s* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, IppiBorderType border, const Ipp16s* pBorderValue, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiResizeLanczos_16s_C3R_L, (const Ipp16s* pSrc, IppSizeL srcStep, Ipp16s* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, IppiBorderType border, const Ipp16s* pBorderValue, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiResizeLanczos_16s_C4R_L, (const Ipp16s* pSrc, IppSizeL srcStep, Ipp16s* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, IppiBorderType border, const Ipp16s* pBorderValue, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiResizeLanczos_32f_C1R_L, (const Ipp32f* pSrc, IppSizeL srcStep, Ipp32f* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, IppiBorderType border, const Ipp32f* pBorderValue, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiResizeLanczos_32f_C3R_L, (const Ipp32f* pSrc, IppSizeL srcStep, Ipp32f* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, IppiBorderType border, const Ipp32f* pBorderValue, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiResizeLanczos_32f_C4R_L, (const Ipp32f* pSrc, IppSizeL srcStep, Ipp32f* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, IppiBorderType border, const Ipp32f* pBorderValue, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))

IPPAPI (IppStatus, ippiResizeSuper_8u_C1R_L,  (const Ipp8u* pSrc, IppSizeL srcStep, Ipp8u* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiResizeSuper_8u_C3R_L,  (const Ipp8u* pSrc, IppSizeL srcStep, Ipp8u* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiResizeSuper_8u_C4R_L,  (const Ipp8u* pSrc, IppSizeL srcStep, Ipp8u* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiResizeSuper_16u_C1R_L, (const Ipp16u* pSrc, IppSizeL srcStep, Ipp16u* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiResizeSuper_16u_C3R_L, (const Ipp16u* pSrc, IppSizeL srcStep, Ipp16u* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiResizeSuper_16u_C4R_L, (const Ipp16u* pSrc, IppSizeL srcStep, Ipp16u* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiResizeSuper_16s_C1R_L, (const Ipp16s* pSrc, IppSizeL srcStep, Ipp16s* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiResizeSuper_16s_C3R_L, (const Ipp16s* pSrc, IppSizeL srcStep, Ipp16s* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiResizeSuper_16s_C4R_L, (const Ipp16s* pSrc, IppSizeL srcStep, Ipp16s* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiResizeSuper_32f_C1R_L, (const Ipp32f* pSrc, IppSizeL srcStep, Ipp32f* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiResizeSuper_32f_C3R_L, (const Ipp32f* pSrc, IppSizeL srcStep, Ipp32f* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiResizeSuper_32f_C4R_L, (const Ipp32f* pSrc, IppSizeL srcStep, Ipp32f* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))

/* /////////////////////////////////////////////////////////////////////////////
//  Name:               ippiResizeNearestAntialiasingInit_L
//                      ippiResizeLinearAntialiasingInit_L
//                      ippiResizeCubicAntialiasingInit_L
//
//  Purpose:            Initializes the Spec structure for the Resize transform
//                      with antialiasing by different interpolation methods
//
//  Parameters:
//    srcSize           Size of the input image (in pixels)
//    dstSize           Size of the output image (in pixels)
//    dataType          Data type {ipp8u|ipp16u|ipp16s|ipp32f}
//    valueB            The first parameter (B) for specifying Cubic filters
//    valueC            The second parameter (C) for specifying Cubic filters
//    numLobes          The parameter for specifying Lanczos (2 or 3) or Hahn (3 or 4) filters
//    pInitBuf          Pointer to the temporal buffer for several filter initialization
//    pSpec             Pointer to the Spec structure for resize filter
//
//  Return Values:
//    ippStsNoErr               Indicates no error
//    ippStsNullPtrErr          Indicates an error if one of the specified pointers is NULL
//    ippStsNoOperation         Indicates a warning if width or height of any image is zero
//    ippStsSizeErr             Indicates an error if width or height of the source image is negative
//    ippStsExceededSizeErr     Indicates an error if one of width or height of the destination image or
//                              the source image with borders exceeds 536870911 (0x1FFFFFFF)
//    ippStsNotSupportedModeErr Indicates an error if the requested mode is not supported.
//
//  Notes/References:
//    1. The equation shows the family of cubic filters:
//           ((12-9B-6C)*|x|^3 + (-18+12B+6C)*|x|^2                  + (6-2B)  ) / 6   for |x| < 1
//    K(x) = ((   -B-6C)*|x|^3 + (    6B+30C)*|x|^2 + (-12B-48C)*|x| + (8B+24C)) / 6   for 1 <= |x| < 2
//           0   elsewhere
//    Some values of (B,C) correspond to known cubic splines: Catmull-Rom (B=0,C=0.5), B-Spline (B=1,C=0) and other.
//      Mitchell, Don P.; Netravali, Arun N. (Aug. 1988). "Reconstruction filters in computer graphics"
//      http://www.mentallandscape.com/Papers_siggraph88.pdf
//
//    2. Hahn filter does not supported now.
//    3. The implemented interpolation algorithms have the following filter sizes:
//       Linear 2x2, Cubic 4x4, 2-lobed Lanczos 4x4, 3-lobed Lanczos 6x6.
*/

IPPAPI (IppStatus, ippiResizeAntialiasingLinearInit_L, (IppiSizeL srcSize, IppiSizeL dstSize, IppDataType dataType, IppiResizeSpec* pSpec, Ipp8u* pInitBuf))
IPPAPI (IppStatus, ippiResizeAntialiasingCubicInit_L, (IppiSizeL srcSize, IppiSizeL dstSize, IppDataType dataType, Ipp32f valueB, Ipp32f valueC, IppiResizeSpec* pSpec, Ipp8u* pInitBuf))
IPPAPI (IppStatus, ippiResizeAntialiasingLanczosInit_L, (IppiSizeL srcSize, IppiSizeL dstSize, IppDataType dataType, Ipp32u numLobes, IppiResizeSpec* pSpec, Ipp8u* pInitBuf))


/* /////////////////////////////////////////////////////////////////////////////
//  Name:               ippiResizeAntialiasing
//
//  Purpose:            Changes an image size by different interpolation methods with antialiasing technique
//
//  Parameters:
//    pSrc              Pointer to the source image
//    srcStep           Distance (in bytes) between of consecutive lines in the source image
//    pDst              Pointer to the destination image
//    dstStep           Distance (in bytes) between of consecutive lines in the destination image
//    dstOffset         Offset of tiled image respectively destination image origin
//    dstSize           Size of the destination image (in pixels)
//    border            Type of the border
//    borderValue       Pointer to the constant value(s) if border type equals ippBorderConstant
//    pSpec             Pointer to the Spec structure for resize filter
//    pBuffer           Pointer to the work buffer
//
//  Return Values:
//    ippStsNoErr               Indicates no error
//    ippStsNullPtrErr          Indicates an error if one of the specified pointers is NULL
//    ippStsBorderErr           Indicates an error if border type has an illegal value
//    ippStsContextMatchErr     Indicates an error if pointer to an invalid pSpec structure is passed
//    ippStsNotSupportedModeErr Indicates an error if requested mode is currently not supported
//    ippStsStepErr             Indicates an error if the step value is not data type multiple
//    ippStsOutOfRangeErr       Indicates an error if the destination image offset point is outside the
//                              destination image origin
//    ippStsSizeWrn             Indicates a warning if the destination image size is more than
//                              the destination image origin size
//
//  Notes:
//    1. Supported border types are ippBorderInMemory and ippBorderReplicate.
//    2. Hahn filter does not supported now.
*/
IPPAPI (IppStatus, ippiResizeAntialiasing_8u_C1R_L, (const Ipp8u* pSrc, IppSizeL srcStep, Ipp8u* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, IppiBorderType border, Ipp8u* pBorderValue, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiResizeAntialiasing_8u_C3R_L, (const Ipp8u* pSrc, IppSizeL srcStep, Ipp8u* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, IppiBorderType border, Ipp8u* pBorderValue, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiResizeAntialiasing_8u_C4R_L, (const Ipp8u* pSrc, IppSizeL srcStep, Ipp8u* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, IppiBorderType border, Ipp8u* pBorderValue, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiResizeAntialiasing_16u_C1R_L, (const Ipp16u* pSrc, IppSizeL srcStep, Ipp16u* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, IppiBorderType border, Ipp16u* pBorderValue, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiResizeAntialiasing_16u_C3R_L, (const Ipp16u* pSrc, IppSizeL srcStep, Ipp16u* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, IppiBorderType border, Ipp16u* pBorderValue, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiResizeAntialiasing_16u_C4R_L, (const Ipp16u* pSrc, IppSizeL srcStep, Ipp16u* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, IppiBorderType border, Ipp16u* pBorderValue, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiResizeAntialiasing_16s_C1R_L, (const Ipp16s* pSrc, IppSizeL srcStep, Ipp16s* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, IppiBorderType border, Ipp16s* pBorderValue, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiResizeAntialiasing_16s_C3R_L, (const Ipp16s* pSrc, IppSizeL srcStep, Ipp16s* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, IppiBorderType border, Ipp16s* pBorderValue, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiResizeAntialiasing_16s_C4R_L, (const Ipp16s* pSrc, IppSizeL srcStep, Ipp16s* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, IppiBorderType border, Ipp16s* pBorderValue, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiResizeAntialiasing_32f_C1R_L, (const Ipp32f* pSrc, IppSizeL srcStep, Ipp32f* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, IppiBorderType border, Ipp32f* pBorderValue, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiResizeAntialiasing_32f_C3R_L, (const Ipp32f* pSrc, IppSizeL srcStep, Ipp32f* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, IppiBorderType border, Ipp32f* pBorderValue, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))
IPPAPI (IppStatus, ippiResizeAntialiasing_32f_C4R_L, (const Ipp32f* pSrc, IppSizeL srcStep, Ipp32f* pDst, IppSizeL dstStep, IppiPointL dstOffset, IppiSizeL dstSize, IppiBorderType border, Ipp32f* pBorderValue, const IppiResizeSpec* pSpec, Ipp8u* pBuffer))



/* =============================================================================
                            ippCV
============================================================================= */

/* ///////////////////////////////////////////////////////////////////////////////////////
//  Name:       ippiFilterGaussianGetBufferSize
//
//  Purpose:    Computes the size of the working buffer for the Gaussian filter
//
//  Return:
//     ippStsNoErr          Ok. Any other value indicates an error or a warning.
//     ippStsNullPtrErr     One of the pointers is NULL.
//     ippStsSizeErr        maxRoiSize  has a field with zero or negative value.
//     ippStsDataTypeErr    Indicates an error when dataType has an illegal value.
//     ippStsBadArgErr      Indicates an error if kernelSize is even or is less than 3.
//     ippStsChannelErr     Indicates an error when numChannels has an illegal value.
//
//  Arguments:
//     maxRoiSize           Maximal size of the image ROI in pixels.
//     kernelSize           Size of the Gaussian kernel (odd, greater or equal to 3).
//     dataType             Data type of the source and destination images.
//     numChannels          Number of channels in the images. Possible values are 1 and 3.
//     pSpecSize            Pointer to the computed size (in bytes) of the Gaussian
//                            specification structure.
//     pBufferSize          Pointer to the computed size (in bytes) of the external buffer.
*/
IPPAPI(IppStatus, ippiFilterGaussianGetBufferSize_L,(IppiSizeL maxRoiSize, int kernelSize,
                  IppDataType dataType, IppiBorderType borderType, int numChannels, IppSizeL* pBufferSize))

/* ///////////////////////////////////////////////////////////////////////////////////////
//  Name:       ippiFilterGaussianGetSpecSize_L
//
//  Purpose:    Computes the size of the working buffer for the Gaussian filter GaussianSpec
//
//  Return:
//     ippStsNoErr          Ok. Any other value indicates an error or a warning.
//     ippStsNullPtrErr     One of the pointers is NULL.
//     ippStsSizeErr        maxRoiSize  has a field with zero or negative value.
//     ippStsDataTypeErr    Indicates an error when dataType has an illegal value.
//     ippStsBadArgErr      Indicates an error if kernelSize is even or is less than 3.
//     ippStsChannelErr     Indicates an error when numChannels has an illegal value.
//
//  Arguments:
//     kernelSize           Size of the Gaussian kernel (odd, greater or equal to 3).
//     dataType             Data type of the source and destination images.
//     numChannels          Number of channels in the images. Possible values are 1 and 3.
//     pSpecSize            Pointer to the computed size (in bytes) of the Gaussian
//                            specification structure.
//     pInitBufferSize      Pointer to the computed size (in bytes) of the external buffer for the Gaussian filter GaussianSpec.
*/
IPPAPI(IppStatus, ippiFilterGaussianGetSpecSize_L,(int kernelSize, IppDataType dataType, int numChannels, IppSizeL *pSpecSize, IppSizeL* pInitBufferSize))


/* ///////////////////////////////////////////////////////////////////////////////////////
//  Name:       ippiFilterGaussianInit
//
//  Purpose:    initialization of Spec for Gaussian filter
//
//  Return:
//     ippStsNoErr          Ok. Any other value indicates an error or a warning.
//     ippStsNullPtrErr     One of the pointers is NULL.
//     ippStsSizeErr        roiSize has a field with zero or negative value.
//     ippStsDataTypeErr    Indicates an error when borderType has an illegal value.
//     ippStsBadArgErr      kernelSize is even or is less than 3.
//     ippStsChannelErr     Indicates an error when numChannels has an illegal value.
//     ippStsBorderErr      Indicates an error condition if borderType has a illegal
//                           value.
//
//  Arguments:
//     roiSize              Size of the image ROI in pixels.
//     kernelSize           Size of the Gaussian kernel (odd, greater or equal to 3).
//     sigma                Standard deviation of the Gaussian kernel.
//     borderType           One of border supported types.
//     dataType             Data type of the source and destination images.
//     numChannels          Number of channels in the images. Possible values are 1 and 3.
//     pSpec                Pointer to the Spec.
//     pBuffer              Pointer to the buffer:
*/


IPPAPI(IppStatus, ippiFilterGaussianInit_L,(IppiSizeL roiSize, int kernelSize, Ipp32f sigma, IppiBorderType borderType, IppDataType dataType, int numChannels,
       IppFilterGaussianSpec* pSpec, Ipp8u* pInitBuffer))


/* ///////////////////////////////////////////////////////////////////////////////////////
//  Name:       ippiFilterGaussian
//
//  Purpose:    Applies Gaussian filter with borders
//
//  Return:
//     ippStsNoErr      Ok. Any other value indicates an error or a warning.
//     ippStsNullPtrErr One of the specified pointers is NULL.
//     ippStsSizeErr    roiSize has a field with zero or negative value.
//     ippStsStepErr    Indicates an error condition if srcStep or dstStep is less
//                        than  roiSize.width * <pixelSize>.
//     ippStsNotEvenStepErr One of the step values is not divisible by 4 for floating-point images.
//     ippStsBadArgErr  kernelSize is less than 3 or sigma is less or equal than 0.
//
//  Arguments:
//     pSrc             Pointer to the source image ROI.
//     srcStep          Distance in bytes between starts of consecutive lines in the source image.
//     pDst             Pointer to the destination image ROI.
//     dstStep          Distance in bytes between starts of consecutive lines in the destination image.
//     roiSize          Size of the source and destination image ROI.
//     borderType       One of border supported types.
//     borderValue      Constant value to assign to pixels of the constant border. if border type equals ippBorderConstant
//     pSpec            Pointer to the Gaussian specification structure.
//     pBuffer          Pointer to the working buffer.
*/

//////////////////////////////////////////////////////////////////////////////////////////

IPPAPI(IppStatus, ippiFilterGaussian_32f_C1R_L,(const Ipp32f* pSrc, IppSizeL srcStep, 
      Ipp32f* pDst, IppSizeL dstStep, IppiSizeL roiSize, IppiBorderType borderType, const Ipp32f borderValue[1],
      IppFilterGaussianSpec* pSpec, Ipp8u* pBuffer))

IPPAPI(IppStatus, ippiFilterGaussian_16u_C1R_L,(const Ipp16u * pSrc, IppSizeL srcStep,
      Ipp16u * pDst, IppSizeL dstStep, IppiSizeL roiSize, IppiBorderType borderType,  const Ipp16u borderValue[1],
      IppFilterGaussianSpec* pSpec, Ipp8u* pBuffer))

IPPAPI(IppStatus, ippiFilterGaussian_16s_C1R_L,(const Ipp16s* pSrc, IppSizeL srcStep,
      Ipp16s* pDst, IppSizeL dstStep, IppiSizeL roiSize, IppiBorderType borderType,  const Ipp16s borderValue[1],
      IppFilterGaussianSpec* pSpec, Ipp8u* pBuffer))

IPPAPI(IppStatus, ippiFilterGaussian_8u_C1R_L,(const Ipp8u* pSrc, IppSizeL srcStep,
      Ipp8u* pDst, IppSizeL dstStep, IppiSizeL roiSize, IppiBorderType borderType,  const Ipp8u borderValue[1],
      IppFilterGaussianSpec* pSpec, Ipp8u* pBuffer))

IPPAPI(IppStatus, ippiFilterGaussian_32f_C3R_L,(const Ipp32f* pSrc, IppSizeL srcStep,
      Ipp32f* pDst, IppSizeL dstStep, IppiSizeL roiSize, IppiBorderType borderType,  const Ipp32f borderValue[3],
      IppFilterGaussianSpec* pSpec, Ipp8u* pBuffer))

IPPAPI(IppStatus, ippiFilterGaussian_16u_C3R_L,(const Ipp16u * pSrc, IppSizeL srcStep,
      Ipp16u * pDst, IppSizeL dstStep, IppiSizeL roiSize, IppiBorderType borderType,  const Ipp16u borderValue[3],
      IppFilterGaussianSpec* pSpec, Ipp8u* pBuffer))

IPPAPI(IppStatus, ippiFilterGaussian_16s_C3R_L,(const Ipp16s* pSrc, IppSizeL srcStep,
      Ipp16s* pDst, IppSizeL dstStep, IppiSizeL roiSize, IppiBorderType borderType,  const Ipp16s borderValue[3],
      IppFilterGaussianSpec* pSpec, Ipp8u* pBuffer))

IPPAPI(IppStatus, ippiFilterGaussian_8u_C3R_L,(const Ipp8u* pSrc, IppSizeL srcStep,
      Ipp8u* pDst, IppSizeL dstStep, IppiSizeL roiSize, IppiBorderType borderType,  const Ipp8u borderValue[3],
      IppFilterGaussianSpec* pSpec, Ipp8u* pBuffer))

/****************************************************************************************\
*                                Morphological Operations                                *
\****************************************************************************************/

/* ///////////////////////////////////////////////////////////////////////////////////////
//  Name:   ippiDilateGetBufferSize_L,   ippiErodeGetBufferSize_L
//
//
//  Purpose:  Gets the size of the internal state or specification structure for morphological operations.
//
//  Return:
//    ippStsNoErr              Ok.
//    ippStsNullPtrErr         One of the pointers is NULL.
//    ippStsSizeErr            Width of the image, or width or height of the structuring
//                             element is less than,or equal to zero.
//
//  Parameters:
//    roiSize                  Size of the source and destination image ROI in pixels.
//    maskSize                 Size of the structuring element.
//    dataType                 The type of data
//    numChannels              The number of channels
//    pBufferSize              Pointer to the buffer size value for the morphological initialization function.
*/

IPPAPI(IppStatus, ippiDilateGetBufferSize_L, (IppiSizeL roiSize,  IppiSizeL maskSize, IppDataType datatype, int numChannels, IppSizeL* pBufferSize))

IPPAPI(IppStatus, ippiErodeGetBufferSize_L, (IppiSizeL roiSize,  IppiSizeL maskSize, IppDataType datatype, int numChannels, IppSizeL* pBufferSize))


/* ///////////////////////////////////////////////////////////////////////////////////////
//  Name:   ippiDilateGetSpecSize_L,     ippiErodeGetSpecSize_L
//
//
//  Purpose:  Gets the size of the internal state or specification structure for morphological operations.
//
//  Return:
//    ippStsNoErr              Ok.
//    ippStsNullPtrErr         One of the pointers is NULL.
//    ippStsSizeErr            Width of the image, or width or height of the structuring
//                             element is less than,or equal to zero.
//
//  Parameters:
//    roiSize                  Size of the source and destination image ROI in pixels.
//    maskSize                 Size of the structuring element.
//    pSpecSize                Pointer to the specification structure size.
*/

IPPAPI(IppStatus, ippiDilateGetSpecSize_L,(IppiSizeL roiSize, IppiSizeL maskSize, IppSizeL* pSpecSize))
IPPAPI(IppStatus, ippiErodeGetSpecSize_L,(IppiSizeL roiSize, IppiSizeL maskSize, IppSizeL* pSpecSize))


/* ///////////////////////////////////////////////////////////////////////////////////////
//  Name:   ippiDilateInit_L,        ippiErodeInit_L
//
//  Purpose:  Initialize the internal state or specification structure for morphological operation.
//
//  Return:
//    ippStsNoErr              Ok.
//    ippStsNullPtrErr         One of the pointers is NULL.
//    ippStsSizeErr            Width of the image or width or height of the structuring
//                             element is less than, or equal to zero.
//    ippStsAnchorErr          Anchor point is outside the structuring element.
//
//  Parameters:
//    roiSize                  Size of the source and destination image ROI in pixels.
//    pMask                    Pointer to the structuring element (mask).
//    maskSize                 Size of the structuring element.
//    pMorphSpec               Pointer to the morphology specification structure.
*/


IPPAPI(IppStatus, ippiDilateInit_L,(IppiSizeL roiSize, const Ipp8u* pMask, IppiSizeL maskSize, IppiMorphStateL* pMorphSpec))
IPPAPI(IppStatus, ippiErodeInit_L,(IppiSizeL roiSize, const Ipp8u* pMask, IppiSizeL maskSize, IppiMorphStateL* pMorphSpec))


/* ///////////////////////////////////////////////////////////////////////////////////////
//  Name:   ippiDilate_8u_C1R,    ippiDilate_8u_C3R,
//          ippiDilate_8u_C4R,    ippiDilate_32f_C1R,
//          ippiDilate_32f_C3R,   ippiDilate_32f_C4R
//
//          ippiErode_8u_C1R,     ippiErode_8u_C3R,
//          ippiErode_8u_C4R,     ippiErode_32f_C1R,
//          ippiErode_32f_C3R,    ippiErode_32f_C4R,
//
//          ippiDilate_16u_C1R,            ippiDilate_16s_C1R,
//          ippiDilate_1u_C1R
//
//  Purpose:    Perform erosion/dilation of the image arbitrary shape structuring element.
//
//  Return:
//    ippStsNoErr              Ok.
//    ippStsNullPtrErr         One of the pointers is NULL.
//    ippStsSizeErr            The ROI width or height is less than 1,
//                             or ROI width is bigger than ROI width in the state structure.
//    ippStsStepErr            Step is too small to fit the image.
//    ippStsNotEvenStepErr     Step is not multiple of the element.
//    ippStsBadArgErr          Incorrect border type.
//

//  Parameters:
//    pSrc                     Pointer to the source image.
//    srcStep                  Step in the source image.
//    pDst                     Pointer to the destination image.
//    dstStep                  Step in the destination image.
//    roiSize                  Size of the source and destination image ROI in pixels.
//    borderType               Type of border (ippBorderRepl now).
//    borderValue              Pointer to the vector of values for the constant border.
//    pMorphSpec               Pointer to the morphology specification structure.
//    pBuffer                  Pointer to the external work buffer.
*/

IPPAPI(IppStatus, ippiDilate_8u_C1R_L, (const Ipp8u* pSrc, IppSizeL srcStep,
                                   Ipp8u* pDst, IppSizeL dstStep, IppiSizeL roiSize,
                                   IppiBorderType borderType, const Ipp8u borderValue[1], const IppiMorphStateL* pMorphSpec, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiDilate_8u_C3R_L, (const Ipp8u* pSrc, IppSizeL srcStep,
                                   Ipp8u* pDst, IppSizeL dstStep, IppiSizeL roiSize,
                                   IppiBorderType borderType, const Ipp8u borderValue[3], const IppiMorphStateL* pMorphSpec, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiDilate_8u_C4R_L, (const Ipp8u* pSrc, IppSizeL srcStep,
                                   Ipp8u* pDst, IppSizeL dstStep, IppiSizeL roiSize,
                                   IppiBorderType borderType, const Ipp8u borderValue[4], const IppiMorphStateL* pMorphSpec, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiDilate_32f_C1R_L, (const Ipp32f* pSrc, IppSizeL srcStep,
                                   Ipp32f* pDst, IppSizeL dstStep, IppiSizeL roiSize,
                                   IppiBorderType borderType, const Ipp32f borderValue[1], const IppiMorphStateL* pMorphSpec, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiDilate_32f_C3R_L, (const Ipp32f* pSrc, IppSizeL srcStep,
                                   Ipp32f* pDst, IppSizeL dstStep, IppiSizeL roiSize,
                                   IppiBorderType borderType, const Ipp32f borderValue[3], const IppiMorphStateL* pMorphSpec, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiDilate_32f_C4R_L, (const Ipp32f* pSrc, IppSizeL srcStep,
                                   Ipp32f* pDst, IppSizeL dstStep, IppiSizeL roiSize,
                                   IppiBorderType borderType, const Ipp32f borderValue[4], const IppiMorphStateL* pMorphSpec, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiDilate_1u_C1R_L,( const Ipp8u* pSrc, IppSizeL srcStep, int srcBitOffset, Ipp8u* pDst, IppSizeL dstStep, int dstBitOffset,
                                   IppiSizeL roiSize, IppiBorderType borderType, const Ipp8u borderValue[1], const IppiMorphStateL* pMorphSpec, Ipp8u* pBuffer ))
IPPAPI(IppStatus, ippiDilate_16u_C1R_L,(const Ipp16u* pSrc, IppSizeL srcStep,
                                   Ipp16u* pDst, IppSizeL dstStep, IppiSizeL roiSize,
                                   IppiBorderType borderType, const Ipp16u borderValue[1], const IppiMorphStateL* pMorphSpec, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiDilate_16s_C1R_L,(const Ipp16s* pSrc, IppSizeL srcStep,
                                   Ipp16s* pDst, IppSizeL dstStep, IppiSizeL roiSize,
                                   IppiBorderType borderType, const Ipp16s borderValue[1], const IppiMorphStateL* pMorphSpec, Ipp8u* pBuffer))

IPPAPI(IppStatus, ippiErode_8u_C1R_L, (const Ipp8u* pSrc, IppSizeL srcStep,
                                   Ipp8u* pDst, IppSizeL dstStep, IppiSizeL roiSize,
                                   IppiBorderType borderType,const  Ipp8u borderValue[1], const IppiMorphStateL* pMorphSpec, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiErode_8u_C3R_L, (const Ipp8u* pSrc, IppSizeL srcStep,
                                   Ipp8u* pDst, IppSizeL dstStep, IppiSizeL roiSize,
                                   IppiBorderType borderType, const Ipp8u borderValue[3], const IppiMorphStateL* pMorphSpec, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiErode_8u_C4R_L, (const Ipp8u* pSrc, IppSizeL srcStep,
                                   Ipp8u* pDst, IppSizeL dstStep, IppiSizeL roiSize,
                                   IppiBorderType borderType, const Ipp8u borderValue[4], const IppiMorphStateL* pMorphSpec, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiErode_32f_C1R_L, (const Ipp32f* pSrc, IppSizeL srcStep,
                                   Ipp32f* pDst, IppSizeL dstStep, IppiSizeL roiSize,
                                   IppiBorderType borderType, const Ipp32f borderValue[1], const IppiMorphStateL* pMorphSpec, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiErode_32f_C3R_L, (const Ipp32f* pSrc, IppSizeL srcStep,
                                   Ipp32f* pDst, IppSizeL dstStep, IppiSizeL roiSize,
                                   IppiBorderType borderType, const Ipp32f borderValue[3], const IppiMorphStateL* pMorphSpec, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiErode_32f_C4R_L, (const Ipp32f* pSrc, IppSizeL srcStep,
                                   Ipp32f* pDst, IppSizeL dstStep, IppiSizeL roiSize,
                                   IppiBorderType borderType, const Ipp32f borderValue[4], const IppiMorphStateL* pMorphSpec, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiErode_16u_C1R_L,(const Ipp16u* pSrc, IppSizeL srcStep,
                                   Ipp16u* pDst, IppSizeL dstStep, IppiSizeL roiSize,
                                   IppiBorderType borderType, const Ipp16u borderValue[1], const IppiMorphStateL* pMorphSpec, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiErode_16s_C1R_L,(const Ipp16s* pSrc, IppSizeL srcStep,
                                   Ipp16s* pDst, IppSizeL dstStep, IppiSizeL roiSize,
                                   IppiBorderType borderType, const Ipp16s borderValue[1], const IppiMorphStateL* pMorphSpec, Ipp8u* pBuffer))
IPPAPI(IppStatus, ippiErode_1u_C1R_L,( const Ipp8u* pSrc, IppSizeL srcStep, int srcBitOffset, Ipp8u* pDst, IppSizeL dstStep, int dstBitOffset,
                                   IppiSizeL roiSize, IppiBorderType borderType, const Ipp8u borderValue[1], const IppiMorphStateL* pMorphSpec, Ipp8u* pBuffer ))



/****************************************************************************************\
*                       Advanced Morphological Operations                                *
\****************************************************************************************/


/////////////////////////////////////////////////////////////

/* ///////////////////////////////////////////////////////////////////////////////////////
//  Name:   ippiMorphGetSpecSize_L
//
//  Purpose:  Gets the size of the internal state or specification structure for advanced morphological operations.
//
//  Return:
//    ippStsNoErr              Ok.
//    ippStsNullPtrErr         One of the pointers is NULL.
//    ippStsSizeErr            Width of the image, or width or height of the structuring.
//                             element is less than, or equal to zero.
//
//  Parameters:
//    roiSize                  Maximum size of the image ROI, in pixels.
//    maskSize                 Size of the structuring element.
//    dataType                 The type of data
//    numChannels              The number of channels
//    pSpecSize                Pointer to the specification structure size.
*/

IPPAPI(IppStatus, ippiMorphGetSpecSize_L,(IppiSizeL roiSize, IppiSizeL maskSize, IppDataType depth, int numChannels, IppSizeL* pSpecSize))


/* ///////////////////////////////////////////////////////////////////////////////////////
//  Name:   ippiMorphGetBufferSize_L
//
//  Purpose:  Gets the size of the work buffer for the advanced morphological operations.
//
//  Return:
//    ippStsNoErr              Ok.
//    ippStsNullPtrErr         One of the pointers is NULL.
//    ippStsSizeErr            Width of the image, or width or height of the structuring.
//                             element is less than, or equal to zero.
//
//  Parameters:
//    roiSize                  Maximum size of the image ROI, in pixels.
//    maskSize                 Size of the structuring element.
//    dataType                 The type of data
//    numChannels              The number of channels
//    pBufferSize              Pointer to the buffer size value for the morphology initialization function.
*/

IPPAPI(IppStatus, ippiMorphGetBufferSize_L,(IppiSizeL roiSize, IppiSizeL maskSize, IppDataType depth, int numChannels, IppSizeL* bufferSize))



/* ///////////////////////////////////////////////////////////////////////////////////////
//  Name:   ippiMorphInit_L
//
//  Purpose:  Initialize the internal state or specification structure for advanced morphological operations.
//
//  Return:
//    ippStsNoErr              Ok.
//    ippStsNullPtrErr         One of the pointers is NULL.
//    ippStsSizeErr            Width of the image or width or height of the structuring
//                             element is less than, or equal to zero.
//    ippStsAnchorErr          Anchor point is outside the structuring element.
//
//  Parameters:
//    roiSize                  Maximum size of the image ROI, in pixels.
//    pMask                    Pointer to the structuring element (mask).
//    maskSize                 Size of the structuring element.
//    dataType                 The type of data
//    numChannels              The number of channels
//    pMorphSpec               Pointer to the advanced morphology specification structure.
*/

IPPAPI( IppStatus, ippiMorphInit_L,( IppiSizeL roiSize, const Ipp8u* pMask, IppiSizeL maskSize, IppDataType depth, int numChannels, 
                                              IppiMorphAdvStateL* pMorphSpec))


/* ///////////////////////////////////////////////////////////////////////////////////////
//  Name:   ippiMorphClose_8u_C1R_L,             ippiMorphClose_8u_C3R_L,
//          ippiMorphClose_8u_C4R_L,             ippiMorphClose_32f_C1R_L,
//          ippiMorphClose_32f_C3R_L,            ippiMorphClose_32f_C4R_L
//
//          ippiMorphOpen_8u_C1R_L,              ippiMorphOpen_8u_C3R_L,
//          ippiMorphOpen_8u_C4R_L,              ippiMorphOpen_32f_C1R_L,
//          ippiMorphOpen_32f_C3R_L,             ippiMorphOpen_32f_C4R_L,
//
//          ippiMorphClose_16u_C1R_L,            ippiMorphOpen_16u_C1R_L,
//          ippiMorphClose_16s_C1R_L,            ippiMorphOpen_16s_C1R_L,
//          ippiMorphClose_1u_C1R_L,             ippiMorphOpen_1u_C1R_L,

//          ippiMorphTophat_8u_C1R_L,            ippiMorphTophat_8u_C3R_L,
//          ippiMorphTophat_8u_C4R_L,            ippiMorphTophat_32f_C1R_L,
//          ippiMorphTophat_32f_C3R_L,           ippiMorphTophat_32f_C4R_L,
//
//          ippiMorphBlackhat_8u_C1R_L,          ippiMorphBlackhat_8u_C3R_L,
//          ippiMorphBlackhat_8u_C4R_L,          ippiMorphBlackhat_32f_C1R_L,
//          ippiMorphBlackhat_32f_C3R_L,         ippiMorphBlackhat_32f_C4R_L,
//
//          ippiMorphGradient_8u_C1R_L,     ippiMorphGradient_8u_C3R_L,
//          ippiMorphGradient_8u_C4R_L,     ippiMorphGradient_32f_C1R_L,
//          ippiMorphGradient_32f_C3R_L,    ippiMorphGradient_32f_C4R_L,
//
//  Purpose:    Perform advanced morphologcal operations on the image arbitrary shape structuring element.
//
//  Return:
//    ippStsNoErr              Ok.
//    ippStsNullPtrErr         One of the pointers is NULL.
//    ippStsSizeErr            The ROI width or height is less than 1,
//                             or ROI width is bigger than ROI width in the state structure.
//    ippStsStepErr            Step is too small to fit the image.
//    ippStsNotEvenStepErr     Step is not multiple of the element.
//    ippStsBadArgErr          Incorrect border type.
//
//  Parameters:
//    pSrc                     Pointer to the source image.
//    srcStep                  Step in the source image.
//    pDst                     Pointer to the destination image.
//    dstStep                  Step in the destination image.
//    roiSize                  ROI size.
//    borderType               Type of border (ippBorderRepl now).
//    borderValue              Value for the constant border.
//    pMorphSpec               Pointer to the morphology specification structure.
//    pBuffer                  Pointer to the external work buffer.
*/

/////////////////////////////////////////////////////////////////////////

IPPAPI( IppStatus, ippiMorphOpen_16u_C1R_L,(
                const Ipp16u* pSrc, IppSizeL srcStep, Ipp16u* pDst, IppSizeL dstStep,
                IppiSizeL roiSize, IppiBorderType borderType, Ipp16u borderValue[1], const IppiMorphAdvStateL* pMorthSpec, Ipp8u* pBuffer ))
IPPAPI( IppStatus, ippiMorphOpen_16s_C1R_L,(
                const Ipp16s* pSrc, IppSizeL srcStep, Ipp16s* pDst, IppSizeL dstStep,
                IppiSizeL roiSize, IppiBorderType borderType, Ipp16s borderValue[1], const IppiMorphAdvStateL* pMorthSpec, Ipp8u* pBuffer ))
IPPAPI( IppStatus, ippiMorphOpen_1u_C1R_L,(
                const Ipp8u* pSrc, IppSizeL srcStep, int srcBitOffset, Ipp8u* pDst, IppSizeL dstStep, int dstBitOffset,
                IppiSizeL roiSize, IppiBorderType borderType, Ipp8u borderValue[1], const IppiMorphAdvStateL* pMorthSpec, Ipp8u* pBuffer ))
IPPAPI( IppStatus, ippiMorphOpen_8u_C1R_L,(
                const Ipp8u* pSrc, IppSizeL srcStep, Ipp8u* pDst, IppSizeL dstStep,
                IppiSizeL roiSize, IppiBorderType borderType, Ipp8u borderValue[1], const IppiMorphAdvStateL* pMorthSpec, Ipp8u* pBuffer ))
IPPAPI( IppStatus, ippiMorphOpen_8u_C3R_L,(
                const Ipp8u* pSrc, IppSizeL srcStep, Ipp8u* pDst, IppSizeL dstStep,
                IppiSizeL roiSize, IppiBorderType borderType, Ipp8u borderValue[3], const IppiMorphAdvStateL* pMorthSpec, Ipp8u* pBuffer ))
IPPAPI( IppStatus, ippiMorphOpen_8u_C4R_L,(
                const Ipp8u* pSrc, IppSizeL srcStep, Ipp8u* pDst, IppSizeL dstStep,
                IppiSizeL roiSize, IppiBorderType borderType, Ipp8u borderValue[4], const IppiMorphAdvStateL* pMorthSpec, Ipp8u* pBuffer ))

IPPAPI( IppStatus, ippiMorphClose_8u_C1R_L,(
                const Ipp8u* pSrc, IppSizeL srcStep, Ipp8u* pDst, IppSizeL dstStep,
                IppiSizeL roiSize, IppiBorderType borderType, Ipp8u borderValue[1], const IppiMorphAdvStateL* pMorthSpec, Ipp8u* pBuffer ))
IPPAPI( IppStatus, ippiMorphClose_8u_C3R_L,(
                const Ipp8u* pSrc, IppSizeL srcStep, Ipp8u* pDst, IppSizeL dstStep,
                IppiSizeL roiSize, IppiBorderType borderType, Ipp8u borderValue[3], const IppiMorphAdvStateL* pMorthSpec, Ipp8u* pBuffer ))
IPPAPI( IppStatus, ippiMorphClose_8u_C4R_L,(
                const Ipp8u* pSrc, IppSizeL srcStep, Ipp8u* pDst, IppSizeL dstStep,
                IppiSizeL roiSize, IppiBorderType borderType, Ipp8u borderValue[4], const IppiMorphAdvStateL* pMorthSpec, Ipp8u* pBuffer ))

IPPAPI( IppStatus, ippiMorphClose_16u_C1R_L,(
                const Ipp16u* pSrc, IppSizeL srcStep, Ipp16u* pDst, IppSizeL dstStep,
                IppiSizeL roiSize, IppiBorderType borderType, Ipp16u borderValue[1], const IppiMorphAdvStateL* pMorthSpec, Ipp8u* pBuffer ))
IPPAPI( IppStatus, ippiMorphClose_16s_C1R_L,(
                const Ipp16s* pSrc, IppSizeL srcStep, Ipp16s* pDst, IppSizeL dstStep,
                IppiSizeL roiSize, IppiBorderType borderType, Ipp16s borderValue[1], const IppiMorphAdvStateL* pMorthSpec, Ipp8u* pBuffer ))
IPPAPI( IppStatus, ippiMorphClose_1u_C1R_L,(
                const Ipp8u* pSrc, IppSizeL srcStep, int srcBitOffset, Ipp8u* pDst, IppSizeL dstStep, int dstBitOffset,
                IppiSizeL roiSize, IppiBorderType borderType, Ipp8u borderValue[1], const IppiMorphAdvStateL* pMorthSpec, Ipp8u* pBuffer ))
IPPAPI( IppStatus, ippiMorphOpen_32f_C1R_L,(
                const Ipp32f* pSrc, IppSizeL srcStep, Ipp32f* pDst, IppSizeL dstStep,
                IppiSizeL roiSize, IppiBorderType borderType, Ipp32f borderValue[1], const IppiMorphAdvStateL* pMorthSpec, Ipp8u* pBuffer ))
IPPAPI( IppStatus, ippiMorphOpen_32f_C3R_L,(
                const Ipp32f* pSrc, IppSizeL srcStep, Ipp32f* pDst, IppSizeL dstStep,
                IppiSizeL roiSize, IppiBorderType borderType, Ipp32f borderValue[3], const IppiMorphAdvStateL* pMorthSpec, Ipp8u* pBuffer ))
IPPAPI( IppStatus, ippiMorphOpen_32f_C4R_L,(
                const Ipp32f* pSrc, IppSizeL srcStep, Ipp32f* pDst, IppSizeL dstStep,
                IppiSizeL roiSize, IppiBorderType borderType, Ipp32f borderValue[4], const IppiMorphAdvStateL* pMorthSpec, Ipp8u* pBuffer ))
IPPAPI( IppStatus, ippiMorphClose_32f_C1R_L,(
                const Ipp32f* pSrc, IppSizeL srcStep, Ipp32f* pDst, IppSizeL dstStep,
                IppiSizeL roiSize, IppiBorderType borderType, Ipp32f borderValue[1], const IppiMorphAdvStateL* pMorthSpec, Ipp8u* pBuffer ))
IPPAPI( IppStatus, ippiMorphClose_32f_C3R_L,(
                const Ipp32f* pSrc, IppSizeL srcStep, Ipp32f* pDst, IppSizeL dstStep,
                IppiSizeL roiSize, IppiBorderType borderType, Ipp32f borderValue[3], const IppiMorphAdvStateL* pMorthSpec, Ipp8u* pBuffer ))
IPPAPI( IppStatus, ippiMorphClose_32f_C4R_L,(
                const Ipp32f* pSrc, IppSizeL srcStep, Ipp32f* pDst, IppSizeL dstStep,
                IppiSizeL roiSize, IppiBorderType borderType, Ipp32f borderValue[4], const IppiMorphAdvStateL* pMorthSpec, Ipp8u* pBuffer ))

IPPAPI( IppStatus, ippiMorphTophat_16u_C1R_L,(
                const Ipp16u* pSrc, IppSizeL srcStep, Ipp16u* pDst, IppSizeL dstStep,
                IppiSizeL roiSize, IppiBorderType borderType, Ipp16u borderValue[1], const IppiMorphAdvStateL* pMorthSpec, Ipp8u* pBuffer ))
IPPAPI( IppStatus, ippiMorphTophat_16s_C1R_L,(
                const Ipp16s* pSrc, IppSizeL srcStep, Ipp16s* pDst, IppSizeL dstStep,
                IppiSizeL roiSize, IppiBorderType borderType, Ipp16s borderValue[1], const IppiMorphAdvStateL* pMorthSpec, Ipp8u* pBuffer ))
IPPAPI( IppStatus, ippiMorphTophat_1u_C1R_L,(
                const Ipp8u* pSrc, IppSizeL srcStep, int srcBitOffset, Ipp8u* pDst, IppSizeL dstStep, int dstBitOffset,
                IppiSizeL roiSize, IppiBorderType borderType, Ipp8u borderValue[1], const IppiMorphAdvStateL* pMorthSpec, Ipp8u* pBuffer ))
IPPAPI( IppStatus, ippiMorphTophat_8u_C1R_L,(
                const Ipp8u* pSrc, IppSizeL srcStep, Ipp8u* pDst, IppSizeL dstStep,
                IppiSizeL roiSize, IppiBorderType borderType, Ipp8u borderValue[1], const IppiMorphAdvStateL* pMorthSpec, Ipp8u* pBuffer ))
IPPAPI( IppStatus, ippiMorphTophat_8u_C3R_L,(
                const Ipp8u* pSrc, IppSizeL srcStep, Ipp8u* pDst, IppSizeL dstStep,
                IppiSizeL roiSize, IppiBorderType borderType, Ipp8u borderValue[3], const IppiMorphAdvStateL* pMorthSpec, Ipp8u* pBuffer ))
IPPAPI( IppStatus, ippiMorphTophat_8u_C4R_L,(
                const Ipp8u* pSrc, IppSizeL srcStep, Ipp8u* pDst, IppSizeL dstStep,
                IppiSizeL roiSize, IppiBorderType borderType, Ipp8u borderValue[4], const IppiMorphAdvStateL* pMorthSpec, Ipp8u* pBuffer ))
IPPAPI( IppStatus, ippiMorphTophat_32f_C1R_L,(
                const Ipp32f* pSrc, IppSizeL srcStep, Ipp32f* pDst, IppSizeL dstStep,
                IppiSizeL roiSize, IppiBorderType borderType, Ipp32f borderValue[1], const IppiMorphAdvStateL* pMorthSpec, Ipp8u* pBuffer ))
IPPAPI( IppStatus, ippiMorphTophat_32f_C3R_L,(
                const Ipp32f* pSrc, IppSizeL srcStep, Ipp32f* pDst, IppSizeL dstStep,
                IppiSizeL roiSize, IppiBorderType borderType, Ipp32f borderValue[3], const IppiMorphAdvStateL* pMorthSpec, Ipp8u* pBuffer ))
IPPAPI( IppStatus, ippiMorphTophat_32f_C4R_L,(
                const Ipp32f* pSrc, IppSizeL srcStep, Ipp32f* pDst, IppSizeL dstStep,
                IppiSizeL roiSize, IppiBorderType borderType, Ipp32f borderValue[4], const IppiMorphAdvStateL* pMorthSpec, Ipp8u* pBuffer ))

IPPAPI( IppStatus, ippiMorphBlackhat_16u_C1R_L,(
                const Ipp16u* pSrc, IppSizeL srcStep, Ipp16u* pDst, IppSizeL dstStep,
                IppiSizeL roiSize, IppiBorderType borderType, Ipp16u borderValue[1], const IppiMorphAdvStateL* pMorthSpec, Ipp8u* pBuffer ))
IPPAPI( IppStatus, ippiMorphBlackhat_16s_C1R_L,(
                const Ipp16s* pSrc, IppSizeL srcStep, Ipp16s* pDst, IppSizeL dstStep,
                IppiSizeL roiSize, IppiBorderType borderType, Ipp16s borderValue[1], const IppiMorphAdvStateL* pMorthSpec, Ipp8u* pBuffer ))
IPPAPI( IppStatus, ippiMorphBlackhat_1u_C1R_L,(
                const Ipp8u* pSrc, IppSizeL srcStep, int srcBitOffset, Ipp8u* pDst, IppSizeL dstStep, int dstBitOffset,
                IppiSizeL roiSize, IppiBorderType borderType, Ipp8u borderValue[1], const IppiMorphAdvStateL* pMorthSpec, Ipp8u* pBuffer ))
IPPAPI( IppStatus, ippiMorphBlackhat_8u_C1R_L,(
                const Ipp8u* pSrc, IppSizeL srcStep, Ipp8u* pDst, IppSizeL dstStep,
                IppiSizeL roiSize, IppiBorderType borderType, Ipp8u borderValue[1], const IppiMorphAdvStateL* pMorthSpec, Ipp8u* pBuffer ))
IPPAPI( IppStatus, ippiMorphBlackhat_8u_C3R_L,(
                const Ipp8u* pSrc, IppSizeL srcStep, Ipp8u* pDst, IppSizeL dstStep,
                IppiSizeL roiSize, IppiBorderType borderType, Ipp8u borderValue[3], const IppiMorphAdvStateL* pMorthSpec, Ipp8u* pBuffer ))
IPPAPI( IppStatus, ippiMorphBlackhat_8u_C4R_L,(
                const Ipp8u* pSrc, IppSizeL srcStep, Ipp8u* pDst, IppSizeL dstStep,
                IppiSizeL roiSize, IppiBorderType borderType, Ipp8u borderValue[4], const IppiMorphAdvStateL* pMorthSpec, Ipp8u* pBuffer ))
IPPAPI( IppStatus, ippiMorphBlackhat_32f_C1R_L,(
                const Ipp32f* pSrc, IppSizeL srcStep, Ipp32f* pDst, IppSizeL dstStep,
                IppiSizeL roiSize, IppiBorderType borderType, Ipp32f borderValue[1], const IppiMorphAdvStateL* pMorthSpec, Ipp8u* pBuffer ))
IPPAPI( IppStatus, ippiMorphBlackhat_32f_C3R_L,(
                const Ipp32f* pSrc, IppSizeL srcStep, Ipp32f* pDst, IppSizeL dstStep,
                IppiSizeL roiSize, IppiBorderType borderType, Ipp32f borderValue[3], const IppiMorphAdvStateL* pMorthSpec, Ipp8u* pBuffer ))
IPPAPI( IppStatus, ippiMorphBlackhat_32f_C4R_L,(
                const Ipp32f* pSrc, IppSizeL srcStep, Ipp32f* pDst, IppSizeL dstStep,
                IppiSizeL roiSize, IppiBorderType borderType, Ipp32f borderValue[4], const IppiMorphAdvStateL* pMorthSpec, Ipp8u* pBuffer ))

IPPAPI( IppStatus, ippiMorphGradient_16u_C1R_L,(
                const Ipp16u* pSrc, IppSizeL srcStep, Ipp16u* pDst, IppSizeL dstStep,
                IppiSizeL roiSize, IppiBorderType borderType, Ipp16u borderValue[1], const IppiMorphAdvStateL* pMorthSpec, Ipp8u* pBuffer ))
IPPAPI( IppStatus, ippiMorphGradient_16s_C1R_L,(
                const Ipp16s* pSrc, IppSizeL srcStep, Ipp16s* pDst, IppSizeL dstStep,
                IppiSizeL roiSize, IppiBorderType borderType, Ipp16s borderValue[1], const IppiMorphAdvStateL* pMorthSpec, Ipp8u* pBuffer ))
IPPAPI( IppStatus, ippiMorphGradient_1u_C1R_L,(
                const Ipp8u* pSrc, IppSizeL srcStep, int srcBitOffset, Ipp8u* pDst, IppSizeL dstStep, int dstBitOffset,
                IppiSizeL roiSize, IppiBorderType borderType, Ipp8u borderValue[1], const IppiMorphAdvStateL* pMorthSpec, Ipp8u* pBuffer ))
IPPAPI( IppStatus, ippiMorphGradient_8u_C1R_L,(
                const Ipp8u* pSrc, IppSizeL srcStep, Ipp8u* pDst, IppSizeL dstStep,
                IppiSizeL roiSize, IppiBorderType borderType, Ipp8u borderValue[1], const IppiMorphAdvStateL* pMorthSpec, Ipp8u* pBuffer ))
IPPAPI( IppStatus, ippiMorphGradient_8u_C3R_L,(
                const Ipp8u* pSrc, IppSizeL srcStep, Ipp8u* pDst, IppSizeL dstStep,
                IppiSizeL roiSize, IppiBorderType borderType, Ipp8u borderValue[3], const IppiMorphAdvStateL* pMorthSpec, Ipp8u* pBuffer ))
IPPAPI( IppStatus, ippiMorphGradient_8u_C4R_L,(
                const Ipp8u* pSrc, IppSizeL srcStep, Ipp8u* pDst, IppSizeL dstStep,
                IppiSizeL roiSize, IppiBorderType borderType, Ipp8u borderValue[4], const IppiMorphAdvStateL* pMorthSpec, Ipp8u* pBuffer ))
IPPAPI( IppStatus, ippiMorphGradient_32f_C1R_L,(
                const Ipp32f* pSrc, IppSizeL srcStep, Ipp32f* pDst, IppSizeL dstStep,
                IppiSizeL roiSize, IppiBorderType borderType, Ipp32f borderValue[1], const IppiMorphAdvStateL* pMorthSpec, Ipp8u* pBuffer ))
IPPAPI( IppStatus, ippiMorphGradient_32f_C3R_L,(
                const Ipp32f* pSrc, IppSizeL srcStep, Ipp32f* pDst, IppSizeL dstStep,
                IppiSizeL roiSize, IppiBorderType borderType, Ipp32f borderValue[3], const IppiMorphAdvStateL* pMorthSpec, Ipp8u* pBuffer ))
IPPAPI( IppStatus, ippiMorphGradient_32f_C4R_L,(
                const Ipp32f* pSrc, IppSizeL srcStep, Ipp32f* pDst, IppSizeL dstStep,
                IppiSizeL roiSize, IppiBorderType borderType, Ipp32f borderValue[4], const IppiMorphAdvStateL* pMorthSpec, Ipp8u* pBuffer ))

/* ///////////////////////////////////////////////////////////////////////////////////////
//  Name:   ippiMorphSetMode_L            
//
//  Purpose:    Set mode for symmetrical operation in advanced morphology
//              IPP_MORPH_DEFAULT - default behavior
//              IPP_MORPH_MASK_NO_FLIP - don't flip mask(the same mask for (erode(dilate)->dilate(erode)) operations)
//
//
//  Return:
//    ippStsNoErr               Ok.
//    ippStsNullPtrErr          One of the pointers is NULL.
//    ippStsNotSupportedModeErr Incorrect mode
//
//  Parameters:
//    mode                     Mode. One of IPP_MORPH_DEFAULT(flip), IPP_MORPH_MASK_NO_FLIP(no flip)
//    pMorphSpec               Pointer to the morphology specification structure.
*/

IPPAPI(IppStatus, ippiMorphSetMode_L, (int mode, IppiMorphAdvStateL* pMorphSpec))

/*F///////////////////////////////////////////////////////////////////////////////////////
//  Name:    ippiCannyGetSize_L
//
//  Purpose: Calculates size of temporary buffer, required to run Canny function.
//
//  Return:
//    ippStsNoErr              Ok
//    ippStsNullPtrErr         Pointer bufferSize is NULL
//    ippStsSizeErr            roiSize has a field with zero or negative value
//
//  Parameters:
//    roi                  Size of image ROI in pixel
//    bufferSize               Pointer to the variable that returns the size of the temporary buffer
//F*/

IPPAPI(IppStatus, ippiCannyGetSize_L, ( IppiSizeL roi, IppSizeL* bufferSize ))
/*F///////////////////////////////////////////////////////////////////////////////////////
//  Name:    ippiCanny_16s8u_C1R_L
//
//  Purpose: Creates binary image of source's image edges,
//                using derivatives of the first order.
//
//  Return:
//    ippStsNoErr              Ok
//    ippStsNullPtrErr         One of pointers is NULL
//    ippStsSizeErr            The width or height of images is less or equal zero
//    ippStsStepErr            The steps in images are too small
//    ippStsNotEvenStepErr     Step is not multiple of element.
//    ippStsBadArgErr          Bad thresholds
//
//  Parameters:
//    pDX                   Pointers to the source image ( first derivatives  with respect to X )
//    dxStep                Step in bytes through the source image pSrcDx
//    pDY                   Pointers to the source image ( first derivatives  with respect to Y )
//    dyStep                Step in bytes through the source image pSrcDy
//    pDst                  Pointers to the destination image
//    dstStep               Step in bytes through the destination image
//
//    roiL                  Size of the source images ROI in pixels
//    lowThreshold          Low threshold for edges detection
//    highThreshold         Upper threshold for edges detection
//    norm                  Norm type, {ippNormL1, ippNormL2}
//    pBuffer               Pointer to the pre-allocated temporary buffer, which size can be
//                          calculated using ippiCannyGetSize function
//F*/

IPPAPI(IppStatus, ippiCanny_16s8u_C1R_L, ( Ipp16s* pSrcDx, IppSizeL srcDxStep, Ipp16s* pSrcDy, IppSizeL srcDyStep, Ipp8u* pDstEdges, IppSizeL dstEdgeStep, IppiSizeL roiSize, Ipp32f lowThreshold, Ipp32f highThreshold, IppNormType norm, Ipp8u* pBuffer ))
IPPAPI(IppStatus, ippiCanny_32f8u_C1R_L, ( Ipp32f* pSrcDx, IppSizeL srcDxStep, Ipp32f* pSrcDy, IppSizeL srcDyStep, Ipp8u* pDstEdges, IppSizeL dstEdgeStep, IppiSizeL roiSize, Ipp32f lowThreshold, Ipp32f highThreshold, IppNormType norm, Ipp8u* pBuffer ))


#ifdef __cplusplus
}
#endif

#endif /* __IPPICV_H__ */
