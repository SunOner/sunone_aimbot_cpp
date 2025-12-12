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

#include "iw/iw_image_transform.h"
#include "iw_owni.h"

/* /////////////////////////////////////////////////////////////////////////////
//                   iwiRotate
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(IppStatus) iwiRotate_GetDstSize(IwiSize srcSize, double angle, IwiSize *pDstSize)
{
    IppStatus  status;
    IppiRect   rect  = {0};

    double bound[2][2]  = {0};
    double coeffs[2][3] = {0};

    if(!pDstSize)
        return ippStsNullPtrErr;

    status = ippiGetRotateTransform(angle, 0, 0, coeffs);
    if(status < 0)
        return status;

    rect.width  = (int)srcSize.width;
    rect.height = (int)srcSize.height;
    status = ippiGetAffineBound(rect, bound, (const double (*)[3])coeffs);
    if(status < 0)
        return status;

    pDstSize->width  = (IwSize)(bound[1][0] - bound[0][0] + 1.5);
    pDstSize->height = (IwSize)(bound[1][1] - bound[0][1] + 1.5);

    return ippStsNoErr;
}

IW_DECL(IppStatus) iwiRotate(const IwiImage *pSrcImage, IwiImage *pDstImage, double angle, IppiInterpolationType interpolation, const IwiRotateParams *pAuxParams, IwiBorderType border, const Ipp64f *pBorderVal, const IwiTile *pTile)
{
    IppStatus status;

    (void)pAuxParams;

    status = owniCheckImageRead(pSrcImage);
    if(status)
        return status;
    status = owniCheckImageWrite(pDstImage);
    if(status)
        return status;

    if(pSrcImage->m_ptrConst == pDstImage->m_ptrConst)
        return ippStsNoOperation;

    if(pSrcImage->m_typeSize != pDstImage->m_typeSize ||
        pSrcImage->m_channels != pDstImage->m_channels)
        return ippStsBadArgErr;

    if(!pSrcImage->m_size.width || !pSrcImage->m_size.height ||
        !pDstImage->m_size.width || !pDstImage->m_size.height)
        return ippStsNoOperation;

    for(;;)
    {
        IppiRect rect         = {0};
        double   bound[2][2]  = {0};
        double   coeffs[2][3] = {0};

        status = ippiGetRotateTransform(angle, 0, 0, coeffs);
        if(status < 0)
            break;

        rect.width  = (int)pSrcImage->m_size.width;
        rect.height = (int)pSrcImage->m_size.height;
        status = ippiGetAffineBound(rect, bound, (const double (*)[3])coeffs);
        if(status < 0)
            return status;

        coeffs[0][2] -= bound[0][0];
        coeffs[1][2] -= bound[0][1];

        status = iwiWarpAffine(pSrcImage, pDstImage, (const double (*)[3])coeffs, iwTransForward, interpolation, NULL, border, pBorderVal, pTile);
        break;
    }

    return status;
}
