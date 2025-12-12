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

#include "iw/iw_image_op.h"
#include "iw_owni.h"

IW_DECL(IppStatus) llwiSetChannel(double value, void *pDst, int dstStep,
    IppiSize size, IppDataType dataType, int channels, int channelNum);

/**/////////////////////////////////////////////////////////////////////////////
//                   iwiSetChannel
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(IppStatus) iwiSetChannel(double value, IwiImage *pDstImage, int channelNum, const IwiSetChannelParams *pAuxParams, const IwiTile *pTile)
{
    IppStatus status;

    (void)pAuxParams;

    status = owniCheckImageWrite(pDstImage);
    if(status)
        return status;

    if(pDstImage->m_channels == 1)
        return iwiSet(&value, 1, pDstImage, NULL, NULL, pTile);

    if(channelNum >= pDstImage->m_channels || channelNum < 0)
        return ippStsBadArgErr;

    {
        void*     pDst  = pDstImage->m_ptr;
        IwiSize   size  = pDstImage->m_size;

        if(pTile && pTile->m_initialized != ownTileInitNone)
        {
            if(pTile->m_initialized == ownTileInitSimple)
            {
                IwiRoi dstRoi = pTile->m_dstRoi;

                if(!owniTile_BoundToSize(&dstRoi, &size))
                    return ippStsNoOperation;

                pDst = iwiImage_GetPtr(pDstImage, dstRoi.y, dstRoi.x, 0);
            }
            else if(pTile->m_initialized == ownTileInitPipe)
            {
                IwiRoi dstLim; iwiTilePipeline_GetBoundedDstRoi(pTile, &dstLim);

                pDst = iwiImage_GetPtr(pDstImage, dstLim.y, dstLim.x, 0);

                size.width  = dstLim.width;
                size.height = dstLim.height;
            }
            else
                return ippStsContextMatchErr;
        }

        // Long compatibility check
        {
            IppiSize _size;

            status = ownLongCompatCheckValue(pDstImage->m_step, NULL);
            if(status < 0)
                return status;

            status = owniLongCompatCheckSize(size, &_size);
            if(status < 0)
                return status;

            return llwiSetChannel(value, pDst, (int)pDstImage->m_step, _size, pDstImage->m_dataType, pDstImage->m_channels, channelNum);
        }
    }
}


/**/////////////////////////////////////////////////////////////////////////////
//                   Low-Level Wrappers
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(IppStatus) llwiSetChannel(double value, void *pDst, int dstStep,
    IppiSize size, IppDataType dataType, int channels, int channelNum)
{
    switch(dataType)
    {
    case ipp8u:
        switch(channels)
        {
        case 3:  return ippiSet_8u_C3CR(ownCast_64f8u(value), ((Ipp8u*)pDst)+channelNum, dstStep, size);
        case 4:  return ippiSet_8u_C4CR(ownCast_64f8u(value), ((Ipp8u*)pDst)+channelNum, dstStep, size);
        default: return ippStsNumChannelsErr;
        }
    case ipp8s:
        switch(channels)
        {
        case 3:  return ippiSet_8u_C3CR(ownCast_64f8s(value), ((Ipp8u*)pDst)+channelNum, dstStep, size);
        case 4:  return ippiSet_8u_C4CR(ownCast_64f8s(value), ((Ipp8u*)pDst)+channelNum, dstStep, size);
        default: return ippStsNumChannelsErr;
        }
    case ipp16u:
        switch(channels)
        {
        case 3:  return ippiSet_16u_C3CR(ownCast_64f16u(value), ((Ipp16u*)pDst)+channelNum, dstStep, size);
        case 4:  return ippiSet_16u_C4CR(ownCast_64f16u(value), ((Ipp16u*)pDst)+channelNum, dstStep, size);
        default: return ippStsNumChannelsErr;
        }
    case ipp16s:
        switch(channels)
        {
        case 3:  return ippiSet_16u_C3CR(ownCast_64f16s(value), ((Ipp16u*)pDst)+channelNum, dstStep, size);
        case 4:  return ippiSet_16u_C4CR(ownCast_64f16s(value), ((Ipp16u*)pDst)+channelNum, dstStep, size);
        default: return ippStsNumChannelsErr;
        }
    case ipp32u:
        switch(channels)
        {
        case 3:  return ippiSet_32s_C3CR(ownCast_64f32u(value), ((Ipp32s*)pDst)+channelNum, dstStep, size);
        case 4:  return ippiSet_32s_C4CR(ownCast_64f32u(value), ((Ipp32s*)pDst)+channelNum, dstStep, size);
        default: return ippStsNumChannelsErr;
        }
    case ipp32s:
        switch(channels)
        {
        case 3:  return ippiSet_32s_C3CR(ownCast_64f32s(value), ((Ipp32s*)pDst)+channelNum, dstStep, size);
        case 4:  return ippiSet_32s_C4CR(ownCast_64f32s(value), ((Ipp32s*)pDst)+channelNum, dstStep, size);
        default: return ippStsNumChannelsErr;
        }
    case ipp32f:
        switch(channels)
        {
        case 3:  return ippiSet_32f_C3CR(ownCast_64f32f(value), ((Ipp32f*)pDst)+channelNum, dstStep, size);
        case 4:  return ippiSet_32f_C4CR(ownCast_64f32f(value), ((Ipp32f*)pDst)+channelNum, dstStep, size);
        default: return ippStsNumChannelsErr;
        }
    default: return ippStsDataTypeErr;
    }
}
