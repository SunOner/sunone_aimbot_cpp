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

IW_DECL(IppStatus) llwiCopy(const void *pSrc, IwSize srcStep, void *pDst, IwSize dstStep,
                              IwiSize size, int typeSize, int channels);

IW_DECL(IppStatus) llwiCopyMakeBorder(const void *pSrc, IwSize srcStep, void *pDst, IwSize dstStep,
                                  IwiSize size, IppDataType dataType, int channels, IwiBorderSize borderSize, IwiBorderType border, const Ipp64f *pBorderVal);

/**/////////////////////////////////////////////////////////////////////////////
//                   iwiCreateBorder
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(IppStatus) iwiCreateBorder(const IwiImage *pSrcImage, IwiImage *pDstImage, IwiBorderSize borderSize, IwiBorderType border, const Ipp64f *pBorderVal, const IwiCreateBorderParams *pAuxParams, const IwiTile *pTile)
{
    IppStatus status;

    (void)pAuxParams;

    status = owniCheckImageRead(pSrcImage);
    if(status)
        return status;
    status = owniCheckImageWrite(pDstImage);
    if(status)
        return status;

    if(pSrcImage->m_typeSize != pDstImage->m_typeSize ||
        pSrcImage->m_channels != pDstImage->m_channels)
        return ippStsBadArgErr;

    if((border&ippBorderInMem) == ippBorderInMem)
        return ippStsNoOperation;

    if(borderSize.top > pDstImage->m_inMemSize.top ||
        borderSize.left > pDstImage->m_inMemSize.left)
        return ippStsSizeErr;

    {
        const void *pSrc  = pSrcImage->m_ptrConst;
        void       *pDst  = pDstImage->m_ptr;
        IwiSize     size  = owniGetMinSize(&pSrcImage->m_size, &pDstImage->m_size);

        if(pDstImage->m_size.width + pDstImage->m_inMemSize.right < size.width + borderSize.right ||
            pDstImage->m_size.height + pDstImage->m_inMemSize.bottom < size.height + borderSize.bottom)
            return ippStsSizeErr;

        if(pTile && pTile->m_initialized != ownTileInitNone)
        {
            if(OWN_GET_PURE_BORDER(border) == ippBorderWrap)
                return ippStsNotSupportedModeErr;

            if(pTile->m_initialized == ownTileInitSimple)
            {
                IwiRoi         dstRoi         = pTile->m_dstRoi;
                IwiBorderSize  tileBorderSize = borderSize;

                if(!owniTile_BoundToSize(&dstRoi, &size))
                    return ippStsNoOperation;

                if(border == ippBorderMirror)
                {
                    tileBorderSize.left   = borderSize.left+1;
                    tileBorderSize.top    = borderSize.top+1;
                    tileBorderSize.right  = borderSize.right+1;
                    tileBorderSize.bottom = borderSize.bottom+1;
                }
                owniTile_CorrectBordersOverlap(&dstRoi, &size, &border, &tileBorderSize, &tileBorderSize, &pSrcImage->m_size);
                owniTile_GetTileBorder(&border, &dstRoi, &borderSize, &pSrcImage->m_size);

                pSrc  = iwiImage_GetPtrConst(pSrcImage, dstRoi.y, dstRoi.x, 0);
                pDst  = iwiImage_GetPtr(pDstImage, dstRoi.y, dstRoi.x, 0);
            }
            else if(pTile->m_initialized == ownTileInitPipe)
            {
                IwiRoi srcLim;
                IwiRoi dstLim;
                iwiTilePipeline_GetBoundedSrcRoi(pTile, &srcLim);
                iwiTilePipeline_GetBoundedDstRoi(pTile, &dstLim);

                pSrc   = iwiImage_GetPtrConst(pSrcImage, srcLim.y, srcLim.x, 0);
                pDst   = iwiImage_GetPtr(pDstImage, dstLim.y, dstLim.x, 0);
                iwiTilePipeline_GetTileBorder(pTile, &border);

                size = owniGetMinSizeFromRect(&srcLim, &dstLim);
            }
            else
                return ippStsContextMatchErr;
        }

        return llwiCopyMakeBorder(pSrc, pSrcImage->m_step, pDst, pDstImage->m_step, size, pSrcImage->m_dataType, pSrcImage->m_channels, borderSize, border, pBorderVal);
    }
}


/**/////////////////////////////////////////////////////////////////////////////
//                   Low-Level Wrappers
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(IppStatus) llwiCopyMakeBorder(const void *pSrc, IwSize srcStep, void *pDst, IwSize dstStep,
    IwiSize size, IppDataType dataType, int channels, IwiBorderSize borderSize, IwiBorderType border, const Ipp64f *pBorderVal)
{
    IppStatus status;
    Ipp64f    borderVal[4] = {0};
    int       depth = iwTypeToSize(dataType);

    if(border & ippBorderInMemLeft)
    {
        pSrc = owniShiftPtr(pSrc, srcStep, depth, channels, -((IwSize)borderSize.left), 0);
        pDst = owniShiftPtr(pDst, dstStep, depth, channels, -((IwSize)borderSize.left), 0);
        size.width += borderSize.left;
        borderSize.left = 0;
    }
    if(border & ippBorderInMemTop)
    {
        pSrc = owniShiftPtr(pSrc, srcStep, depth, channels, 0, -((IwSize)borderSize.top));
        pDst = owniShiftPtr(pDst, dstStep, depth, channels, 0, -((IwSize)borderSize.top));
        size.height += borderSize.top;
        borderSize.top = 0;
    }
    if(border & ippBorderInMemRight)
    {
        size.width += borderSize.right;
        borderSize.right = 0;
    }
    if(border & ippBorderInMemBottom)
    {
        size.height += borderSize.bottom;
        borderSize.bottom = 0;
    }
    if(!borderSize.left && !borderSize.right && !borderSize.top && !borderSize.bottom)
        return llwiCopy(pSrc, srcStep, pDst, dstStep, size, depth, channels);

    border = OWN_GET_PURE_BORDER(border);

    if(pSrc == pDst)
    {
        IwiSize dstSize = size;
        dstSize.width  += (borderSize.right + borderSize.left);
        dstSize.height += (borderSize.bottom + borderSize.top);

        if(border == ippBorderConst)
        {
#if IPP_VERSION_COMPLEX >= 20170002
            switch(dataType)
            {
            case ipp8u:
                switch(channels)
                {
                case 1:  status = ippiCopyConstBorder_8u_C1IR_L((Ipp8u*)pSrc, srcStep, size, dstSize, borderSize.top, borderSize.left, OWN_GET_BORDER_VAL(8u)); break;
                case 3:  status = ippiCopyConstBorder_8u_C3IR_L((Ipp8u*)pSrc, srcStep, size, dstSize, borderSize.top, borderSize.left, OWN_GET_BORDER_VALP(8u, 3)); break;
                case 4:  status = ippiCopyConstBorder_8u_C4IR_L((Ipp8u*)pSrc, srcStep, size, dstSize, borderSize.top, borderSize.left, OWN_GET_BORDER_VALP(8u, 4)); break;
                default: status = ippStsNumChannelsErr; break;
                }
                break;
            case ipp16u:
                switch(channels)
                {
                case 1:  status = ippiCopyConstBorder_16u_C1IR_L((Ipp16u*)pSrc, srcStep, size, dstSize, borderSize.top, borderSize.left, OWN_GET_BORDER_VAL(16u)); break;
                case 3:  status = ippiCopyConstBorder_16u_C3IR_L((Ipp16u*)pSrc, srcStep, size, dstSize, borderSize.top, borderSize.left, OWN_GET_BORDER_VALP(16u, 3)); break;
                case 4:  status = ippiCopyConstBorder_16u_C4IR_L((Ipp16u*)pSrc, srcStep, size, dstSize, borderSize.top, borderSize.left, OWN_GET_BORDER_VALP(16u, 4)); break;
                default: status = ippStsNumChannelsErr; break;
                }
                break;
            case ipp16s:
                switch(channels)
                {
                case 1:  status = ippiCopyConstBorder_16s_C1IR_L((Ipp16s*)pSrc, srcStep, size, dstSize, borderSize.top, borderSize.left, OWN_GET_BORDER_VAL(16s)); break;
                case 3:  status = ippiCopyConstBorder_16s_C3IR_L((Ipp16s*)pSrc, srcStep, size, dstSize, borderSize.top, borderSize.left, OWN_GET_BORDER_VALP(16s, 3)); break;
                case 4:  status = ippiCopyConstBorder_16s_C4IR_L((Ipp16s*)pSrc, srcStep, size, dstSize, borderSize.top, borderSize.left, OWN_GET_BORDER_VALP(16s, 4)); break;
                default: status = ippStsNumChannelsErr; break;
                }
                break;
            case ipp32s:
                switch(channels)
                {
                case 1:  status = ippiCopyConstBorder_32s_C1IR_L((Ipp32s*)pSrc, srcStep, size, dstSize, borderSize.top, borderSize.left, OWN_GET_BORDER_VAL(32s)); break;
                case 3:  status = ippiCopyConstBorder_32s_C3IR_L((Ipp32s*)pSrc, srcStep, size, dstSize, borderSize.top, borderSize.left, OWN_GET_BORDER_VALP(32s, 3)); break;
                case 4:  status = ippiCopyConstBorder_32s_C4IR_L((Ipp32s*)pSrc, srcStep, size, dstSize, borderSize.top, borderSize.left, OWN_GET_BORDER_VALP(32s, 4)); break;
                default: status = ippStsNumChannelsErr; break;
                }
                break;
            case ipp32f:
                switch(channels)
                {
                case 1:  status = ippiCopyConstBorder_32f_C1IR_L((Ipp32f*)pSrc, srcStep, size, dstSize, borderSize.top, borderSize.left, OWN_GET_BORDER_VAL(32f)); break;
                case 3:  status = ippiCopyConstBorder_32f_C3IR_L((Ipp32f*)pSrc, srcStep, size, dstSize, borderSize.top, borderSize.left, OWN_GET_BORDER_VALP(32f, 3)); break;
                case 4:  status = ippiCopyConstBorder_32f_C4IR_L((Ipp32f*)pSrc, srcStep, size, dstSize, borderSize.top, borderSize.left, OWN_GET_BORDER_VALP(32f, 4)); break;
                default: status = ippStsNumChannelsErr; break;
                }
                break;
            default: status = ippStsDataTypeErr; break;
            }
#else
            return ippStsInplaceModeNotSupportedErr;
#endif
        }
        else if(border == ippBorderRepl)
        {
            switch(dataType)
            {
            case ipp8u:
                switch(channels)
                {
                case 1:  status = ippiCopyReplicateBorder_8u_C1IR_L((Ipp8u*)pSrc, srcStep, size, dstSize, borderSize.top, borderSize.left); break;
                case 3:  status = ippiCopyReplicateBorder_8u_C3IR_L((Ipp8u*)pSrc, srcStep, size, dstSize, borderSize.top, borderSize.left); break;
                case 4:  status = ippiCopyReplicateBorder_8u_C4IR_L((Ipp8u*)pSrc, srcStep, size, dstSize, borderSize.top, borderSize.left); break;
                default: status = ippStsNumChannelsErr; break;
                }
                break;
            case ipp16u:
                switch(channels)
                {
                case 1:  status = ippiCopyReplicateBorder_16u_C1IR_L((Ipp16u*)pSrc, srcStep, size, dstSize, borderSize.top, borderSize.left); break;
                case 3:  status = ippiCopyReplicateBorder_16u_C3IR_L((Ipp16u*)pSrc, srcStep, size, dstSize, borderSize.top, borderSize.left); break;
                case 4:  status = ippiCopyReplicateBorder_16u_C4IR_L((Ipp16u*)pSrc, srcStep, size, dstSize, borderSize.top, borderSize.left); break;
                default: status = ippStsNumChannelsErr; break;
                }
                break;
            case ipp16s:
                switch(channels)
                {
                case 1:  status = ippiCopyReplicateBorder_16s_C1IR_L((Ipp16s*)pSrc, srcStep, size, dstSize, borderSize.top, borderSize.left); break;
                case 3:  status = ippiCopyReplicateBorder_16s_C3IR_L((Ipp16s*)pSrc, srcStep, size, dstSize, borderSize.top, borderSize.left); break;
                case 4:  status = ippiCopyReplicateBorder_16s_C4IR_L((Ipp16s*)pSrc, srcStep, size, dstSize, borderSize.top, borderSize.left); break;
                default: status = ippStsNumChannelsErr; break;
                }
                break;
            case ipp32s:
                switch(channels)
                {
                case 1:  status = ippiCopyReplicateBorder_32s_C1IR_L((Ipp32s*)pSrc, srcStep, size, dstSize, borderSize.top, borderSize.left); break;
                case 3:  status = ippiCopyReplicateBorder_32s_C3IR_L((Ipp32s*)pSrc, srcStep, size, dstSize, borderSize.top, borderSize.left); break;
                case 4:  status = ippiCopyReplicateBorder_32s_C4IR_L((Ipp32s*)pSrc, srcStep, size, dstSize, borderSize.top, borderSize.left); break;
                default: status = ippStsNumChannelsErr; break;
                }
                break;
            case ipp32f:
                switch(channels)
                {
                case 1:  status = ippiCopyReplicateBorder_32f_C1IR_L((Ipp32f*)pSrc, srcStep, size, dstSize, borderSize.top, borderSize.left); break;
                case 3:  status = ippiCopyReplicateBorder_32f_C3IR_L((Ipp32f*)pSrc, srcStep, size, dstSize, borderSize.top, borderSize.left); break;
                case 4:  status = ippiCopyReplicateBorder_32f_C4IR_L((Ipp32f*)pSrc, srcStep, size, dstSize, borderSize.top, borderSize.left); break;
                default: status = ippStsNumChannelsErr; break;
                }
                break;
            default: status = ippStsDataTypeErr; break;
            }
        }
        else if(border == ippBorderMirror)
        {
            switch(dataType)
            {
            case ipp8u:
                switch(channels)
                {
                case 1:  status = ippiCopyMirrorBorder_8u_C1IR_L((Ipp8u*)pSrc, srcStep, size, dstSize, borderSize.top, borderSize.left); break;
                case 3:  status = ippiCopyMirrorBorder_8u_C3IR_L((Ipp8u*)pSrc, srcStep, size, dstSize, borderSize.top, borderSize.left); break;
                case 4:  status = ippiCopyMirrorBorder_8u_C4IR_L((Ipp8u*)pSrc, srcStep, size, dstSize, borderSize.top, borderSize.left); break;
                default: status = ippStsNumChannelsErr; break;
                }
                break;
            case ipp16u:
                switch(channels)
                {
                case 1:  status = ippiCopyMirrorBorder_16u_C1IR_L((Ipp16u*)pSrc, srcStep, size, dstSize, borderSize.top, borderSize.left); break;
                case 3:  status = ippiCopyMirrorBorder_16u_C3IR_L((Ipp16u*)pSrc, srcStep, size, dstSize, borderSize.top, borderSize.left); break;
                case 4:  status = ippiCopyMirrorBorder_16u_C4IR_L((Ipp16u*)pSrc, srcStep, size, dstSize, borderSize.top, borderSize.left); break;
                default: status = ippStsNumChannelsErr; break;
                }
                break;
            case ipp16s:
                switch(channels)
                {
                case 1:  status = ippiCopyMirrorBorder_16s_C1IR_L((Ipp16s*)pSrc, srcStep, size, dstSize, borderSize.top, borderSize.left); break;
                case 3:  status = ippiCopyMirrorBorder_16s_C3IR_L((Ipp16s*)pSrc, srcStep, size, dstSize, borderSize.top, borderSize.left); break;
                case 4:  status = ippiCopyMirrorBorder_16s_C4IR_L((Ipp16s*)pSrc, srcStep, size, dstSize, borderSize.top, borderSize.left); break;
                default: status = ippStsNumChannelsErr; break;
                }
                break;
            case ipp32s:
                switch(channels)
                {
                case 1:  status = ippiCopyMirrorBorder_32s_C1IR_L((Ipp32s*)pSrc, srcStep, size, dstSize, borderSize.top, borderSize.left); break;
                case 3:  status = ippiCopyMirrorBorder_32s_C3IR_L((Ipp32s*)pSrc, srcStep, size, dstSize, borderSize.top, borderSize.left); break;
                case 4:  status = ippiCopyMirrorBorder_32s_C4IR_L((Ipp32s*)pSrc, srcStep, size, dstSize, borderSize.top, borderSize.left); break;
                default: status = ippStsNumChannelsErr; break;
                }
                break;
            case ipp32f:
                switch(channels)
                {
                case 1:  status = ippiCopyMirrorBorder_32f_C1IR_L((Ipp32f*)pSrc, srcStep, size, dstSize, borderSize.top, borderSize.left); break;
                case 3:  status = ippiCopyMirrorBorder_32f_C3IR_L((Ipp32f*)pSrc, srcStep, size, dstSize, borderSize.top, borderSize.left); break;
                case 4:  status = ippiCopyMirrorBorder_32f_C4IR_L((Ipp32f*)pSrc, srcStep, size, dstSize, borderSize.top, borderSize.left); break;
                default: status = ippStsNumChannelsErr; break;
                }
                break;
            default: status = ippStsDataTypeErr; break;
            }
        }
        else if(border == ippBorderWrap)
        {
            switch(dataType)
            {
            case ipp32s:
                switch(channels)
                {
                case 1:  status = ippiCopyWrapBorder_32s_C1IR_L((Ipp32s*)pSrc, srcStep, size, dstSize, borderSize.top, borderSize.left); break;
                default: status = ippStsNumChannelsErr; break;
                }
                break;
            case ipp32f:
                switch(channels)
                {
                case 1:  status = ippiCopyWrapBorder_32f_C1IR_L((Ipp32f*)pSrc, srcStep, size, dstSize, borderSize.top, borderSize.left); break;
                default: status = ippStsNumChannelsErr; break;
                }
                break;
            default: status = ippStsDataTypeErr; break;
            }
        }
        else
            status = ippStsBorderErr;
    }
    else
    {
        IwiSize dstSize = size;
        pDst = owniShiftPtr(pDst, dstStep, depth, channels, -(IwSize)borderSize.left, -(IwSize)borderSize.top);
        dstSize.width  += (borderSize.right + borderSize.left);
        dstSize.height += (borderSize.bottom + borderSize.top);

        if(border == ippBorderConst)
        {
            switch(dataType)
            {
            case ipp8u:
                switch(channels)
                {
                case 1:  status = ippiCopyConstBorder_8u_C1R_L((Ipp8u*)pSrc, srcStep, size, (Ipp8u*)pDst, dstStep, dstSize, borderSize.top, borderSize.left, OWN_GET_BORDER_VAL(8u)); break;
                case 3:  status = ippiCopyConstBorder_8u_C3R_L((Ipp8u*)pSrc, srcStep, size, (Ipp8u*)pDst, dstStep, dstSize, borderSize.top, borderSize.left, OWN_GET_BORDER_VALP(8u, 3)); break;
                case 4:  status = ippiCopyConstBorder_8u_C4R_L((Ipp8u*)pSrc, srcStep, size, (Ipp8u*)pDst, dstStep, dstSize, borderSize.top, borderSize.left, OWN_GET_BORDER_VALP(8u, 4)); break;
                default: status = ippStsNumChannelsErr; break;
                }
                break;
            case ipp16u:
                switch(channels)
                {
                case 1:  status = ippiCopyConstBorder_16u_C1R_L((Ipp16u*)pSrc, srcStep, size, (Ipp16u*)pDst, dstStep, dstSize, borderSize.top, borderSize.left, OWN_GET_BORDER_VAL(16u)); break;
                case 3:  status = ippiCopyConstBorder_16u_C3R_L((Ipp16u*)pSrc, srcStep, size, (Ipp16u*)pDst, dstStep, dstSize, borderSize.top, borderSize.left, OWN_GET_BORDER_VALP(16u, 3)); break;
                case 4:  status = ippiCopyConstBorder_16u_C4R_L((Ipp16u*)pSrc, srcStep, size, (Ipp16u*)pDst, dstStep, dstSize, borderSize.top, borderSize.left, OWN_GET_BORDER_VALP(16u, 4)); break;
                default: status = ippStsNumChannelsErr; break;
                }
                break;
            case ipp16s:
                switch(channels)
                {
                case 1:  status = ippiCopyConstBorder_16s_C1R_L((Ipp16s*)pSrc, srcStep, size, (Ipp16s*)pDst, dstStep, dstSize, borderSize.top, borderSize.left, OWN_GET_BORDER_VAL(16s)); break;
                case 3:  status = ippiCopyConstBorder_16s_C3R_L((Ipp16s*)pSrc, srcStep, size, (Ipp16s*)pDst, dstStep, dstSize, borderSize.top, borderSize.left, OWN_GET_BORDER_VALP(16s, 3)); break;
                case 4:  status = ippiCopyConstBorder_16s_C4R_L((Ipp16s*)pSrc, srcStep, size, (Ipp16s*)pDst, dstStep, dstSize, borderSize.top, borderSize.left, OWN_GET_BORDER_VALP(16s, 4)); break;
                default: status = ippStsNumChannelsErr; break;
                }
                break;
            case ipp32s:
                switch(channels)
                {
                case 1:  status = ippiCopyConstBorder_32s_C1R_L((Ipp32s*)pSrc, srcStep, size, (Ipp32s*)pDst, dstStep, dstSize, borderSize.top, borderSize.left, OWN_GET_BORDER_VAL(32s)); break;
                case 3:  status = ippiCopyConstBorder_32s_C3R_L((Ipp32s*)pSrc, srcStep, size, (Ipp32s*)pDst, dstStep, dstSize, borderSize.top, borderSize.left, OWN_GET_BORDER_VALP(32s, 3)); break;
                case 4:  status = ippiCopyConstBorder_32s_C4R_L((Ipp32s*)pSrc, srcStep, size, (Ipp32s*)pDst, dstStep, dstSize, borderSize.top, borderSize.left, OWN_GET_BORDER_VALP(32s, 4)); break;
                default: status = ippStsNumChannelsErr; break;
                }
                break;
            case ipp32f:
                switch(channels)
                {
                case 1:  status = ippiCopyConstBorder_32f_C1R_L((Ipp32f*)pSrc, srcStep, size, (Ipp32f*)pDst, dstStep, dstSize, borderSize.top, borderSize.left, OWN_GET_BORDER_VAL(32f)); break;
                case 3:  status = ippiCopyConstBorder_32f_C3R_L((Ipp32f*)pSrc, srcStep, size, (Ipp32f*)pDst, dstStep, dstSize, borderSize.top, borderSize.left, OWN_GET_BORDER_VALP(32f, 3)); break;
                case 4:  status = ippiCopyConstBorder_32f_C4R_L((Ipp32f*)pSrc, srcStep, size, (Ipp32f*)pDst, dstStep, dstSize, borderSize.top, borderSize.left, OWN_GET_BORDER_VALP(32f, 4)); break;
                default: status = ippStsNumChannelsErr; break;
                }
                break;
            default: status = ippStsDataTypeErr; break;
            }
        }
        else if(border == ippBorderRepl)
        {
            switch(dataType)
            {
            case ipp8u:
                switch(channels)
                {
                case 1:  status = ippiCopyReplicateBorder_8u_C1R_L((Ipp8u*)pSrc, srcStep, size, (Ipp8u*)pDst, dstStep, dstSize, borderSize.top, borderSize.left); break;
                case 3:  status = ippiCopyReplicateBorder_8u_C3R_L((Ipp8u*)pSrc, srcStep, size, (Ipp8u*)pDst, dstStep, dstSize, borderSize.top, borderSize.left); break;
                case 4:  status = ippiCopyReplicateBorder_8u_C4R_L((Ipp8u*)pSrc, srcStep, size, (Ipp8u*)pDst, dstStep, dstSize, borderSize.top, borderSize.left); break;
                default: status = ippStsNumChannelsErr; break;
                }
                break;
            case ipp16u:
                switch(channels)
                {
                case 1:  status = ippiCopyReplicateBorder_16u_C1R_L((Ipp16u*)pSrc, srcStep, size, (Ipp16u*)pDst, dstStep, dstSize, borderSize.top, borderSize.left); break;
                case 3:  status = ippiCopyReplicateBorder_16u_C3R_L((Ipp16u*)pSrc, srcStep, size, (Ipp16u*)pDst, dstStep, dstSize, borderSize.top, borderSize.left); break;
                case 4:  status = ippiCopyReplicateBorder_16u_C4R_L((Ipp16u*)pSrc, srcStep, size, (Ipp16u*)pDst, dstStep, dstSize, borderSize.top, borderSize.left); break;
                default: status = ippStsNumChannelsErr; break;
                }
                break;
            case ipp16s:
                switch(channels)
                {
                case 1:  status = ippiCopyReplicateBorder_16s_C1R_L((Ipp16s*)pSrc, srcStep, size, (Ipp16s*)pDst, dstStep, dstSize, borderSize.top, borderSize.left); break;
                case 3:  status = ippiCopyReplicateBorder_16s_C3R_L((Ipp16s*)pSrc, srcStep, size, (Ipp16s*)pDst, dstStep, dstSize, borderSize.top, borderSize.left); break;
                case 4:  status = ippiCopyReplicateBorder_16s_C4R_L((Ipp16s*)pSrc, srcStep, size, (Ipp16s*)pDst, dstStep, dstSize, borderSize.top, borderSize.left); break;
                default: status = ippStsNumChannelsErr; break;
                }
                break;
            case ipp32s:
                switch(channels)
                {
                case 1:  status = ippiCopyReplicateBorder_32s_C1R_L((Ipp32s*)pSrc, srcStep, size, (Ipp32s*)pDst, dstStep, dstSize, borderSize.top, borderSize.left); break;
                case 3:  status = ippiCopyReplicateBorder_32s_C3R_L((Ipp32s*)pSrc, srcStep, size, (Ipp32s*)pDst, dstStep, dstSize, borderSize.top, borderSize.left); break;
                case 4:  status = ippiCopyReplicateBorder_32s_C4R_L((Ipp32s*)pSrc, srcStep, size, (Ipp32s*)pDst, dstStep, dstSize, borderSize.top, borderSize.left); break;
                default: status = ippStsNumChannelsErr; break;
                }
                break;
            case ipp32f:
                switch(channels)
                {
                case 1:  status = ippiCopyReplicateBorder_32f_C1R_L((Ipp32f*)pSrc, srcStep, size, (Ipp32f*)pDst, dstStep, dstSize, borderSize.top, borderSize.left); break;
                case 3:  status = ippiCopyReplicateBorder_32f_C3R_L((Ipp32f*)pSrc, srcStep, size, (Ipp32f*)pDst, dstStep, dstSize, borderSize.top, borderSize.left); break;
                case 4:  status = ippiCopyReplicateBorder_32f_C4R_L((Ipp32f*)pSrc, srcStep, size, (Ipp32f*)pDst, dstStep, dstSize, borderSize.top, borderSize.left); break;
                default: status = ippStsNumChannelsErr; break;
                }
                break;
            default: status = ippStsDataTypeErr; break;
            }
        }
        else if(border == ippBorderMirror)
        {
            switch(dataType)
            {
            case ipp8u:
                switch(channels)
                {
                case 1:  status = ippiCopyMirrorBorder_8u_C1R_L((Ipp8u*)pSrc, srcStep, size, (Ipp8u*)pDst, dstStep, dstSize, borderSize.top, borderSize.left); break;
                case 3:  status = ippiCopyMirrorBorder_8u_C3R_L((Ipp8u*)pSrc, srcStep, size, (Ipp8u*)pDst, dstStep, dstSize, borderSize.top, borderSize.left); break;
                case 4:  status = ippiCopyMirrorBorder_8u_C4R_L((Ipp8u*)pSrc, srcStep, size, (Ipp8u*)pDst, dstStep, dstSize, borderSize.top, borderSize.left); break;
                default: status = ippStsNumChannelsErr; break;
                }
                break;
            case ipp16u:
                switch(channels)
                {
                case 1:  status = ippiCopyMirrorBorder_16u_C1R_L((Ipp16u*)pSrc, srcStep, size, (Ipp16u*)pDst, dstStep, dstSize, borderSize.top, borderSize.left); break;
                case 3:  status = ippiCopyMirrorBorder_16u_C3R_L((Ipp16u*)pSrc, srcStep, size, (Ipp16u*)pDst, dstStep, dstSize, borderSize.top, borderSize.left); break;
                case 4:  status = ippiCopyMirrorBorder_16u_C4R_L((Ipp16u*)pSrc, srcStep, size, (Ipp16u*)pDst, dstStep, dstSize, borderSize.top, borderSize.left); break;
                default: status = ippStsNumChannelsErr; break;
                }
                break;
            case ipp16s:
                switch(channels)
                {
                case 1:  status = ippiCopyMirrorBorder_16s_C1R_L((Ipp16s*)pSrc, srcStep, size, (Ipp16s*)pDst, dstStep, dstSize, borderSize.top, borderSize.left); break;
                case 3:  status = ippiCopyMirrorBorder_16s_C3R_L((Ipp16s*)pSrc, srcStep, size, (Ipp16s*)pDst, dstStep, dstSize, borderSize.top, borderSize.left); break;
                case 4:  status = ippiCopyMirrorBorder_16s_C4R_L((Ipp16s*)pSrc, srcStep, size, (Ipp16s*)pDst, dstStep, dstSize, borderSize.top, borderSize.left); break;
                default: status = ippStsNumChannelsErr; break;
                }
                break;
            case ipp32s:
                switch(channels)
                {
                case 1:  status = ippiCopyMirrorBorder_32s_C1R_L((Ipp32s*)pSrc, srcStep, size, (Ipp32s*)pDst, dstStep, dstSize, borderSize.top, borderSize.left); break;
                case 3:  status = ippiCopyMirrorBorder_32s_C3R_L((Ipp32s*)pSrc, srcStep, size, (Ipp32s*)pDst, dstStep, dstSize, borderSize.top, borderSize.left); break;
                case 4:  status = ippiCopyMirrorBorder_32s_C4R_L((Ipp32s*)pSrc, srcStep, size, (Ipp32s*)pDst, dstStep, dstSize, borderSize.top, borderSize.left); break;
                default: status = ippStsNumChannelsErr; break;
                }
                break;
            case ipp32f:
                switch(channels)
                {
                case 1:  status = ippiCopyMirrorBorder_32f_C1R_L((Ipp32f*)pSrc, srcStep, size, (Ipp32f*)pDst, dstStep, dstSize, borderSize.top, borderSize.left); break;
                case 3:  status = ippiCopyMirrorBorder_32f_C3R_L((Ipp32f*)pSrc, srcStep, size, (Ipp32f*)pDst, dstStep, dstSize, borderSize.top, borderSize.left); break;
                case 4:  status = ippiCopyMirrorBorder_32f_C4R_L((Ipp32f*)pSrc, srcStep, size, (Ipp32f*)pDst, dstStep, dstSize, borderSize.top, borderSize.left); break;
                default: status = ippStsNumChannelsErr; break;
                }
                break;
            default: status = ippStsDataTypeErr; break;
            }
        }
        else if(border == ippBorderWrap)
        {
            switch(dataType)
            {
            case ipp32s:
                switch(channels)
                {
                case 1:  status = ippiCopyWrapBorder_32s_C1R_L((Ipp32s*)pSrc, srcStep, size, (Ipp32s*)pDst, dstStep, dstSize, borderSize.top, borderSize.left); break;
                default: status = ippStsNumChannelsErr; break;
                }
                break;
            case ipp32f:
                switch(channels)
                {
                case 1:  status = ippiCopyWrapBorder_32f_C1R_L((Ipp32f*)pSrc, srcStep, size, (Ipp32f*)pDst, dstStep, dstSize, borderSize.top, borderSize.left); break;
                default: status = ippStsNumChannelsErr; break;
                }
                break;
            default: status = ippStsDataTypeErr; break;
            }
        }
        else
            status = ippStsBorderErr;
    }

    return status;
}
