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

#include "iw_owni.h"
#include "iw/iw_image.h"

IW_DECL(IppStatus) llwiCopyMakeBorder(const void *pSrc, IwSize srcStep, void *pDst, IwSize dstStep,
    IwiSize size, IppDataType dataType, int channels, IwiBorderSize borderSize, IwiBorderType border, const Ipp64f *pBorderVal);

/* /////////////////////////////////////////////////////////////////////////////
//                   Utility functions
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(IwiSize) iwiMaskToSize(IppiMaskSize mask)
{
    IwiSize size;
    size.width  = 0;
    size.height = 0;

    switch(mask)
    {
    case ippMskSize1x3:
        size.width  = 1;
        size.height = 3;
        return size;
    case ippMskSize1x5:
        size.width  = 1;
        size.height = 5;
        return size;
    case ippMskSize3x1:
        size.width  = 3;
        size.height = 1;
        return size;
    case ippMskSize3x3:
        size.width  = 3;
        size.height = 3;
        return size;
    case ippMskSize5x1:
        size.width  = 5;
        size.height = 1;
        return size;
    case ippMskSize5x5:
        size.width  = 5;
        size.height = 5;
        return size;
    default:
        return size;
    }
}

IW_DECL(OwniChCodes) owniChDescriptorToCode(IwiChDescriptor chDesc, int srcChannels, int dstChannels)
{
    if(srcChannels == dstChannels)
    {
        if((int)OWN_DESC_GET_CH(chDesc) != srcChannels)
            chDesc = iwiChDesc_None;
    }
    else
        chDesc = iwiChDesc_None;

    switch(srcChannels)
    {
    case 1:
        switch(dstChannels)
        {
        case 1:
            switch(chDesc)
            {
            case iwiChDesc_None:    return owniC1;
            default:                return owniC_Invalid;
            }
        case 3:
            switch(chDesc)
            {
            case iwiChDesc_None:    return owniC1C3;
            default:                return owniC_Invalid;
            }
        case 4:
            switch(chDesc)
            {
            case iwiChDesc_None:    return owniC1C4;
            default:                return owniC_Invalid;
            }
        default: return owniC_Invalid;
        }
    case 3:
        switch(dstChannels)
        {
        case 1:
            switch(chDesc)
            {
            case iwiChDesc_None:    return owniC3C1;
            default:                return owniC_Invalid;
            }
        case 3:
            switch(chDesc)
            {
            case iwiChDesc_None:    return owniC3;
            default:                return owniC_Invalid;
            }
        case 4:
            switch(chDesc)
            {
            case iwiChDesc_None:    return owniC3C4;
            default:                return owniC_Invalid;
            }
        default: return owniC_Invalid;
        }
    case 4:
        switch(dstChannels)
        {
        case 1:
            switch(chDesc)
            {
            case iwiChDesc_None: return owniC4C1;
            default:             return owniC_Invalid;
            }
        case 3:
            switch(chDesc)
            {
            case iwiChDesc_None: return owniC4C3;
            default:             return owniC_Invalid;
            }
        case 4:
            switch(chDesc)
            {
            case iwiChDesc_None:    return owniC4;
            case iwiChDesc_C4M1110: return owniC4M1110;
            case iwiChDesc_C4M1000: return owniC4M1000;
            case iwiChDesc_C4M1001: return owniC4M1001;
            case iwiChDesc_C4M1XX0: return owniC4M1RR0;
            case iwiChDesc_C4M1XX1: return owniC4M1RR1;
            default:                return owniC_Invalid;
            }
        default: return owniC_Invalid;
        }
    default: return owniC_Invalid;
    }
}

/* /////////////////////////////////////////////////////////////////////////////
//                   IwiImage - Image structure
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(void) iwiImage_Init(IwiImage *pImage)
{
    if(!pImage)
        return;

    pImage->m_channels    = 0;
    pImage->m_dataType    = ipp8u;
    pImage->m_typeSize    = 0;
    pImage->m_size.width  = 0;
    pImage->m_size.height = 0;
    pImage->m_pBuffer     = NULL;
    pImage->m_ptr         = NULL;
    pImage->m_ptrConst    = NULL;
    pImage->m_step        = 0;

    pImage->m_inMemSize.left = pImage->m_inMemSize.right  = 0;
    pImage->m_inMemSize.top  = pImage->m_inMemSize.bottom = 0;
}

IW_DECL(IppStatus) iwiImage_InitExternal(IwiImage *pImage, IwiSize size, IppDataType dataType, int channels, IwiBorderSize const *pInMemBorder, void *pBuffer, IwSize step)
{
    if(!pImage)
        return ippStsNullPtrErr;

    iwiImage_Init(pImage);
    if(channels < 0)
        return ippStsNumChannelsErr;
    if(size.width < 0 || size.height < 0)
        return ippStsSizeErr;

    pImage->m_typeSize = iwTypeToSize(dataType);
    if(!pImage->m_typeSize)
        return ippStsDataTypeErr;

    pImage->m_dataType = dataType;
    pImage->m_size     = size;
    pImage->m_channels = channels;

    if(pInMemBorder)
    {
        if(owniBorderSizeIsNegative(pInMemBorder))
            return iwStsBorderNegSizeErr;

        pImage->m_inMemSize.left   = pInMemBorder->left;
        pImage->m_inMemSize.right  = pInMemBorder->right;
        pImage->m_inMemSize.top    = pInMemBorder->top;
        pImage->m_inMemSize.bottom = pInMemBorder->bottom;
    }

    pImage->m_ptr      = pBuffer;
    pImage->m_ptrConst = pImage->m_ptr;
    pImage->m_step     = step;

    return ippStsNoErr;
}

IW_DECL(IppStatus) iwiImage_InitExternalConst(IwiImage *pImage, IwiSize size, IppDataType dataType, int channels, IwiBorderSize const *pInMemBorder, const void *pBuffer, IwSize step)
{
    if(!pImage)
        return ippStsNullPtrErr;

    iwiImage_Init(pImage);

    if(channels < 0)
        return ippStsNumChannelsErr;
    if(size.width < 0 || size.height < 0)
        return ippStsSizeErr;

    pImage->m_typeSize = iwTypeToSize(dataType);
    if(!pImage->m_typeSize)
        return ippStsDataTypeErr;

    pImage->m_dataType = dataType;
    pImage->m_size     = size;
    pImage->m_channels = channels;

    if(pInMemBorder)
    {
        if(owniBorderSizeIsNegative(pInMemBorder))
            return iwStsBorderNegSizeErr;

        pImage->m_inMemSize.left   = pInMemBorder->left;
        pImage->m_inMemSize.right  = pInMemBorder->right;
        pImage->m_inMemSize.top    = pInMemBorder->top;
        pImage->m_inMemSize.bottom = pInMemBorder->bottom;
    }

    pImage->m_ptrConst = pBuffer;
    pImage->m_step     = step;

    return ippStsNoErr;
}

IW_DECL(IppStatus) iwiImage_Alloc(IwiImage *pImage, IwiSize size, IppDataType dataType, int channels, IwiBorderSize const *pInMemBorder)
{
    IwSize     step;
    IwiSize    allocSize;

    if(!pImage)
        return ippStsNullPtrErr;

    iwiImage_Release(pImage);

    if(size.width < 0 || size.height < 0)
        return ippStsSizeErr;

    if(channels < 0)
        return ippStsNumChannelsErr;

    pImage->m_typeSize = iwTypeToSize(dataType);
    if(!pImage->m_typeSize)
        return ippStsDataTypeErr;

    pImage->m_dataType = dataType;
    pImage->m_size     = size;
    pImage->m_channels = channels;

    if(pInMemBorder)
    {
        if(owniBorderSizeIsNegative(pInMemBorder))
            return iwStsBorderNegSizeErr;

        pImage->m_inMemSize.left   = pInMemBorder->left;
        pImage->m_inMemSize.right  = pInMemBorder->right;
        pImage->m_inMemSize.top    = pInMemBorder->top;
        pImage->m_inMemSize.bottom = pInMemBorder->bottom;
    }
    allocSize.width  = pImage->m_size.width + pImage->m_inMemSize.left + pImage->m_inMemSize.right;
    allocSize.height = pImage->m_size.height + pImage->m_inMemSize.top + pImage->m_inMemSize.bottom;
    step             = allocSize.width*pImage->m_typeSize*pImage->m_channels;
    if(!step || !allocSize.height) // zero size memory request
        return ippStsNoErr;

    if(allocSize.height > 1 && step*allocSize.height > 64)
    {
        // Align rows by cache lines
        if(step < 16)
            step = 16;
        else if(step < 32)
            step = 32;
        else
            step = owniAlignStep(step, 64);
    }

    pImage->m_pBuffer = ippsMalloc_8u_L(step*allocSize.height);
    if(!pImage->m_pBuffer)
        return ippStsMemAllocErr;

    pImage->m_ptr      = (Ipp8u*)pImage->m_pBuffer + pImage->m_inMemSize.left*pImage->m_typeSize*pImage->m_channels + step*pImage->m_inMemSize.top;
    pImage->m_ptrConst = pImage->m_ptr;
    pImage->m_step     = step;

    return ippStsNoErr;
}

IW_DECL(void) iwiImage_Release(IwiImage *pImage)
{
    if(!pImage)
        return;

    if(pImage->m_pBuffer)
    {
        ippsFree(pImage->m_pBuffer);
        pImage->m_pBuffer  = NULL;
        pImage->m_ptr      = NULL;
        pImage->m_ptrConst = NULL;
        pImage->m_step     = 0;
    }
}

IW_DECL(void*) iwiImage_GetPtr(const IwiImage *pImage, IwSize y, IwSize x, int ch)
{
    if(!pImage || !pImage->m_ptr)
        return NULL;
    return (((Ipp8u*)pImage->m_ptr) + pImage->m_step*y + x*pImage->m_typeSize*pImage->m_channels + ch*pImage->m_typeSize);
}

IW_DECL(const void*) iwiImage_GetPtrConst(const IwiImage *pImage, IwSize y, IwSize x, int ch)
{
    if(!pImage || !pImage->m_ptrConst)
        return NULL;
    return (((const Ipp8u*)pImage->m_ptrConst) + pImage->m_step*y + x*pImage->m_typeSize*pImage->m_channels + ch*pImage->m_typeSize);
}

IW_DECL(IppStatus) iwiImage_BorderAdd(IwiImage *pImage, IwiBorderSize borderSize)
{
    if(!pImage || !pImage->m_ptrConst)
        return ippStsNullPtrErr;

    if(owniBorderSizeIsNegative(&borderSize))
        return iwStsBorderNegSizeErr;

    if(borderSize.left + borderSize.right >= pImage->m_size.width)
        return ippStsSizeErr;
    if(borderSize.top + borderSize.bottom >= pImage->m_size.height)
        return ippStsSizeErr;

    if(pImage->m_ptr)
        pImage->m_ptrConst = pImage->m_ptr = iwiImage_GetPtr(pImage, borderSize.top, borderSize.left, 0);
    else
        pImage->m_ptrConst = iwiImage_GetPtrConst(pImage, borderSize.top, borderSize.left, 0);
    pImage->m_size.width  = pImage->m_size.width  - borderSize.left - borderSize.right;
    pImage->m_size.height = pImage->m_size.height - borderSize.top  - borderSize.bottom;
    pImage->m_inMemSize.left   += borderSize.left;
    pImage->m_inMemSize.top    += borderSize.top;
    pImage->m_inMemSize.right  += borderSize.right;
    pImage->m_inMemSize.bottom += borderSize.bottom;

    return ippStsNoErr;
}

IW_DECL(IppStatus) iwiImage_BorderSub(IwiImage *pImage, IwiBorderSize borderSize)
{
    if(!pImage || !pImage->m_ptrConst)
        return ippStsNullPtrErr;

    if(owniBorderSizeIsNegative(&borderSize))
        return iwStsBorderNegSizeErr;

    if(borderSize.left > pImage->m_inMemSize.left)
        return ippStsOutOfRangeErr;
    if(borderSize.top > pImage->m_inMemSize.top)
        return ippStsOutOfRangeErr;
    if(borderSize.right > pImage->m_inMemSize.right)
        return ippStsOutOfRangeErr;
    if(borderSize.bottom > pImage->m_inMemSize.bottom)
        return ippStsOutOfRangeErr;

    if(pImage->m_ptr)
        pImage->m_ptrConst = pImage->m_ptr = iwiImage_GetPtr(pImage, -borderSize.top, -borderSize.left, 0);
    else
        pImage->m_ptrConst = iwiImage_GetPtrConst(pImage, -borderSize.top, -borderSize.left, 0);
    pImage->m_size.width  = pImage->m_size.width  + borderSize.left + borderSize.right;
    pImage->m_size.height = pImage->m_size.height + borderSize.top  + borderSize.bottom;
    pImage->m_inMemSize.left   -= borderSize.left;
    pImage->m_inMemSize.top    -= borderSize.top;
    pImage->m_inMemSize.right  -= borderSize.right;
    pImage->m_inMemSize.bottom -= borderSize.bottom;

    return ippStsNoErr;
}

IW_DECL(IppStatus) iwiImage_BorderSet(IwiImage *pImage, IwiBorderSize borderSize)
{
    IwSize diffLeft, diffTop, diffRight, diffBottom;

    if(!pImage || !pImage->m_ptrConst)
        return ippStsNullPtrErr;

    if(owniBorderSizeIsNegative(&borderSize))
        return iwStsBorderNegSizeErr;

    diffLeft    = borderSize.left    - pImage->m_inMemSize.left;
    diffTop     = borderSize.top     - pImage->m_inMemSize.top;
    diffRight   = borderSize.right   - pImage->m_inMemSize.right;
    diffBottom  = borderSize.bottom  - pImage->m_inMemSize.bottom;

    if(diffLeft+diffRight >= pImage->m_size.width)
        return ippStsSizeErr;
    if(diffTop+diffBottom >= pImage->m_size.height)
        return ippStsSizeErr;

    if(pImage->m_ptr)
        pImage->m_ptrConst = pImage->m_ptr = iwiImage_GetPtr(pImage, diffTop, diffLeft, 0);
    else
        pImage->m_ptrConst = iwiImage_GetPtrConst(pImage, diffTop, diffLeft, 0);
    pImage->m_size.width  = pImage->m_size.width  - (diffLeft + diffRight);
    pImage->m_size.height = pImage->m_size.height - (diffTop + diffBottom);
    pImage->m_inMemSize   = borderSize;

    return ippStsNoErr;
}

IW_DECL(IppStatus) iwiImage_RoiSet(IwiImage *pImage, IwiRoi roi)
{
    if(!pImage || !pImage->m_ptrConst)
        return ippStsNullPtrErr;

    // Unroll border
    if(pImage->m_ptr)
        pImage->m_ptrConst = pImage->m_ptr = iwiImage_GetPtr(pImage, -pImage->m_inMemSize.top, -pImage->m_inMemSize.left, 0);
    else
        pImage->m_ptrConst = iwiImage_GetPtrConst(pImage, -pImage->m_inMemSize.top, -pImage->m_inMemSize.left, 0);
    pImage->m_size.width  = pImage->m_size.width  + pImage->m_inMemSize.left + pImage->m_inMemSize.right;
    pImage->m_size.height = pImage->m_size.height + pImage->m_inMemSize.top  + pImage->m_inMemSize.bottom;
    roi.x += pImage->m_inMemSize.left;
    roi.y += pImage->m_inMemSize.top;

    // ROI saturation
    if(roi.width < 0) // Inverted ROI
    {
        roi.x     = roi.x + roi.width;
        roi.width = -roi.width;
    }
    if(roi.x < 0) // "Left" saturation
    {
        roi.width += roi.x;
        roi.x = 0;
    }
    if(roi.x + roi.width > pImage->m_size.width) // "Right" saturation
    {
        if(roi.x > pImage->m_size.width)
        {
            roi.x = pImage->m_size.width;
            roi.width = 0;
        }
        else
            roi.width = pImage->m_size.width - roi.x;
    }

    if(roi.height < 0) // Inverted ROI
    {
        roi.y      = roi.y + roi.height;
        roi.height = -roi.height;
    }
    if(roi.y < 0) // "Left" saturation
    {
        roi.height += roi.y;
        roi.y = 0;
    }
    if(roi.y + roi.height > pImage->m_size.height) // "Right" saturation
    {
        if(roi.y > pImage->m_size.height)
        {
            roi.y = pImage->m_size.height;
            roi.height = 0;
        }
        else
            roi.height = pImage->m_size.height - roi.y;
    }

    // Rebuild border
    pImage->m_inMemSize.left   = roi.x;
    pImage->m_inMemSize.top    = roi.y;
    pImage->m_inMemSize.right  = pImage->m_size.width  - roi.x - roi.width;
    pImage->m_inMemSize.bottom = pImage->m_size.height - roi.y - roi.height;
    pImage->m_size.width    = roi.width;
    pImage->m_size.height   = roi.height;
    if(pImage->m_ptr)
        pImage->m_ptrConst = pImage->m_ptr = iwiImage_GetPtr(pImage, pImage->m_inMemSize.top, pImage->m_inMemSize.left, 0);
    else
        pImage->m_ptrConst = iwiImage_GetPtrConst(pImage, pImage->m_inMemSize.top, pImage->m_inMemSize.left, 0);

    return ippStsNoErr;
}

IW_DECL(IwiImage) iwiImage_GetRoiImage(const IwiImage *pImage, IwiRoi roi)
{
    IwiImage image;

    iwiImage_Init(&image);
    if(!pImage || !pImage->m_ptrConst)
        return image;

    if(pImage->m_ptr)
    {
        if(iwiImage_InitExternal(&image, pImage->m_size, pImage->m_dataType, pImage->m_channels, &pImage->m_inMemSize, pImage->m_ptr, pImage->m_step) < 0)
        {
            iwiImage_Init(&image);
            return image;
        }
    }
    else
    {
        if(iwiImage_InitExternalConst(&image, pImage->m_size, pImage->m_dataType, pImage->m_channels, &pImage->m_inMemSize, pImage->m_ptrConst, pImage->m_step) < 0)
        {
            iwiImage_Init(&image);
            return image;
        }
    }
    if(iwiImage_RoiSet(&image, roi) < 0)
    {
        iwiImage_Init(&image);
        return image;
    }
    return image;
}

/* /////////////////////////////////////////////////////////////////////////////
//                   IW Tiling
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(IwiSize) owniSuggestTileSize_k2(const IwiImage *pImage, IwiSize kernelSize, double multiplier)
{
    float      nCaches;
    IwiSize    roi       = {0,0};
    IwSize     opMemory  = (IwSize)(pImage->m_size.width*pImage->m_size.height*pImage->m_typeSize*pImage->m_channels*multiplier);
    IwSize     minHeight = IPP_MAX(kernelSize.height, 64);
    IwSize     minWidth  = IPP_MAX(kernelSize.width, 64);
    IppCache  *pCache;
    int        l2cache = 262144;
    //int        l3cache = 1048576;
    if(ippGetCacheParams(&pCache) >= 0 && pCache[0].type != 0 && pCache[1].type != 0)
    {
        if(pCache[2].type != 0)
        {
            l2cache = pCache[2].size;
            //if(pCache[3].type != 0)
            //    l3cache = pCache[3].size;
        }
    }

    nCaches = ((float)opMemory/l2cache);
    if(nCaches < 1)     // Cache is enough to contain all operations
        return pImage->m_size;
    else if(pImage->m_size.height <= minHeight) // Height is small, divide by width only
    {
        roi.width  = (IwSize)((pImage->m_size.width+nCaches-1)/nCaches);
        roi.height = kernelSize.height;
    }
    else // General case
    {
        IwSize pixChunk = (IwSize)((pImage->m_size.width*pImage->m_size.height)/nCaches);
        roi.height      = (pixChunk+pImage->m_size.width-1)/pImage->m_size.width;
        roi.width       = pImage->m_size.width;
        if(roi.height < minHeight && roi.width > minWidth) // Not enough space to process whole row efficiently, divide row
        {
            while(roi.height < minHeight)
            {
                roi.width /= 2;
                roi.height = (pixChunk+roi.width-1)/roi.width;
            }
        }
    }
    return roi;
}

/* /////////////////////////////////////////////////////////////////////////////
//                   Manual tiling control
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(IwiBorderType) iwiTile_GetTileBorder(IwiRoi roi, IwiBorderType border, IwiBorderSize borderSize, IwiSize srcImageSize)
{
    owniTile_GetTileBorder(&border, &roi, &borderSize, &srcImageSize);
    return border;
}

IW_DECL(IwiSize) iwiTile_GetMinTileSize(IwiBorderType border, IwiBorderSize borderSize)
{
    IwiSize size;
    if(border&ippBorderInMemLeft)
        borderSize.left   = 0;
    if(border&ippBorderInMemRight)
        borderSize.right  = 0;
    if(border&ippBorderInMemTop)
        borderSize.top    = 0;
    if(border&ippBorderInMemBottom)
        borderSize.bottom = 0;

    border      = OWN_GET_PURE_BORDER(border);
    size.width  = IPP_MAX(borderSize.left, borderSize.right);
    size.height = IPP_MAX(borderSize.top, borderSize.bottom);
    if(size.width == 0)
        size.width  = 1;
    if(size.height == 0)
        size.height = 1;
    if(border == ippBorderMirror)
    {
        size.width++;
        size.height++;
    }
    return size;
}

IW_DECL(IwiRoi) iwiTile_CorrectBordersOverlap(IwiRoi roi, IwiBorderType border, IwiBorderSize borderSize, IwiSize srcImageSize)
{
    owniTile_CorrectBordersOverlap(&roi, NULL, &border, &borderSize, &borderSize, &srcImageSize);
    return roi;
}

/* /////////////////////////////////////////////////////////////////////////////
//                   IwiTile tiling
///////////////////////////////////////////////////////////////////////////// */
IW_DECL(int) owniTile_BoundToSize(IwiRoi *pRoi, IwiSize *pMinSize)
{
    if(pRoi->x >= pMinSize->width)
        return 0;
    else if(pRoi->x < 0)
        pRoi->x = 0;
    if(pRoi->y >= pMinSize->height)
        return 0;
    else if(pRoi->y < 0)
        pRoi->y = 0;

    if(pRoi->width + pRoi->x > pMinSize->width)
        pRoi->width  = pMinSize->width  - pRoi->x;
    if(pRoi->height + pRoi->y > pMinSize->height)
        pRoi->height = pMinSize->height - pRoi->y;

    if(pRoi->width <= 0 || pRoi->height <= 0)
        return 0;
    else
    {
        pMinSize->width  = pRoi->width;
        pMinSize->height = pRoi->height;
        return 1;
    }
}


IW_DECL(int) owniTile_CorrectBordersOverlap(IwiRoi *pRoi, IwiSize *pMinSize, const IwiBorderType *pBorder, const IwiBorderSize *pBorderSize, const IwiBorderSize *pBorderSizeAcc, const IwiSize *pSrcImageSize)
{
    int corrected = 0;

    // Check right border
    if(pBorderSize->right > 1 && !(*pBorder&ippBorderInMemRight))
    {
        IwSize rightPos  = pRoi->x + pRoi->width; // Get right tile border position
        IwSize leftCheck = pRoi->x + (pBorderSizeAcc->left - pBorderSize->left) + (pBorderSizeAcc->right - pBorderSize->right); // Reconstruct previous tile position

        // Border-phobic tile part
        if(rightPos < pSrcImageSize->width &&
            (rightPos + pBorderSize->right) > pSrcImageSize->width)
        {
            pRoi->width = pSrcImageSize->width - pBorderSize->right - pRoi->x;
            corrected = 1;
        }
        // Border-filing tile part
        else if(leftCheck < pSrcImageSize->width &&
            (leftCheck + pBorderSize->right) > pSrcImageSize->width)
        {
            pRoi->x     = pSrcImageSize->width - pBorderSize->right - (pBorderSizeAcc->left - pBorderSize->left)*2;
            pRoi->width = pSrcImageSize->width - pRoi->x;
            corrected = 1;
        }
    }

    // Check bottom border
    if(pBorderSize->bottom > 1 && !(*pBorder&ippBorderInMemBottom))
    {
        IwSize bottomPos = pRoi->y + pRoi->height; // Get bottom tile border position
        IwSize topCheck  = pRoi->y + (pBorderSizeAcc->top - pBorderSize->top) + (pBorderSizeAcc->bottom - pBorderSize->bottom); // Reconstruct previous tile position

        // Border-phobic tile part
        if(bottomPos < pSrcImageSize->height &&
            (bottomPos + pBorderSize->bottom) > pSrcImageSize->height)
        {
            pRoi->height = pSrcImageSize->height - pBorderSize->bottom - pRoi->y;
            corrected = 1;
        }
        // Border-filing tile part
        else if(topCheck < pSrcImageSize->height &&
            (topCheck + pBorderSize->bottom) > pSrcImageSize->height)
        {
            pRoi->y      = pSrcImageSize->height - pBorderSize->bottom - (pBorderSizeAcc->top - pBorderSize->top)*2;
            pRoi->height = pSrcImageSize->height - pRoi->y;
            corrected = 1;
        }
    }

    if(corrected && pMinSize)
    {
        pMinSize->width  = pRoi->width;
        pMinSize->height = pRoi->height;
    }

    return corrected;
}

IW_DECL(void) owniTile_GetTileBorder(IwiBorderType *pBorder, const IwiRoi *pRoi, const IwiBorderSize *pBorderSize, const IwiSize *pSrcImageSize)
{
    if((*pBorder & ippBorderInMem) != ippBorderInMem)
    {
        int flags = (*pBorder)&(~0xF);

        // Check left border
        if(pBorderSize->left && (pRoi->x >= pBorderSize->left))
            flags |= ippBorderInMemLeft;

        // Check top border
        if(pBorderSize->top && (pRoi->y >= pBorderSize->top))
            flags |= ippBorderInMemTop;

        // Check right border
        if(pBorderSize->right && (pRoi->x + pRoi->width + pBorderSize->right <= pSrcImageSize->width))
            flags |= ippBorderInMemRight;

        // Check bottom border
        if(pBorderSize->bottom && (pRoi->y + pRoi->height + pBorderSize->bottom <= pSrcImageSize->height))
            flags |= ippBorderInMemBottom;

        // If we have full InMem, then we should pass pure InMem border to function
        if(flags == ippBorderInMem)
            *pBorder = ippBorderInMem;
        else
            *pBorder = (IwiBorderType)((*pBorder)|flags);
    }
}

/* /////////////////////////////////////////////////////////////////////////////
//                   IwiTile pipeline tiling
///////////////////////////////////////////////////////////////////////////// */

static IW_INLINE IwiTile* owniTilePipeline_GetRoot(const IwiTile *pTile)
{
    IwiTile *pRoiRoot = (IwiTile*)pTile;
    if(!pTile || pTile->m_initialized != ownTileInitPipe)
        return NULL;
    while(pRoiRoot->m_pParent)
        pRoiRoot = pRoiRoot->m_pParent;
    return pRoiRoot;
}
static IW_INLINE IppStatus owniTilePipeline_InitCheck(const IwiTile *pTile)
{
    if(!pTile)
        return ippStsNullPtrErr;

    if(pTile->m_initialized != ownTileInitPipe)
        return ippStsContextMatchErr;

    return ippStsNoErr;
}

IW_DECL(IwiTile) iwiTile_SetRoi(IwiRoi tileRoi)
{
    IwiTile roi = {0};
    roi.m_dstRoi      = tileRoi;
    roi.m_initialized = ownTileInitSimple;
    return roi;
}

static IppStatus owniTilePipeline_GetSrcSizeMax(IwiRoi *pSrcRoiSize, IwiSize *pDstMaxTile, IwiSize *pDstImageSize, const IwiTileTransform *pTransformStruct)
{
    IwiRoi   srcTemp   = {0, 0, 0, 0};
    IwiRoi   dstRoi    = {0, 0, 0, 0};
    pSrcRoiSize->width  = 0;
    pSrcRoiSize->height = 0;
    dstRoi.width  = pDstMaxTile->width;
    dstRoi.height = pDstMaxTile->height;

    for(dstRoi.x = 0; dstRoi.x <= pDstImageSize->width - dstRoi.width; dstRoi.x++)
    {
        if(pTransformStruct->getSrcRoiFun(dstRoi, &srcTemp, pTransformStruct->pParams))
            return ippStsErr;

        pSrcRoiSize->width = IPP_MAX(pSrcRoiSize->width, srcTemp.width);
    }

    for(dstRoi.y = 0; dstRoi.y <= pDstImageSize->height - dstRoi.height; dstRoi.y++)
    {
        if(pTransformStruct->getSrcRoiFun(dstRoi, &srcTemp, pTransformStruct->pParams))
            return ippStsErr;

        pSrcRoiSize->height = IPP_MAX(pSrcRoiSize->height, srcTemp.height);
    }

    pSrcRoiSize->x = 0;
    pSrcRoiSize->y = 0;

    return ippStsNoErr;
}

static IppStatus owniTilePipeline_BuildBorder(const IwiTile *pTile, IwiImage *pSrcImage, IwiBorderType *pBorder, const Ipp64f *pBorderVal, const IwiBorderSize *pBorderSize)
{
    IppStatus status;

    if(((*pBorder)&ippBorderInMem) != ippBorderInMem)
    {
        if(pBorderSize)
            status = llwiCopyMakeBorder(pSrcImage->m_ptrConst, pSrcImage->m_step, pSrcImage->m_ptr, pSrcImage->m_step, pSrcImage->m_size,
                pSrcImage->m_dataType, pSrcImage->m_channels, *pBorderSize, *pBorder, pBorderVal);
        else
            status = llwiCopyMakeBorder(pSrcImage->m_ptrConst, pSrcImage->m_step, pSrcImage->m_ptr, pSrcImage->m_step, pSrcImage->m_size,
                pSrcImage->m_dataType, pSrcImage->m_channels, pTile->m_borderSize, *pBorder, pBorderVal);
        if(status < 0)
            return status;

        *pBorder = ippBorderInMem;
    }

    return ippStsNoErr;
}

IW_DECL(IppStatus) owniTilePipeline_ProcBorder(const IwiTile *pTile, IwiImage *pSrcImage, IwiBorderType *pBorder, const Ipp64f *pBorderVal)
{
    *pBorder = pTile->m_borderType;
    owniTile_GetTileBorder(pBorder, &pTile->m_srcRoi, &pTile->m_borderSize, &pTile->m_srcImageSize);

    if(pTile->m_pChild)
    {
        IppStatus     status;
        IwiBorderSize borderSize     = pTile->m_borderSize;
        IwiBorderSize borderSizeDiff = {0, 0, 0, 0};
        int           overlap        = 0;
        IwSize        rightPos       = pTile->m_srcRoi.x + pTile->m_srcRoi.width;
        IwSize        bottomPos      = pTile->m_srcRoi.y + pTile->m_srcRoi.height;

        // Pre-process partial InMem border. Shift buffer in InMem boundary and extrapolate the rest
        // Check left border
        if(pTile->m_borderSize.left && (pTile->m_srcRoi.x > 0 && pTile->m_srcRoi.x < pTile->m_borderSize.left))
        {
            borderSizeDiff.left = pTile->m_srcRoi.x;
            borderSize.left    -= borderSizeDiff.left;
            overlap             = 1;
        }

        // Check top border
        if(pTile->m_borderSize.top && (pTile->m_srcRoi.y > 0 && pTile->m_srcRoi.y < pTile->m_borderSize.top))
        {
            borderSizeDiff.top = pTile->m_srcRoi.y;
            borderSize.top    -= borderSizeDiff.top;
            overlap            = 1;
        }

        // Check right border
        if(pTile->m_borderSize.right && (rightPos + pTile->m_borderSize.right > pTile->m_srcImageSize.width && rightPos < pTile->m_srcImageSize.width))
        {
            borderSizeDiff.right = pTile->m_srcImageSize.width - rightPos;
            borderSize.right    -= borderSizeDiff.right;
            overlap              = 1;
        }

        // Check bottom border
        if(pTile->m_borderSize.bottom && (bottomPos + pTile->m_borderSize.bottom > pTile->m_srcImageSize.height && bottomPos < pTile->m_srcImageSize.height))
        {
            borderSizeDiff.bottom = pTile->m_srcImageSize.height - bottomPos;
            borderSize.bottom    -= borderSizeDiff.bottom;
            overlap               = 1;
        }

        if(overlap)
        {
            IwiImage tmpImage;

            iwiImage_InitExternal(&tmpImage, pSrcImage->m_size, pSrcImage->m_dataType, pSrcImage->m_channels, &pSrcImage->m_inMemSize, pSrcImage->m_ptr, pSrcImage->m_step);

            status = iwiImage_BorderSub(&tmpImage, borderSizeDiff);
            if(status < 0)
                return status;

            // Extrapolate border for intermediate buffers
            return owniTilePipeline_BuildBorder(pTile, &tmpImage, pBorder, pBorderVal, &borderSize);
        }
        else
        {
            // Extrapolate border for intermediate buffers
            return owniTilePipeline_BuildBorder(pTile, pSrcImage, pBorder, pBorderVal, &borderSize);
        }
    }
    return ippStsNoErr;
}

static IppStatus owniTilePipeline_InitCommon(IwiTile *pTile, const IwiBorderType *pBorderType, const IwiBorderSize *pBorderSize, const IwiTileTransform *pTransformStruct)
{
    IppStatus status;

    if(pTile->m_maxTileSize.width > pTile->m_dstImageSize.width)
        pTile->m_maxTileSize.width = pTile->m_dstImageSize.width;
    if(pTile->m_maxTileSize.height > pTile->m_dstImageSize.height)
        pTile->m_maxTileSize.height = pTile->m_dstImageSize.height;

    if(pTransformStruct && pTransformStruct->getSrcRoiFun)
    {
        pTile->m_transformStruct = *pTransformStruct;

        status = owniTilePipeline_GetSrcSizeMax(&pTile->m_srcRoi, &pTile->m_maxTileSize, &pTile->m_dstExImageSize, pTransformStruct);
        if(status < 0)
            return status;

        pTile->m_srcImageSize  = pTile->m_transformStruct.srcImageSize;
        pTile->m_srcBufferSize = pTile->m_transformStruct.srcImageSize;
    }

    pTile->m_borderType = ippBorderRepl;
    if(pBorderSize)
    {
        pTile->m_borderSize = *pBorderSize;
        if(pBorderType)
        {
            pTile->m_borderType = *pBorderType;
            if(pTile->m_borderType&ippBorderInMemLeft)
            {
                pTile->m_externalMemAcc.left += pTile->m_borderSize.left;
                pTile->m_externalMem.left = pTile->m_borderSize.left;
            }
            if(pTile->m_borderType&ippBorderInMemRight)
            {
                pTile->m_externalMemAcc.right += pTile->m_borderSize.right;
                pTile->m_externalMem.right = pTile->m_borderSize.right;
            }
            if(pTile->m_borderType&ippBorderInMemTop)
            {
                pTile->m_externalMemAcc.top += pTile->m_borderSize.top;
                pTile->m_externalMem.top = pTile->m_borderSize.top;
            }
            if(pTile->m_borderType&ippBorderInMemBottom)
            {
                pTile->m_externalMemAcc.bottom += pTile->m_borderSize.bottom;
                pTile->m_externalMem.bottom = pTile->m_borderSize.bottom;
            }

            pTile->m_srcExImageSize.width  += (pTile->m_externalMem.left + pTile->m_externalMem.right);
            pTile->m_srcExImageSize.height += (pTile->m_externalMem.top + pTile->m_externalMem.bottom);
        }

        pTile->m_borderSizeAcc.left   += pTile->m_borderSize.left;
        pTile->m_borderSizeAcc.top    += pTile->m_borderSize.top;
        pTile->m_borderSizeAcc.right  += pTile->m_borderSize.right;
        pTile->m_borderSizeAcc.bottom += pTile->m_borderSize.bottom;
    }

    return ippStsNoErr;
}

IW_DECL(IppStatus) iwiTilePipeline_Init(IwiTile *pTile, IwiSize tileSizeMax, IwiSize imageSize, const IwiBorderType *pBorderType, const IwiBorderSize *pBorderSize, const IwiTileTransform *pTransformStruct)
{
    IppStatus status;

    if(!pTile)
        return ippStsNullPtrErr;

    ippsZero_8u((Ipp8u*)pTile, sizeof(IwiTile));

    if(tileSizeMax.width <= 0 || tileSizeMax.height <= 0)
        return ippStsSizeErr;

    if(tileSizeMax.width > imageSize.width)
        tileSizeMax.width = imageSize.width;
    if(tileSizeMax.height > imageSize.height)
        tileSizeMax.height = imageSize.height;

    pTile->m_maxTileSize   = tileSizeMax;

    pTile->m_srcRoi.width  = pTile->m_dstRoi.width  = pTile->m_maxTileSize.width;
    pTile->m_srcRoi.height = pTile->m_dstRoi.height = pTile->m_maxTileSize.height;

    pTile->m_srcImageSize   = pTile->m_dstImageSize   = imageSize;
    pTile->m_dstBufferSize  = pTile->m_srcBufferSize  = imageSize;
    pTile->m_dstExImageSize = pTile->m_srcExImageSize = imageSize;

    status = owniTilePipeline_InitCommon(pTile, pBorderType, pBorderSize, pTransformStruct);
    if(status < 0)
        return status;

    pTile->m_initialized   = ownTileInitPipe;
    return ippStsNoErr;
}

IW_DECL(IppStatus) iwiTilePipeline_InitChild(IwiTile *pTile, IwiTile *pParent, const IwiBorderType *pBorderType, const IwiBorderSize *pBorderSize, const IwiTileTransform *pTransformStruct)
{
    IppStatus status = owniTilePipeline_InitCheck(pParent);
    if(status < 0)
        return status;

    if(!pTile)
        return ippStsNullPtrErr;

    ippsZero_8u((Ipp8u*)pTile, sizeof(IwiTile));

    pParent->m_pChild = pTile;
    pTile->m_pParent   = pParent;

    pTile->m_maxTileSize.width  = pParent->m_srcRoi.width  + pParent->m_borderSize.left + pParent->m_borderSize.right;
    pTile->m_maxTileSize.height = pParent->m_srcRoi.height + pParent->m_borderSize.top  + pParent->m_borderSize.bottom;
    if(pParent->m_borderType == ippBorderMirror)
    {
        if(pParent->m_borderSize.left || pParent->m_borderSize.right)
            pTile->m_maxTileSize.width++;
        if(pParent->m_borderSize.top || pParent->m_borderSize.bottom)
            pTile->m_maxTileSize.height++;
    }

    pTile->m_dstImageSize    = pParent->m_srcImageSize;
    pTile->m_dstExImageSize  = pParent->m_srcExImageSize;
    pTile->m_srcExImageSize  = pTile->m_dstExImageSize;

    pTile->m_srcRoi.width  = pTile->m_dstRoi.width  = pTile->m_maxTileSize.width;
    pTile->m_srcRoi.height = pTile->m_dstRoi.height = pTile->m_maxTileSize.height;

    pTile->m_dstBufferSize.width  = pParent->m_srcBufferSize.width  = pTile->m_dstRoi.width;
    pTile->m_dstBufferSize.height = pParent->m_srcBufferSize.height = pTile->m_dstRoi.height;

    pTile->m_srcBufferSize  = pTile->m_dstImageSize;
    pTile->m_srcImageSize   = pTile->m_dstImageSize;
    pTile->m_borderSizeAcc  = pParent->m_borderSizeAcc;
    pTile->m_externalMemAcc = pParent->m_externalMemAcc;

    status = owniTilePipeline_InitCommon(pTile, pBorderType, pBorderSize, pTransformStruct);
    if(status < 0)
        return status;

    pTile->m_initialized = pParent->m_initialized;
    return ippStsNoErr;
}

IW_DECL(void) iwiTilePipeline_Release(IwiTile *pTile)
{
    pTile = owniTilePipeline_GetRoot(pTile);
    if(owniTilePipeline_InitCheck(pTile) < 0)
        return;

    while(pTile)
    {
        // Nothing to actually deallocate yet, but this may be helpful in the future.
        pTile->m_initialized = ownTileInitNone;

        pTile = pTile->m_pChild;
    }
}

IW_DECL(IppStatus) iwiTilePipeline_SetRoi(IwiTile *pTile, IwiRoi tileRoi)
{
    IppStatus  status;
    IwiTile   *pRoot          = NULL;
    int        hasScaling     = 0;
    double     scaleX         = 1;
    double     scaleY         = 1;

    pRoot  = owniTilePipeline_GetRoot(pTile);
    status = owniTilePipeline_InitCheck(pTile);
    if(status < 0)
        return status;

    if(tileRoi.x < 0)
        tileRoi.x = 0;
    if(tileRoi.y < 0)
        tileRoi.y = 0;

    if(tileRoi.width <= 0 || tileRoi.height <= 0)
        return ippStsSizeErr;

    if(tileRoi.width > pRoot->m_maxTileSize.width)
        tileRoi.width  = pRoot->m_maxTileSize.width;
    if(tileRoi.height > pRoot->m_maxTileSize.height)
        tileRoi.height = pRoot->m_maxTileSize.height;

    pTile = pRoot;
    while(pTile)
    {
        if(pTile->m_pParent)
        {
            pTile->m_dstRoi        = pTile->m_pParent->m_srcRoi;
            pTile->m_untaintDstPos = pTile->m_pParent->m_untaintSrcPos;

            pTile->m_dstRoi.width    += pTile->m_pParent->m_borderSize.left;
            pTile->m_dstRoi.x        -= pTile->m_pParent->m_borderSize.left;
            pTile->m_untaintDstPos.x -= pTile->m_pParent->m_borderSize.left;

            pTile->m_dstRoi.height    += pTile->m_pParent->m_borderSize.top;
            pTile->m_dstRoi.y         -= pTile->m_pParent->m_borderSize.top;
            pTile->m_untaintDstPos.y  -= pTile->m_pParent->m_borderSize.top;

            pTile->m_dstRoi.width  += pTile->m_pParent->m_borderSize.right;
            pTile->m_dstRoi.height += pTile->m_pParent->m_borderSize.bottom;

            if(pTile->m_dstRoi.x < 0)
            {
                if(!pTile->m_pParent->m_externalMem.left)
                {
                    pTile->m_dstRoi.width += pTile->m_dstRoi.x;
                    pTile->m_dstRoi.x = 0;
                }
            }

            if(pTile->m_dstRoi.y < 0)
            {
                if(!pTile->m_pParent->m_externalMem.top)
                {
                    pTile->m_dstRoi.height += pTile->m_dstRoi.y;
                    pTile->m_dstRoi.y = 0;
                }
            }

            if(pTile->m_dstRoi.x + pTile->m_dstRoi.width > pTile->m_dstImageSize.width + pTile->m_pParent->m_externalMemAcc.right)
                pTile->m_dstRoi.width = pTile->m_dstImageSize.width - pTile->m_dstRoi.x + pTile->m_pParent->m_externalMemAcc.right;
            if(pTile->m_dstRoi.y + pTile->m_dstRoi.height > pTile->m_dstImageSize.height + pTile->m_pParent->m_externalMemAcc.bottom)
                pTile->m_dstRoi.height = pTile->m_dstImageSize.height - pTile->m_dstRoi.y + pTile->m_pParent->m_externalMemAcc.bottom;
        }
        else
        {
            pTile->m_dstRoi          = tileRoi;
            pTile->m_untaintDstPos.x = pTile->m_dstRoi.x;
            pTile->m_untaintDstPos.y = pTile->m_dstRoi.y;

            if(pTile->m_dstRoi.x + pTile->m_dstRoi.width > pTile->m_dstImageSize.width)
                pTile->m_dstRoi.width = pTile->m_dstImageSize.width - pTile->m_dstRoi.x;
            if(pTile->m_dstRoi.y + pTile->m_dstRoi.height > pTile->m_dstImageSize.height)
                pTile->m_dstRoi.height = pTile->m_dstImageSize.height - pTile->m_dstRoi.y;
        }

        if(pTile->m_transformStruct.getSrcRoiFun)
        {
            // Scaling function should process partial borders correctly, so no overlap check is needed
            if(pTile->m_transformStruct.getSrcRoiFun(pTile->m_dstRoi, &pTile->m_srcRoi, pTile->m_transformStruct.pParams))
                return ippStsErr;

            if(pTile->m_untaintDstPos.x < 0)
            {
           //     pTile->m_srcRoi.x
           //     pTile->m_untaintSrcPos.x = pTile->m_untaintDstPos.x + pTile->m_srcRoi.x;
           //     pTile->m_srcRoi.width += -pTile->m_untaintDstPos.x;
            }
            if(pTile->m_untaintDstPos.y < 0)
            {
                pTile->m_untaintSrcPos.y = pTile->m_untaintDstPos.y + pTile->m_srcRoi.y;
          //      pTile->m_srcRoi.height += -pTile->m_untaintDstPos.y;
            }

            if(pTile->m_pParent)
            {
                if(pTile->m_srcRoi.x + pTile->m_srcRoi.width > pTile->m_srcImageSize.width + pTile->m_pParent->m_externalMemAcc.right)
                    pTile->m_srcRoi.width = pTile->m_srcImageSize.width - pTile->m_srcRoi.x + pTile->m_pParent->m_externalMemAcc.right;
                if(pTile->m_srcRoi.y + pTile->m_srcRoi.height > pTile->m_srcImageSize.height + pTile->m_pParent->m_externalMemAcc.bottom)
                    pTile->m_srcRoi.height = pTile->m_srcImageSize.height - pTile->m_srcRoi.y + pTile->m_pParent->m_externalMemAcc.bottom;
            }

            //hasScaling = 1;
        }
        else
        {
            pTile->m_srcRoi        = pTile->m_dstRoi;
            pTile->m_untaintSrcPos = pTile->m_untaintDstPos;

            if(!pTile->m_pChild) // Correction is required only for the first tile int the pipe since all other tiles use iwiCreateBorder function
            {
                IwiRoi correctRoi = pTile->m_dstRoi;

                // Shift source tile and start again
                if(owniTile_CorrectBordersOverlap(&correctRoi, NULL, &pTile->m_borderType, &pTile->m_borderSize, &pTile->m_borderSizeAcc, &pTile->m_srcImageSize))
                {
                    if(hasScaling)
                    {
                        tileRoi.x      += (IwSize)((correctRoi.x-pTile->m_srcRoi.x)*(scaleX + 0.5));
                        tileRoi.y      += (IwSize)((correctRoi.y-pTile->m_srcRoi.y)*(scaleY + 0.5));
                        tileRoi.width  += (IwSize)((correctRoi.width-pTile->m_srcRoi.width)*(scaleX + 0.5));
                        tileRoi.height += (IwSize)((correctRoi.height-pTile->m_srcRoi.height)*(scaleY + 0.5));
                    }
                    else
                    {
                        tileRoi.x      += correctRoi.x-pTile->m_srcRoi.x;
                        tileRoi.y      += correctRoi.y-pTile->m_srcRoi.y;
                        tileRoi.width  += correctRoi.width-pTile->m_srcRoi.width;
                        tileRoi.height += correctRoi.height-pTile->m_srcRoi.height;
                    }

                    hasScaling     = 0;
                    scaleX         = 1;
                    scaleY         = 1;
                    pTile = pRoot;
                    continue;
                }
            }
        }

        // Dst offset
        pTile->m_boundDstRoi = pTile->m_dstRoi;
        if(pTile->m_pParent)
        {
            // Intermediate dst buffer is aligned so that actual src image for parent will always be in the center
            if(pTile->m_untaintDstPos.x < 0 && !pTile->m_pParent->m_externalMem.left)
            {
                if(pTile->m_pParent->m_untaintDstPos.x <= 0)
                    pTile->m_boundDstRoi.x  = pTile->m_pParent->m_untaintDstPos.x - pTile->m_untaintDstPos.x;
                else
                    pTile->m_boundDstRoi.x  = -pTile->m_untaintDstPos.x;
            }
            else
                pTile->m_boundDstRoi.x = 0;

            if(pTile->m_untaintDstPos.y < 0 && !pTile->m_pParent->m_externalMem.top)
            {
                if(pTile->m_pParent->m_untaintDstPos.y <= 0)
                    pTile->m_boundDstRoi.y  = pTile->m_pParent->m_untaintDstPos.y - pTile->m_untaintDstPos.y;
                else
                    pTile->m_boundDstRoi.y  = -pTile->m_untaintDstPos.y;
            }
            else
                pTile->m_boundDstRoi.y = 0;
        }

        // Src offset
        pTile->m_boundSrcRoi = pTile->m_srcRoi;
        if(pTile->m_pChild)
        {
            // Intermediate src buffer is aligned to point inside border
            pTile->m_boundSrcRoi.x = pTile->m_borderSize.left;
            pTile->m_boundSrcRoi.y = pTile->m_borderSize.top;
        }
        else
        {
            // Shift source image to make InMem border a part of ROI
            if(pTile->m_pParent)
            {
                if(pTile->m_untaintSrcPos.x < 0)
                {
                    if(pTile->m_pParent->m_externalMemAcc.left >= -pTile->m_untaintSrcPos.x)
                        pTile->m_boundSrcRoi.x = pTile->m_untaintSrcPos.x;
                    else
                        pTile->m_boundSrcRoi.x = 0;
                }

                if(pTile->m_untaintSrcPos.y < 0)
                {
                    if(pTile->m_pParent->m_externalMemAcc.top >= -pTile->m_untaintSrcPos.y)
                        pTile->m_boundSrcRoi.y = pTile->m_untaintSrcPos.y;
                    else
                        pTile->m_boundSrcRoi.y = 0;
                }
            }
        }

        pTile = pTile->m_pChild;
    }

    return ippStsNoErr;
}

IW_DECL(IppStatus) iwiTilePipeline_GetDstBufferSize(const IwiTile *pTile, IwiSize *pDstSize)
{
    IppStatus status = owniTilePipeline_InitCheck(pTile);
    if(status < 0)
        return status;
    if(!pDstSize)
        return ippStsNullPtrErr;

    *pDstSize = pTile->m_dstBufferSize;

    return ippStsNoErr;
}

IW_DECL(IppStatus) iwiTilePipeline_GetChildSrcImageSize(const IwiTile *pTile, IwiSize srcOrigSize, IwiSize *pSrcFullSize)
{
    IppStatus status = owniTilePipeline_InitCheck(pTile);
    if(status < 0)
        return status;
    if(!pSrcFullSize)
        return ippStsNullPtrErr;

    pSrcFullSize->width  = srcOrigSize.width  + pTile->m_externalMemAcc.left + pTile->m_externalMemAcc.right;
    pSrcFullSize->height = srcOrigSize.height + pTile->m_externalMemAcc.top + pTile->m_externalMemAcc.bottom;

    return ippStsNoErr;
}

IW_DECL(IppStatus) iwiTilePipeline_GetChildDstImageSize(const IwiTile *pTile, IwiSize dstOrigSize, IwiSize *pDstFullSize)
{
    IppStatus status = owniTilePipeline_InitCheck(pTile);
    if(status < 0)
        return status;
    if(!pDstFullSize)
        return ippStsNullPtrErr;

    pDstFullSize->width  = dstOrigSize.width  + pTile->m_externalMemAcc.left + pTile->m_externalMemAcc.right;
    pDstFullSize->height = dstOrigSize.height + pTile->m_externalMemAcc.top + pTile->m_externalMemAcc.bottom;

    return ippStsNoErr;
}

IW_DECL(IppStatus) iwiTilePipeline_BuildBorder(const IwiTile *pTile, IwiImage *pSrcImage, IwiBorderType *pBorder, const Ipp64f *pBorderVal)
{
    IppStatus status;

    if(!pBorder)
        return ippStsNullPtrErr;

    status = owniTilePipeline_InitCheck(pTile);
    if(status < 0)
        return status;

    status = owniCheckImageWrite(pSrcImage);
    if(status < 0)
        return status;

    return owniTilePipeline_BuildBorder(pTile, pSrcImage, pBorder, pBorderVal, NULL);
}

IW_DECL(IppStatus) iwiTilePipeline_GetBoundedSrcRoi(const IwiTile *pTile, IwiRoi *pBoundedRoi)
{
    IppStatus status = owniTilePipeline_InitCheck(pTile);
    if(status < 0)
        return status;
    if(!pBoundedRoi)
        return ippStsNullPtrErr;

    *pBoundedRoi = pTile->m_boundSrcRoi;

    return ippStsNoErr;
}

IW_DECL(IppStatus) iwiTilePipeline_GetBoundedDstRoi(const IwiTile *pTile, IwiRoi *pBoundedRoi)
{
    IppStatus status = owniTilePipeline_InitCheck(pTile);
    if(status < 0)
        return status;
    if(!pBoundedRoi)
        return ippStsNullPtrErr;

    *pBoundedRoi = pTile->m_boundDstRoi;

    return ippStsNoErr;
}

IW_DECL(IppStatus) iwiTilePipeline_GetTileBorder(const IwiTile *pTile, IwiBorderType *pBorder)
{
    IppStatus status = owniTilePipeline_InitCheck(pTile);
    if(status < 0)
        return status;
    if(!pBorder)
        return ippStsNullPtrErr;

    owniTile_GetTileBorder(pBorder, &pTile->m_srcRoi, &pTile->m_borderSize, &pTile->m_srcImageSize);
    return ippStsNoErr;
}

IW_DECL(IppStatus) iwiTilePipeline_GetMinTileSize(const IwiTile *pTile, IwiSize *pMinTileSize)
{
    IppStatus       status;
    const IwiTile  *pRoot;

    if(!pMinTileSize)
        return ippStsNullPtrErr;
    pRoot  = owniTilePipeline_GetRoot(pTile);
    status = owniTilePipeline_InitCheck(pTile);
    if(status < 0)
        return status;

    // Check width
    pTile = pRoot;
    while(pTile)
    {
        if(!pTile->m_pChild)
        {
            *pMinTileSize   = iwiTile_GetMinTileSize(pTile->m_borderType, pTile->m_borderSize);
            if(pTile->m_pParent && (pMinTileSize->width > 1 || pMinTileSize->height > 1))
            {
                *pMinTileSize = iwiTile_GetMinTileSize(pTile->m_borderType, pTile->m_borderSizeAcc);
            }

            return ippStsNoErr;
        }

        pTile = pTile->m_pChild;
    }

    return ippStsNoErr;
}

