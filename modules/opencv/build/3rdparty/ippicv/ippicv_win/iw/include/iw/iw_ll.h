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

#if !defined( __IPP_IWLL__ )
#define __IPP_IWLL__

extern "C" {
IW_DECL(IppStatus) llwiCopySplit(const void *pSrc, int srcStep, void* const pDstOrig[], int dstStep,
                                   IppiSize size, int typeSize, int channels, int partial);
IW_DECL(IppStatus) llwiCopyMerge(const void* const pSrc[], int srcStep, void *pDst, int dstStep,
                                   IppiSize size, int typeSize, int channels, int partial);
IW_DECL(IppStatus) llwiCopyChannel(const void *pSrc, int srcStep, int srcChannels, int srcChannel, void *pDst, int dstStep,
    int dstChannels, int dstChannel, IppiSize size, int typeSize);
IW_DECL(IppStatus) llwiCopyMask(const void *pSrc, int srcStep, void *pDst, int dstStep,
    IppiSize size, int typeSize, int channels, const Ipp8u *pMask, int maskStep);
IW_DECL(IppStatus) llwiSet(const double *pValue, void *pDst, int dstStep,
    IppiSize size, IppDataType dataType, int channels);
IW_DECL(IppStatus) llwiSetMask(const double *pValue, void *pDst, int dstStep,
    IppiSize size, IppDataType dataType, int channels, const Ipp8u *pMask, int maskStep);
IW_DECL(IppStatus) llwiCopyMakeBorder(const void *pSrc, IwSize srcStep, void *pDst, IwSize dstStep,
    IwiSize size, IppDataType dataType, int channels, IwiBorderSize borderSize, IwiBorderType border, const Ipp64f *pBorderVal);
}

#endif
