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

#if !defined( __IPP_IW_CONFIG__ )
#define __IPP_IW_CONFIG__

#ifndef IW_BUILD
#error this is a private header
#endif

/*
    These switches are used during IW library compilation to customize the library code and decrease memory footprint
    of the library.
*/

/* /////////////////////////////////////////////////////////////////////////////
//                   Library Features
///////////////////////////////////////////////////////////////////////////// */

#ifndef IW_ENABLE_THREADING_LAYER
#define IW_ENABLE_THREADING_LAYER 0 // Enables Intel IPP Threading Layer calls inside IW if possible (requires OpenMP support)
#endif                              // Parallel version of functions will be used if:
                                    // 1. There is a parallel implementation for a particular function (see function description in the header)
                                    // 2. If iwGetThreadsNum() function result is greater than 1 before functions call or spec initialization call
                                    // Note: tiling cannot be used with internal threading. IwiTile parameter will be ignored if conditions above are true before function call
                                    // To disable threading on run time: call iwSetThreadsNum(1) before a function call

#ifndef IW_ENABLE_TLS
#define IW_ENABLE_TLS 1             // Enables use of Thread Local Storage. This adds dependency on POSIX Threads on POSIX systems.
#endif

/* /////////////////////////////////////////////////////////////////////////////
//                   Data types
///////////////////////////////////////////////////////////////////////////// */

// These switches can remove Intel IPP functions calls with some data types to reduce memory footprint.
// Functions which operates with several types and channels will be enabled if at least one of parameters has enabled type
// Note that some functionality can become completely disabled if some of these defines are switched off
#ifndef IW_ENABLE_DATA_TYPE_8U
#define IW_ENABLE_DATA_TYPE_8U  1
#endif
#ifndef IW_ENABLE_DATA_TYPE_8S
#define IW_ENABLE_DATA_TYPE_8S  1
#endif
#ifndef IW_ENABLE_DATA_TYPE_16U
#define IW_ENABLE_DATA_TYPE_16U 1
#endif
#ifndef IW_ENABLE_DATA_TYPE_16S
#define IW_ENABLE_DATA_TYPE_16S 1
#endif
#ifndef IW_ENABLE_DATA_TYPE_32U
#define IW_ENABLE_DATA_TYPE_32U 0
#endif
#ifndef IW_ENABLE_DATA_TYPE_32S
#define IW_ENABLE_DATA_TYPE_32S 1
#endif
#ifndef IW_ENABLE_DATA_TYPE_32F
#define IW_ENABLE_DATA_TYPE_32F 1
#endif
#ifndef IW_ENABLE_DATA_TYPE_64U
#define IW_ENABLE_DATA_TYPE_64U 0
#endif
#ifndef IW_ENABLE_DATA_TYPE_64S
#define IW_ENABLE_DATA_TYPE_64S 0
#endif
#ifndef IW_ENABLE_DATA_TYPE_64F
#define IW_ENABLE_DATA_TYPE_64F 1
#endif

/* /////////////////////////////////////////////////////////////////////////////
//                   Channels
///////////////////////////////////////////////////////////////////////////// */

#ifndef IW_ENABLE_CHANNELS_C1
#define IW_ENABLE_CHANNELS_C1  1
#endif
#ifndef IW_ENABLE_CHANNELS_C3
#define IW_ENABLE_CHANNELS_C3  1
#endif
#ifndef IW_ENABLE_CHANNELS_C4
#define IW_ENABLE_CHANNELS_C4  1
#endif
#ifndef IW_ENABLE_CHANNELS_AC4
#define IW_ENABLE_CHANNELS_AC4 0
#endif

/* /////////////////////////////////////////////////////////////////////////////
//                   Functions Features
///////////////////////////////////////////////////////////////////////////// */

// iwiResize
#ifndef IW_ENABLE_iwiResize_Nearest
#define IW_ENABLE_iwiResize_Nearest         0
#endif
#ifndef IW_ENABLE_iwiResize_Super
#define IW_ENABLE_iwiResize_Super           1
#endif
#ifndef IW_ENABLE_iwiResize_Linear
#define IW_ENABLE_iwiResize_Linear          1
#endif
#ifndef IW_ENABLE_iwiResize_LinearAA
#define IW_ENABLE_iwiResize_LinearAA        0
#endif
#ifndef IW_ENABLE_iwiResize_Cubic
#define IW_ENABLE_iwiResize_Cubic           1
#endif
#ifndef IW_ENABLE_iwiResize_CubicAA
#define IW_ENABLE_iwiResize_CubicAA         0
#endif
#ifndef IW_ENABLE_iwiResize_Lanczos
#define IW_ENABLE_iwiResize_Lanczos         1
#endif
#ifndef IW_ENABLE_iwiResize_LanczosAA
#define IW_ENABLE_iwiResize_LanczosAA       0
#endif

#endif
