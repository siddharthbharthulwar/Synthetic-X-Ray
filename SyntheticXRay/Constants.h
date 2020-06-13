//
//  Constants.h
//  LungNoduleSynthesizer
//
//  Created by Weicheng on 2017-08-19.
//  Copyright © 2017 Weicheng Cao. All rights reserved.
//
/// 0.32 x, 0.34 z
#ifndef Constants_h
#define Constants_h

#define VOXEL_XY                0.0667968988419   /// y in cm
#define VOXEL_Z                 0.125             /// x   NOTE: confirm that `slice spacing` is x
#define WATER_ATTENUATION       0.1707              /// in cm^2/g
//

#define POINT_SOURCE_Y          -20000
// #define POINT_SOURCE_Y          -14970.7548904      /// 1 meter == 100 cms (100 / 0.0667968988419)

#define VOXEL_SPACING           0.5                 /// ct voxels matrix coordinate spacing

#define PI                      3.14159265

#define IMAGE_DIM               256

#define NONEXIST                -1
#define INVALID                 999999

#define N_XY                    512 // number of voxels in x/y dimensions

#define XDIM                    0
#define YDIM                    1
#define ZDIM                    2

#endif /* Constants_h */

/*
 Front:
 z  ^
    |
    |
    |_____> x

 Right Side:
 z  ^
    |
    |
    |_____> y

 Top:
 y  ^
    |
    |
    |_____> x
 */
