//
//  SimulatedRay.cpp
//  LungNoduleSynthesizer
//
//  Created by Weicheng on 2017-08-19.
//  Copyright © 2017 Weicheng Cao. All rights reserved.
//

#include <cmath>

#include "Constants.h"
#include "SimulatedRay.hpp"

/// assign pointer and coordinate
SimulatedRay::SimulatedRay(Coordinate *_source, Coordinate *_projection)
{
    source = _source;
    projection = _projection;
    lineVector = new Coordinate(_projection->x - _source->x,
                                _projection->y - _source->y,
                                _projection->z - _source->z);

    if (lineVector->x == 0 || lineVector->z == 0){
        cout << "perfectly perpendicular ray" << endl;
    }
    remainingRay = 1;
    valid = false; // must be marked to be traced

    xSign = lineVector->x > 0 ? 1 : -1;
    ySign = lineVector->y > 0 ? 1 : -1;
    zSign = lineVector->z > 0 ? 1 : -1;


}

// void mark_ray(){
//     valid = true;
// }
//
// void unmark_ray(){
//     valid = false;
// }

/// Azimuth is a polar angle in the x-y plane, with positive angles indicating counterclockwise rotation of the source.
// double SimulatedRay::GetTrueAzimuth()
// {
//     /// 1. calculate the acute angle of azimuth and disregard the relative positions of dest and start
//     double deltaY = projectedVoxel.center->y - source->y;
//     double deltaX = projectedVoxel.center->x - source->x;
//
//     /// this assumes length is along x-axis, width is along y-axis and slices are spaced along z-axis
//     double acuteAngle = atan(fabs(deltaY * VOXEL_WIDTH) / fabs(deltaX * VOXEL_LENGTH)) * 180 / PI;
//     double result = acuteAngle;
//
//     /// 2. adjust angle by taking the positions into account
//     /// if source's y is less than destination point's
//     if (deltaY > 0)
//     {
//         result += 90;
//     }
//
//     /// if source's x is less than destination point's
//     if (deltaX < 0)
//     {
//         result = result + 180 - acuteAngle;
//     }
//
//     return result;
// }   /// verified
//
//
// /// Elevation is the angle above (positive angle) or below (negative angle) the x-y plane.
// double SimulatedRay::GetTrueElevation()
// {
//     /// 1. calculate the acute angle of elevation and disregard the relative positions of dest and start
//     double deltaZ = projectedVoxel.center->z - source->z;
//     double deltaX = projectedVoxel.center->x - source->x;
//     double deltaY = projectedVoxel.center->y - source->y;
//
//     /// this assumes length is along x-axis, width is along y-axis and slices are spaced along z-axis
//     double acuteAngle = atan(fabs(deltaZ*VOXEL_HEIGHT) / fabs(sqrt(pow(deltaX*VOXEL_LENGTH,2) + pow(deltaY*VOXEL_WIDTH, 2)))) * 180 / PI;
//
//     /// 2. adjust angle by taking the positions into account
//     /// if source's z is less than destination point's, elevation is negative
//     return deltaZ < 0 ? -acuteAngle : acuteAngle;
// }   /// verified
