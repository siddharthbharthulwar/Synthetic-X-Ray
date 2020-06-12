//
//  SimulatedRay.hpp
//  LungNoduleSynthesizer
//
//  Created by Weicheng on 2017-08-19.
//  Copyright © 2017 Weicheng Cao. All rights reserved.
//

#ifndef SimulatedRay_hpp
#define SimulatedRay_hpp

#include <vector>
#include "Voxel.hpp"
#include <stdio.h>
#include <iostream>

using namespace std;

class SimulatedRay
{

public:
    /// properties
    Coordinate *source;          /// ray source
    Coordinate *projection;          /// pointer to the projected voxel
    Coordinate* lineVector;         /// vector from viewpoint to projected voxel: line = viewpoint.coordinate + lineVector
    double remainingRay, xSign, ySign, zSign;
    bool valid;

    /// constructors & destructors
    SimulatedRay() { remainingRay = INVALID; };
    SimulatedRay(Coordinate *, Coordinate *);

    // ~SimulatedRay()
    // {
    //     cout << "aaaaaaaaaaa" << endl;
    //     delete lineVector;
    // };|

    /// methods
    double GetTrueAzimuth();    /// taken scale into account
    double GetTrueElevation();  /// the angle between ground to a vector of x, y, z

};

#endif /* SimulatedRay_hpp */
