//
//  ProjectionPlane.hpp
//  LungNoduleSynthesizer
//
//  Created by Weicheng on 2017-08-19.
//  Copyright © 2017 Weicheng Cao. All rights reserved.
//

#ifndef ProjectionPlane_hpp
#define ProjectionPlane_hpp

#include <stdio.h>
#include <unordered_map>
#include <vector>
#include "Coordinate.hpp"
#include "SimulatedRay.hpp"
#include "Voxel.hpp"
#include "Pixel.hpp"

using namespace std;

class ProjectionPlane
{

public:
    /// properties
    Coordinate *source;                   /// where xray point source is
    Coordinate topLeft;
    Coordinate topRight;
    Coordinate bottomLeft;
    Coordinate bottomRight;                 /// define the plane
    vector<vector<SimulatedRay *>> rays;      // rays[x][y] -> xth row from the top, yth

    /// constructors
    ProjectionPlane(Coordinate *_source): source(_source) {};

};

#endif /* ProjectionPlane_hpp */
