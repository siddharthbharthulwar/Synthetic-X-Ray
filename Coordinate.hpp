//
//  Coordinate.hpp
//  LungNoduleSynthesizer
//
//  Created by Weicheng on 2017-08-19.
//  Copyright © 2017 Weicheng Cao. All rights reserved.
//

#ifndef Coordinate_hpp
#define Coordinate_hpp

#include "Constants.h"
// #include "SimulatedRay.hpp"

class Coordinate    /// this follows [x][y][z] convention
{

public:
    /// properties
    double x;
    double y;
    double z;   /// cartesian coordinate system

    /// constructors
    Coordinate(): x(INVALID), y(INVALID), z(INVALID) {};
    Coordinate(double _x, double _y, double _z): x(_x), y(_y), z(_z) {};
    Coordinate(const Coordinate &src): x(src.x), y(src.y), z(src.z) {};    /// copy constructor must pass its first argument by reference


};

double distance(Coordinate, Coordinate);

#endif /* Coordinate_hpp */
