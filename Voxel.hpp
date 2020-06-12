//
//  Voxel.hpp
//  LungNoduleSynthesizer
//
//  Created by Weicheng on 2017-08-19.
//  Copyright © 2017 Weicheng Cao. All rights reserved.
//

#ifndef Voxel_hpp
#define Voxel_hpp

#include "Coordinate.hpp"

class Voxel
{

public:
    /// properties
    Coordinate *center;
    double attenuation;

    /// constructors
    Voxel(double _attenuation){
        attenuation = _attenuation;
    }
    Voxel(double _attenuation, double _x, double _y, double _z){
      attenuation = _attenuation;
      center = new Coordinate(_x, _y, _z);
    }
    double GetIntersectLineLen();   /// override abstract

};

#endif /* Voxel_hpp */
