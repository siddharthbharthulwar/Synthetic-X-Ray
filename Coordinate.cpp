//
//  Coordinate.cpp
//  LungNoduleSynthesizer
//
//  Created by Weicheng on 2017-08-19.
//  Copyright © 2017 Weicheng Cao. All rights reserved.
//

#include "Coordinate.hpp"
#include <math.h>
// Coordinate::Coordinate(SimulatedRay ray, double t): x(ray.viewpoint->x + ray.linevector.x * t), y(ray.viewpoint->y + ray.linevector.y * t), z(ray.viewpoint->z + ray.linevector.z * t) {};
// Coordinate::Coordinate(SimulatedRay &ray, double t{
//     x = ray->viewpoint->x + ray->linevector.x * t;
//     y = ray->viewpoint->y + ray->linevector.y * t;
//     z = ray->viewpoint->z + ray->linevector.z * t;
//
// }
double distance(Coordinate c1, Coordinate c2){
  return sqrt(pow(c1.x - c2.x, 2) + pow(c1.y - c2.y, 2) + pow(c1.z - c2.z, 2));
}
