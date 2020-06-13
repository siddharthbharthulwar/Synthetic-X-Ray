//
//  methods.hpp
//  LungNoduleSynthesizer
//
//  Created by Weicheng on 2017-08-24.
//  Copyright © 2017 Weicheng Cao. All rights reserved.
//

#ifndef methods_hpp
#define methods_hpp

#include <mutex>
#include <algorithm>    /// to find common elements in 2 vectors
#include <cmath>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <thread>
#include "Constants.h"
#include "ProjectionPlane.hpp"
#include "Voxel.hpp"
#include "SimulatedRay.hpp"
#include "NoduleSpecs.hpp"
#include "Chunk.hpp"

void newCoordinate(SimulatedRay*, double, Coordinate&);

void SynthesizeXRays(string ct_filename, string nodule_specs_filename, double voxel_xy_dim, double voxel_z_dim);

/// top level wrapper function
void SynthesizeXRaysFromCT(NoduleSpecs nodule_spec, vector<vector<vector<Voxel*>>> ctVoxels, Coordinate*);

void CreateCtVoxels(vector<vector<vector<Voxel*>>>& ctVoxels);
void WriteResultToFile(ProjectionPlane&);

void create_rays(ProjectionPlane&);


/// create a projection plane and set extensions
ProjectionPlane CreateProjectionPlane(Coordinate*, vector<vector<vector<Voxel*>>>&);

/// partition the projection plane into a row * column 2D matrix with small voxels with height equal to 0
void PartitionProjectionPlane(ProjectionPlane& plane, double row, double column);

void GetVoxelsOnRayHelper(vector<vector<vector<Voxel*>>>& ctVoxels, int view, int index, double intercept, double slope, vector<Voxel*>& result, double rightBound, double leftBound);

/// wrapper function
vector<Voxel*> GetVoxelsOnRay(vector<vector<vector<Voxel*>>>& ctVoxels, SimulatedRay* ray);

/// calculate remaining intensity from ray
/// 1. ray: ray to be calculated upon
void CalculateRemainingIntensity(SimulatedRay*, vector<vector<vector<Voxel*>>>&, double, double);

double CalculateLinearFunction(double k, double b, double x);
double CalculateXFromPoint(double k, double b, double y);

void GetBoundingVoxelsIndices(double totalLen, double projectionIndex, double sourceIndex, double& higherBound, double& lowerBound);

#endif /* methods_hpp */
