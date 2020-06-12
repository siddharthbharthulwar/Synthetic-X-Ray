//
//  main.cpp
//  LungNoduleSynthesizer
//
//  Created by Weicheng on 2017-08-17.
//  Copyright © 2017 Weicheng Cao. All rights reserved.
//


/// 1. voxels are static so other objects will use pointer to reference them
/// 2. 3D array will start from (0, 0, 0) and height is 256


#include "methods.hpp"

// args are 1. ct.txt
        //  2. nodule_specs.txt
        //  3. voxel dimensions

// nodule_specs.txt is formatted:
    // 1 line
    // 1 line: xray0 (baseline xray)
    // output filename,
    // z, x, y position,
    // z, x, y dimension,
    // attenuation,
    // nodule.txt



int main(int argc, const char * argv[]) {

    if (argc != 5){
        cout << "USAGE: ct_filename nodule_specs.txt voxel_xy_dim voxel_z_dim" << endl;
        return 1;
    }


    string ct_filename = argv[1];

    string nodule_specs_filename = argv[2];

    double voxel_xy_dim = strtod(argv[3], NULL);

    double voxel_z_dim = strtod(argv[4], NULL);



    SynthesizeXRays(ct_filename, nodule_specs_filename, voxel_xy_dim, voxel_z_dim);

    return 0;
}
