//
//  methods.cpp
//  LungNoduleSynthesizer
//
//  Created by Weicheng on 2017-08-24.
//  Copyright © 2017 Weicheng Cao. All rights reserved.

//TODO: script accepts nodules.txt, ct.txt, voxel dimensions, position, nodule x, z
// init ct
// trace rays (baseline xray)

// insert nodule (make backup)
// calculate rays to trace
// trace rays
// write traced rays on baseline xray

#include "methods.hpp"

// // initialize static member of class Voxel
// Scale Voxel::scale = Scale();

// multithreaded parameters
int numberOfRaysFinished = 0;
int coordsNotInCT = 0;
// const int maxThreads = 1;
// int semaphore = 0;
// mutex mtx;

// projection dimensions
// const int projDim = 256;

void coordinateFromRay(SimulatedRay *ray, double t, Coordinate &new_coord){
  new_coord = Coordinate(ray->source->x + ray->lineVector->x * t, ray->source->y + ray->lineVector->y * t, ray->source->z + ray->lineVector->z * t);
}

bool inCT(vector<vector<vector<Voxel *>>> &ctVoxels, Coordinate coord){
  return coord.x >= 0 &&
         coord.x <= N_XY &&
         coord.y >= 0 &&
         coord.y <= N_XY &&
         coord.z >= 0 &&
         coord.z <= ctVoxels.size();
}

void insert_chunk(Chunk chunk, Chunk &backup, vector<vector<vector<Voxel *>>> &ctVoxels){

    vector<vector<vector<Voxel *>>> insert_array;
    backup.position = chunk.position;
    // make backup same dimensions as chunk
    backup.voxel_array.resize(chunk.voxel_array.size());
    for (int k = 0; k < chunk.voxel_array.size(); k++){
        backup.voxel_array[k].resize(chunk.voxel_array[k].size());
        for (int i = 0; i < chunk.voxel_array[k].size(); i++){
            backup.voxel_array[k][i].resize(chunk.voxel_array[k][i].size());
        }
    }

    int ct_x, ct_y, ct_z;
    for (int k = 0; k < chunk.voxel_array.size(); k++){
        for (int i = 0; i < chunk.voxel_array[k].size(); i++){
            for (int j = 0; j < chunk.voxel_array[k][i].size(); j++){
                ct_x = chunk.position.x + i;
                ct_y = chunk.position.y + j;
                ct_z = chunk.position.z + k;
                backup.voxel_array[k][i][j] = ctVoxels[ct_z][ct_x][ct_y];
                // only insert valid (nodule) voxels
                if (chunk.voxel_array[k][i][j]->attenuation != INVALID){
                    ctVoxels[ct_z][ct_x][ct_y] = chunk.voxel_array[k][i][j];
                }
            }
        }
    }
}

void create_nodule_chunk(NoduleSpecs nodule_spec, Chunk &nodule_chunk){

    int x_dim = nodule_spec.x_dim;
    int y_dim = nodule_spec.y_dim;
    int z_dim = nodule_spec.z_dim;

    nodule_chunk.position = Coordinate(floor(nodule_spec.position.x - x_dim/2),
                                floor(nodule_spec.position.y - y_dim/2),
                                floor(nodule_spec.position.z - z_dim/2));

    ifstream infile(nodule_spec.filename, ifstream::in);

    string line;
    string value;

    nodule_chunk.voxel_array.resize(z_dim);
    for (int z = 0; z < z_dim; z++){
        getline(infile, line);

        stringstream linestream(line);

        nodule_chunk.voxel_array[z].resize(x_dim);
        for (int x = 0; x < x_dim; x++){

            nodule_chunk.voxel_array[z][x].resize(y_dim);
            for (int y = 0; y < y_dim; y++){
                getline(linestream, value, ',');
                cout << value << endl;
                if (stol(value, NULL)){ // (whether or not this voxel is part of the nodule)
                    nodule_chunk.voxel_array[z][x][y] = new Voxel(nodule_spec.attenuation * stod(value, NULL));
                } else {
                    nodule_chunk.voxel_array[z][x][y] = new Voxel(INVALID);
                }
            }
        }
        cout << z << endl;
    }
    cout << "done" << endl;
}


void CreateCtVoxels(vector<vector<vector<Voxel *>>> &ctVoxels, string filename){

    cout << "Reading CT: " << filename << endl;

    string line;
    ifstream infile(filename, ifstream::in);

    // int x, y, z = 0;  /// indices in matrix
    double attenuation;

    vector<vector<Voxel *>> new_slice;
    new_slice.resize(N_XY);
    for (int i = 0; i< N_XY; i++){
        new_slice[i].resize(N_XY);
    }

    int z = 0;
    while(getline(infile, line)){
        ctVoxels.push_back(new_slice);

        stringstream linestream(line);
        string value;

        long long i = 0;

        // parse row/slice
        for (int x = 0; x < N_XY; x++){
            for (int y = 0; y < N_XY; y++){

                // NOTE: by experience, the attenuation factor is calculated using this equation
                getline(linestream, value, ',');
                attenuation = (1000+stod(value))*WATER_ATTENUATION/10000;
                ctVoxels[z][x][y] = new Voxel(attenuation);


                if (i % 100000 == 0){
                    cout << i << " voxels created" << endl;
                }
                i++;
            }
        }
        z++;
    }
    cout << z << " slices" << endl;
}

void WriteResultToFile(ProjectionPlane &plane, string filename)
{
    ofstream outfile;
    // NOTE: change to your absolute file path

    outfile.open(filename);


    for (unsigned int i = 0; i < IMAGE_DIM; i++){
        for (unsigned int j = 0; j < IMAGE_DIM; j++){
            outfile << plane.rays[i][j]->remainingRay << ",";
        }
        outfile << endl;
    }

    outfile.close();
}

vector<string> split(const string &s, char delim) {
    stringstream ss(s);
    string item;
    vector<string> tokens;
    while (getline(ss, item, delim)) {
        tokens.push_back(item);
    }
    return tokens;
}

int findInteger(string s) {

    char buf[10];
    int j = 0;
    for (int i = 0; i < s.length(); i++) {
        if (isdigit(s[i])){
            buf[j] = s[i];
            j++;
        }
    }
    return strtod(buf, NULL);
}



void parse_nodule_specs(string filename, vector<NoduleSpecs *> &specs_vect){

    ifstream infile(filename, ifstream::in);
    string line;
    getline(infile, line); // line 1
    getline(infile, line); // line 2
    while(getline(infile, line)){
        vector<string> line_vect = split(line, ',');

        NoduleSpecs *nodule_specs = new NoduleSpecs();

        nodule_specs->xray_number = findInteger(line_vect[0]);

        nodule_specs->position = Coordinate(
            stod(line_vect[3], NULL),
            stod(line_vect[2], NULL),
            stod(line_vect[1], NULL)
        );
        nodule_specs->z_dim = stod(line_vect[4], NULL);
        nodule_specs->y_dim = stod(line_vect[5], NULL);
        nodule_specs->x_dim = stod(line_vect[6], NULL);
        nodule_specs->attenuation = stod(line_vect[7], NULL);
        nodule_specs->filename = line_vect[9];

        specs_vect.push_back(nodule_specs);
    }
}


void calculate_rays_to_trace(Chunk chunk, ProjectionPlane &plane){

    for (unsigned int i = 0; i < IMAGE_DIM; i++){
        for (unsigned int j = 0; j < IMAGE_DIM; j++){

            double t = (chunk.position.y - plane.source->y)/(j * plane.bottomLeft.y  - plane.source->y);
            Coordinate point_in_plane;
            newCoordinate(plane.rays[i][j], t, point_in_plane);
            if (point_in_plane.x > chunk.position.x &&
                point_in_plane.x < chunk.position.x + chunk.voxel_array[0].size() &&
                point_in_plane.z > chunk.position.z &&
                point_in_plane.z < chunk.position.z + chunk.voxel_array.size()){
                plane.rays[i][j]->valid = true;
            }
        }
    }
}


void mark_all_rays(ProjectionPlane &plane){
    for (int i = 0; i < IMAGE_DIM; i++){
        for (int j = 0; j < IMAGE_DIM; j++){
            plane.rays[i][j]->valid = true;
        }
    }
}


// TODO: add bool arg as setting for baseline trace
void SynthesizeXRaysFromCT(NoduleSpecs nodule_spec, vector<vector<vector<Voxel *>>> ctVoxels,
    double voxel_xy_dim,
    double voxel_z_dim,
    Coordinate *source){

    cout << "Source at:" << "(" << source->x << "," << source->y << "," << source->z << ")" << endl;

    // TODO =====================
    Chunk nodule_chunk, backup_chunk;
    // create_nodule_chunk(nodule_spec, nodule_chunk);
    // insert_chunk(nodule_chunk, backup_chunk, ctVoxels);
    cout << "====" << endl;
    ProjectionPlane plane = CreateProjectionPlane(source, ctVoxels);

    cout << setw(15) << "Top left: (" << plane.topLeft.x << ", " << plane.topLeft.y << ", " << plane.topLeft.z << ")" << endl;
    cout << setw(15) << "Top Right: (" << plane.topRight.x << ", " << plane.topRight.y << ", " << plane.topRight.z << ")" << endl;
    cout << setw(15) << "Bottom left: (" << plane.bottomLeft.x << ", " << plane.bottomLeft.y << ", " << plane.bottomLeft.z << ")" << endl;
    cout << setw(15) << "Bottom Right: (" << plane.bottomRight.x << ", " << plane.bottomRight.y << ", " << plane.bottomRight.z << ")" << endl;
    cout << endl;

    // partition projection plane
    create_rays(plane);

    // vector<thread> workingThreads;

    calculate_rays_to_trace(nodule_chunk, plane);

    mark_all_rays(plane);


    for (unsigned int i=0; i < IMAGE_DIM; i++)
    {
        for (unsigned int j=0; j < IMAGE_DIM; j++)
        {
            if (plane.rays[i][j]->valid){
                CalculateRemainingIntensity(plane.rays[i][j], ctVoxels, voxel_xy_dim, voxel_z_dim);
            }
            // mtx.lock();
            // if (semaphore < maxThreads)
            // {
            //     workingThreads.push_back(thread(CalculateRemainingIntensity, rays[i][j], ref(ctVoxels)));
            //     semaphore++;
            // }
            // else
            // {
            //     j--;
            // }
            // mtx.unlock();

        }
    }

//    join all threads
    // for (unsigned int i=0; i<workingThreads.size(); i++)
    // {
    //     workingThreads[i].join();
    // }
    //
    // cout << endl << "Threads joined: calculating remaining intensity finished" << endl;
    // workingThreads.clear();
    cout << coordsNotInCT << " coordinates not in CT" << endl;

    string filename =  "textXRays/sp_" + to_string((int) source->x) + "_" + to_string((int) source->y) + "_" + to_string((int) source->z) + "_n" + to_string(findInteger(nodule_spec.filename)) + ".txt";
    cout << filename << endl;

    WriteResultToFile(plane, filename);

    // insert_chunk(backup_chunk, nodule_chunk, ctVoxels);

    return;
}



void SynthesizeXRays(string ct_filename, string nodule_specs_filename, double voxel_xy_dim, double voxel_z_dim)
{
    vector<NoduleSpecs *>specs_vect;
    parse_nodule_specs(nodule_specs_filename, specs_vect);


    vector<vector<vector<Voxel *>>> ctVoxels;
    CreateCtVoxels(ctVoxels, ct_filename);


    vector<Coordinate *> source_pts = {
        new Coordinate(0, POINT_SOURCE_Y, ctVoxels.size()/2),
        new Coordinate(512, POINT_SOURCE_Y, ctVoxels.size()/2),
        new Coordinate(256, POINT_SOURCE_Y, ctVoxels.size()),
        new Coordinate(256, POINT_SOURCE_Y, 0),
        new Coordinate(256, POINT_SOURCE_Y + 256, ctVoxels.size()/2),
        new Coordinate(256, POINT_SOURCE_Y - 256, ctVoxels.size()/2),
        new Coordinate(256, POINT_SOURCE_Y, ctVoxels.size()/2)};

        SynthesizeXRaysFromCT(*specs_vect[0], ctVoxels, voxel_xy_dim, voxel_z_dim, source_pts[0]);
        SynthesizeXRaysFromCT(*specs_vect[0], ctVoxels, voxel_xy_dim, voxel_z_dim, source_pts[1]);
        SynthesizeXRaysFromCT(*specs_vect[0], ctVoxels, voxel_xy_dim, voxel_z_dim, source_pts[2]);
        SynthesizeXRaysFromCT(*specs_vect[0], ctVoxels, voxel_xy_dim, voxel_z_dim, source_pts[3]);
        SynthesizeXRaysFromCT(*specs_vect[0], ctVoxels, voxel_xy_dim, voxel_z_dim, source_pts[4]);
        SynthesizeXRaysFromCT(*specs_vect[0], ctVoxels, voxel_xy_dim, voxel_z_dim, source_pts[5]);
        SynthesizeXRaysFromCT(*specs_vect[0], ctVoxels, voxel_xy_dim, voxel_z_dim, source_pts[6]);


}

double findT(SimulatedRay *ray, int dim, double n){

    double delta = 0.0, source;
    if (dim == XDIM){
        delta = ray->lineVector->x;
        source = ray->source->x;
    } else if (dim == YDIM){
        delta = ray->lineVector->y;
        source = ray->source->y;
    } else if (dim == ZDIM){
        delta = ray->lineVector->z;
        source = ray->source->z;
    }
    if (delta == 0){
        return INVALID;
    }
    return (n - source)/delta;

}

void findEntrance(SimulatedRay *ray, vector<vector<vector<Voxel *>>> &ctVoxels, Coordinate &in){

    Coordinate edgeCoordinates[6];
    coordinateFromRay(ray, findT(ray, XDIM, 0), edgeCoordinates[0]);
    coordinateFromRay(ray, findT(ray, XDIM, N_XY), edgeCoordinates[1]);
    coordinateFromRay(ray, findT(ray, YDIM, 0), edgeCoordinates[2]);
    coordinateFromRay(ray, findT(ray, YDIM, N_XY), edgeCoordinates[3]);
    coordinateFromRay(ray, findT(ray, ZDIM, 0), edgeCoordinates[4]);
    coordinateFromRay(ray, findT(ray, ZDIM, ctVoxels.size()), edgeCoordinates[5]);

    vector<Coordinate> inOut;
    for (int i = 0; i <= 5; i++){
        if (inCT(ctVoxels, edgeCoordinates[i])){

            inOut.push_back(edgeCoordinates[i]);
        }
    }
    in = inOut[0];
    if (distance(inOut[0], *ray->source) > distance(inOut[1], *ray->source)){
        in = inOut[1];
    }
}


void newCoordinate(SimulatedRay *ray, double t, Coordinate &coordinate){
    coordinate = Coordinate(ray->source->x + t * ray->lineVector->x,
                        ray->source->y + t * ray->lineVector->y,
                        ray->source->z + t * ray->lineVector->z);
    if (floor(coordinate.y) == -1){
        coordinate.y = 0;
    }
}

double getIntensityLoss(Coordinate c1, Coordinate c2, vector<vector<vector<Voxel *>>> &ctVoxels,
    double voxel_xy_dim, double voxel_z_dim){

    double x, y, z, distance;
    x = floor(min(c1.x, c2.x));
    y = floor(min(c1.y, c2.y));
    z = floor(min(c1.z, c2.z));

    if (x == N_XY || y == N_XY || z == ctVoxels.size() || x < 0 || y == -1 || z < 0){
        cout << "Coordinates out of CT " << x << "," << y << "," << z << endl;
        coordsNotInCT++;
        return 0;
    }

    distance = sqrt(pow((c2.x - c1.x) * voxel_xy_dim, 2) +
    pow((c2.y - c1.y) * voxel_xy_dim, 2) +
    pow((c2.z - c1.z) * voxel_z_dim, 2));

    // cout << x << "," << y << "," << z << endl;

    return ctVoxels[z][x][y]->attenuation * distance;
}

// returns the lost intensity and updates free variable t of the
// next closest integer dimension to prev
double traverseVoxel(Coordinate *prev, double *t, int dim, Coordinate *coordArray,
                     SimulatedRay *ray, vector<vector<vector<Voxel *>>> &ctVoxels,
                     double voxel_xy_dim, double voxel_z_dim){

   double localIntensityLoss = getIntensityLoss(*prev, coordArray[dim], ctVoxels, voxel_xy_dim, voxel_z_dim);

   double nextInt;
   if (dim == XDIM){
       nextInt = coordArray[dim].x + ray->xSign;
   } else if (dim == YDIM){
       nextInt = coordArray[dim].y + ray->ySign;
   } else {
       nextInt = coordArray[dim].z + ray->zSign;
   }

   *prev = coordArray[dim];
   *t = findT(ray, dim, nextInt);
   newCoordinate(ray, *t, coordArray[dim]);

   return localIntensityLoss;
}


bool continueTraversal(Coordinate *coordArray, vector<vector<vector<Voxel *>>> &ctVoxels){
  return inCT(ctVoxels, coordArray[XDIM]) || inCT(ctVoxels, coordArray[YDIM]) || inCT(ctVoxels, coordArray[ZDIM]);
}


void CalculateRemainingIntensity(SimulatedRay *ray, vector<vector<vector<Voxel *>>> &ctVoxels,
    double voxel_xy_dim,
    double voxel_z_dim){
    Coordinate in;
    findEntrance(ray, ctVoxels, in);

    double xStart, yStart, zStart, tx, ty, tz;
    // dimension delimiting the entrance of ray is an integer
    // find initial integer values
    xStart = ray->xSign == 1 ? ceil(in.x) : floor(in.x);
    yStart = ray->ySign == 1 ? ceil(in.y) : floor(in.y);
    zStart = ray->zSign == 1 ? ceil(in.z) : floor(in.z);

    Coordinate prev = in;

    tx = findT(ray, XDIM, xStart);
    ty = findT(ray, YDIM, yStart);
    tz = findT(ray, ZDIM, zStart);


    Coordinate coordArray[3];
    newCoordinate(ray, tx, coordArray[XDIM]);
    newCoordinate(ray, ty, coordArray[YDIM]);
    newCoordinate(ray, tz, coordArray[ZDIM]);


    double power = 0;
    int i = 0;
    while (continueTraversal(coordArray, ctVoxels)){
        i++;
        if (tx <= ty && tx <= tz){
        power += traverseVoxel(&prev, &tx, XDIM, coordArray, ray, ctVoxels, voxel_xy_dim, voxel_z_dim);
        } else if (ty <= tx && ty <=tz){
        power += traverseVoxel(&prev, &ty, YDIM, coordArray, ray, ctVoxels, voxel_xy_dim, voxel_z_dim);
        } else {
        power += traverseVoxel(&prev, &tz, ZDIM, coordArray, ray, ctVoxels, voxel_xy_dim, voxel_z_dim);
        }
    }
    //
    // mtx.lock();
    // cout << power << endl;
    ray->remainingRay *= exp(-1 *power*WATER_ATTENUATION);
    // cout << ray->remainingRay << endl;
    if (numberOfRaysFinished % 1000 == 0)
    {
        cout << numberOfRaysFinished << ":" << ray->remainingRay << endl;
    }
    // semaphore--;
    numberOfRaysFinished++;
    // mtx.unlock();
    ray->valid = false;

}


/**
 *  Note: planeCenter and source must have the same x and z coordinates.
 */
ProjectionPlane CreateProjectionPlane(Coordinate *source, vector<vector<vector<Voxel *>>> &ctMatrix)
{
    ProjectionPlane plane (source);

//    useful variables
    double zlen = double(ctMatrix.size());          /// height

    double sourceToVoxels = -source->y;
    double proportion = (N_XY - source->y) / sourceToVoxels;


    double extLeft     = source->x < 0       ? 0 : source-> x - source->x * proportion;
    double extRight    = source->x > N_XY    ? 0 : source->x + (N_XY - source->x) * proportion;
    double extTop      = source->z > zlen    ? 0 : source->z + (zlen - source->z) * proportion;
    double extBottom   = source->z < 0       ? 0 : source->z - source->z * proportion;

    plane.topLeft = Coordinate(extLeft, N_XY, extTop);
    plane.topRight = Coordinate(extRight, N_XY, extTop);
    plane.bottomLeft = Coordinate(extLeft, N_XY, extBottom);
    plane.bottomRight = Coordinate(extRight, N_XY, extBottom);

    return plane;
}


void create_rays(ProjectionPlane& plane){
    double leftMost = plane.topLeft.x;
    double topMost = plane.topLeft.z;
    double pixel_height = fabs(plane.topLeft.z - plane.bottomLeft.z) / IMAGE_DIM; // should be called height
    double pixel_width = fabs(plane.topRight.x - plane.topLeft.x) / IMAGE_DIM; // should be called width

    plane.rays.resize(IMAGE_DIM);
    for (int i = 0; i < IMAGE_DIM; i++){
        plane.rays[i].resize(IMAGE_DIM);
    }

    Coordinate *projection;
    for (int i = 0; i < IMAGE_DIM; i++){
        for (int j = 0; j < IMAGE_DIM; j++){
            projection = new Coordinate(leftMost + (i + 0.5) * pixel_width,
                                       plane.bottomLeft.y,
                                       topMost - (j + 0.5) * pixel_height);
            plane.rays[i][j] = new SimulatedRay(plane.source, projection);
        }
    }
}
