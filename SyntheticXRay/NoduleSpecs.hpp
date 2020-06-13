
#include "Coordinate.hpp"
#include <string>;

class NoduleSpecs
{
public:
    Coordinate position;
    int xray_number;
    double x_dim;
    double y_dim;
    double z_dim;
    double attenuation;
    string filename;
};
