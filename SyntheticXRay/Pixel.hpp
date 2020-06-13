#include "Coordinate.hpp"

class Pixel
{
public:
    Coordinate* center;
    double intensity;

    Pixel(double x, double y, double z) {
        center = new Coordinate(x, y, z);
    }
};
