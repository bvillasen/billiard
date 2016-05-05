#include <iostream>
#include <math.h>
using namespace std;


class Vector2D{
public:
  double x;
  double y;
  // Constructor
  __host__ __device__ Vector2D( double x0=0.f, double y0=0.f ) : x(x0), y(y0) {}
  // Destructor
//   __host__ __device__ ~Vector2D(){ delete[] &x; delete[] &y; }


  __host__ __device__ double norm( void ) { return sqrt( x*x + y*y ); };

  __host__ __device__ double norm2( void ) { return x*x + y*y ; };

  __host__ __device__ void normalize(){
    double mag = norm();
    x /= mag;
    y /= mag;
  }

  __host__ __device__ Vector2D operator+( Vector2D &v ){
    return Vector2D( x+v.x, y+v.y );
  }

  __host__ __device__ Vector2D operator-( Vector2D &v ){
    return Vector2D( x-v.x, y-v.y );
  }

  __host__ __device__ double operator*( Vector2D &v ){
    return x*v.x + y*v.y;
  }

  __host__ __device__ Vector2D operator/( double a ){
    return Vector2D( a*x, a*y );
  }

  __host__ __device__ void redefine( double x0, double y0 ){
    x = x0;
    y = y0;
  }




};
