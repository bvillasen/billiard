// #include "vector2D.h"

class Circle {
private:
  Vector2D center;
  double radius;

public:
  __device__ Circle(){}
//     radius = 1.f;
//     type = 0;
//   }


  __device__ Circle( double r, Vector2D c  ) : radius(r) { center = c; }

//   __device__ ~Circle() { delete[] &center; delete[] &radius; delete[] &type; }


  __device__ double collideTime( Vector2D &pos, Vector2D &vel){
    Vector2D deltaPos = pos - center;
    double B = vel * deltaPos;
    double C = deltaPos*deltaPos - radius*radius;
    double d = B*B - C;
    if ( d<0 ) return -1; //particle doesnt collide with circle
    // double t1 = -B + sqrt(d);
    double t2 = -B - sqrt(d);
    return t2;
  }

  __device__ void bounce( Vector2D &collidePos, Vector2D &vel ){
    Vector2D normal;
    normal = collidePos - center;
    normal.normalize();
    double factor = -2*(vel*normal);
    normal = normal/factor;
    vel = vel + normal;
    vel.normalize();
  }

};
