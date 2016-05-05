// #include "vector2D.h"


class Line {
private:
  Vector2D center;
  Vector2D normal;

public:
  __device__ Line(){}
//     normal.redefine(0.f,1.f);
//     type = 0;
//   }

  __device__ Line( Vector2D c, Vector2D n ){
    center = c;
    normal = n;
    normal.normalize();
  }

//   __device__ ~Line() { delete[] &center; delete[] &normal; delete[] &type; }

  __device__ double collideTime( Vector2D &pos, Vector2D &vel ){
    if (vel*normal<=0) return -1;
    Vector2D deltaPos = center - pos;
    double distNormal = deltaPos * normal;
    double velNormal = vel * normal;
    return distNormal/velNormal;
  }

  __device__ void bounce( Vector2D &collidePos, Vector2D &vel){ //Real wall
      double factor = -2*( vel*normal );
      Vector2D deltaVel = normal/factor;
      vel = vel + deltaVel;
      vel.normalize();
  }
  __device__ void changePosPeriodic( Vector2D &collidePos, Vector2D &region ){ //Periodic wall
    collidePos = collidePos - normal;
    region = region + normal;
  }



};
