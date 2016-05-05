module circle_module
export Circle, collideTime, bounce

using vector2D_module

immutable Circle
  r::Float64
  center::Vector2D
end

function collideTime( c::Circle, pos::Vector2D, vel::Vector2D )
  deltaPos = pos - c.center
  B = dot( vel, deltaPos )
  C = norm2( deltaPos ) - c.r*c.r
  d = B*B - C
  if d<0
    return -1. #There is not a collision
  end
  # t1 = -B + sqrt(d)
  t2 = -B - sqrt(d)
  return t2
end

function bounce( c::Circle, pos::Vector2D, vel::Vector2D )
  normal = pos - c.center
  normalize!( normal )
  factor = -2 * dot( vel, normal )
  normal = factor * normal
  vel = vel + normal
  normalize!( vel )
  return vel
end

end
