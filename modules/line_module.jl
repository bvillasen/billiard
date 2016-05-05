module line_module
export Line, collideTime, bounce, changePosPeriodic

using vector2D_module

immutable Line
  center::Vector2D
  normal::Vector2D
end

function collideTime( l::Line, pos::Vector2D, vel::Vector2D )
  if dot(vel,l.normal) <= 0
    return -1
  end
  deltaPos = l.center - pos
  distNormal = dot( deltaPos, l.normal )
  velNormal = dot( vel, l.normal )
  return distNormal/velNormal
end

function bounce( l::Line, pos::Vector2D, vel::Vector2D )
  factor = -2 * dot( vel, l.normal )
  deltaVel = factor*l.normal
  vel = vel + deltaVel
  normalize!( vel )
  return vel
end

function changePosPeriodic( l::Line, pos::Vector2D, region::Vector2D   )
  pos = pos - l.normal
  region = region + l.normal
  return pos, region
end

end
