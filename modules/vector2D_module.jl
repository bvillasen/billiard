module vector2D_module
# importall Base
import Base: +, -, *, dot
export Vector2D, +, -, *, dot, norm, norm2, normalize!

type Vector2D
  x::Float64
  y::Float64
end
+(a::Vector2D, b::Vector2D) = Vector2D(a.x+b.x, a.y+b.y)
-(a::Vector2D, b::Vector2D) = Vector2D(a.x-b.x, a.y-b.y)
*(a::Float64, v::Vector2D) = Vector2D(a*v.x, a*v.y)
dot(a::Vector2D, b::Vector2D) = a.x*b.x + a.y*b.y
norm(a::Vector2D) = sqrt(a.x*a.x + a.y*a.y)
norm2(a::Vector2D) = a.x*a.x + a.y*a.y
function normalize!( a::Vector2D )
  nor = norm( a )
  a.x /= nor
  a.y /= nor
end

end
