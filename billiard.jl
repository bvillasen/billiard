# using CUDA
# using PyCall
# using PyPlot
using HDF5
push!(LOAD_PATH, "/home/bruno/Desktop/Dropbox/Developer/billiard/modules")
using tools
using vector2D
using circle_module
using line_module

println("\nBilliard")

const nParticles = 1000
const nSnapshots = 100
const iterPerSnapshot = 50
const timePerSnapshot = 5

#Define the geometry of the billiard
circle = Circle( 0.48, Vector2D(0,0) )
lines = [
  Line( Vector2D( 0.5,    0 ), Vector2D(  1,  0 ), 3 ),  #right
  Line( Vector2D(   0, -0.5 ), Vector2D(  0, -1 ), 4 ),   #down
  Line( Vector2D(-0.5,    0 ), Vector2D( -1,  0 ), 1 ),  #left
  Line( Vector2D(   0,  0.5 ), Vector2D(  0,  1 ), 2 )   #up
]

# posData = zeros(nSnapshots+1, nParticles, 2)
posSnapshot = zeros(nSnapshots+1, nParticles, 2)

pos_x_all = 0.01 * rand( nParticles ) + 0.485
pos_y_all = zeros( nParticles )
theta_all = 2*pi*rand( nParticles )
vel_x_all = cos( theta_all )
vel_y_all = sin( theta_all )
region_x_all = zeros( nParticles )
region_y_all = zeros( nParticles )
collideWith_all = -1 * ones( Int, nParticles)
times_all = zeros( nParticles )
snapshotNumber_all = ones( Int, nParticles )





#Save initial conditions
# posData[1, :, 1] = pos_x_all
# posData[1, :, 2] = pos_y_all
posSnapshot[1, :, 1] = pos_x_all
posSnapshot[1, :, 2] = pos_y_all

function billiard_kernel()
  for i in 1:nParticles
    pos = Vector2D( pos_x_all[i], pos_y_all[i] )
    vel = Vector2D( vel_x_all[i], vel_y_all[i] )
    normalize!( vel )
    region = Vector2D( region_x_all[i], region_y_all[i] )
    collideWith_last = collideWith_all[i]
    timeTotal = times_all[i]
    snapshotNumber = snapshotNumber_all[i]
    for iterNumber in 1:iterPerSnapshot
      timeMin = 1e5
      collideWith_current = collideWith_last
      #Check for collision with lines
      line_counter = 0
      for line in lines
        line_counter += 1
        if line_counter == collideWith_current
          continue
        end
        time = line_module.collideTime( line, pos, vel )
        if time < 0
          continue
        end
        if time < timeMin
          timeMin = time
          collideWith_current = line_counter
        end
      end

      #Chech for collision with circle
      if collideWith_last != 0
        time = circle_module.collideTime( circle, pos, vel )
        if ( time < timeMin ) && ( time > 0 )
          timeMin = time
          collideWith_current = 0
        end
      end

      #Advance position and time of the particle
      pos = pos + timeMin * vel
      timeTotal += timeMin

      #Check if particle has passed a snapshot_time
      if timeTotal > snapshotNumber*timePerSnapshot
        dt = timeTotal - snapshotNumber*timePerSnapshot
        snapshotNumber += 1
        if snapshotNumber <= nSnapshots+1
          pos_snap = (pos + region) - dt*vel
          posSnapshot[snapshotNumber, i, 1] = pos_snap.x
          posSnapshot[snapshotNumber, i, 2] = pos_snap.y
        end
      end

      #Bounce whith circle or change region with periodic line
      if collideWith_current == 0
        vel = circle_module.bounce( circle, pos, vel )
      else
        pos, region = changePosPeriodic( lines[collideWith_current], pos, region)
        collideWith_current += 2   #This only works for 4 rectangular walls
        if collideWith_current > 4
          collideWith_current -= 4
        end
      end
      collideWith_last = collideWith_current
    end
    pos_x_all[i] = pos.x
    pos_y_all[i] = pos.y
    vel_x_all[i] = vel.x
    vel_y_all[i] = vel.y
    region_x_all[i] = region.x
    region_y_all[i] = region.y
    collideWith_all[i] = collideWith_last
    times_all[i] = timeTotal
    snapshotNumber_all[i] = snapshotNumber
  end
end

outDir = ""
outFileName = "data_billard.h5"
file = h5open( outDir * outFileName, "w")




println( "\nnParticles: $nParticles \nnSnapshots: $nSnapshots \nnIterations per snapshot: $iterPerSnapshot \nOutput: $(outDir*outFileName)\n" )
println( "Starting $(nSnapshots*iterPerSnapshot) iterations...\n")


time_compute = 0
for stepNumber in 1:nSnapshots
  printProgress( stepNumber-1, nSnapshots, time_compute )
  time_compute += @elapsed billiard_kernel()
end
printProgress( nSnapshots, nSnapshots, time_compute )

println( "\n\nTotal Time: $(time_compute) secs" )
println( "Compute Time: $(time_compute) secs\n" )


# write( file, "pos_iter", posData)
write( file, "pos_snap", posSnapshot)
# group = g_create(file, "")
# data = d_create( file, "pos", Float64, (nParticles,2), "chunk" , (100,2) )
# # d = d_create(g, "foo", datatype(Float64), ((10,20),(100,200)), "chunk", (1,1)))
# # data = d_c
close(file)


# file = h5open("data_billard.h5", "r")
#
