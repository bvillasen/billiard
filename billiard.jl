using HDF5
current_dir = pwd()
modules_dir = current_dir * "/modules"
push!(LOAD_PATH, modules_dir)
using tools
using vector2D_module
using circle_module
using line_module


# srand(1234)
println("\nBilliard")

nParticles_def = 1024
nSnapshots_def = 100
iterPerSnapshot_def = 10
timePerSnapshot_def = 10
usingCUDA = false

for option in ARGS
  if ( option == "cuda" ) usingCUDA = true
  elseif findInArgument( option, "n_part")[1] nParticles_def = getIntFromArgument( option, "n_part")
  elseif findInArgument( option, "n_snap")[1] nSnapshots_def = getIntFromArgument( option, "n_snap")
  elseif findInArgument( option, "iter_per_snap")[1] iterPerSnapshot_def = getIntFromArgument( option, "iter_per_snap")
  elseif findInArgument( option, "time_per_snap")[1] timePerSnapshot_def = getIntFromArgument( option, "time_per_snap")
  end
end

const nParticles = nParticles_def
const nSnapshots = nSnapshots_def
const iterPerSnapshot = iterPerSnapshot_def
const timePerSnapshot = timePerSnapshot_def




#Define the geometry of the billiard
const nCircles = Int32( 1 )
circle = Circle( 0.48, Vector2D(0,0) )
const nLines = Int32( 4 )
lines = [
  Line( Vector2D( 0.5,    0 ), Vector2D(  1,  0 ) ),  #right
  Line( Vector2D(   0, -0.5 ), Vector2D(  0, -1 ) ),   #down
  Line( Vector2D(-0.5,    0 ), Vector2D( -1,  0 ) ),  #left
  Line( Vector2D(   0,  0.5 ), Vector2D(  0,  1 ) )   #up
]
#Arrays to send circle and lines properties to device
circleProperties = [ circle.r, circle.center.x, circle.center.y ]
linesProperties = Float64[]
for line in lines
  push!( linesProperties, line.center.x ); push!( linesProperties, line.center.y )
  push!( linesProperties, line.normal.x ); push!( linesProperties, line.normal.y )
end


println("\nInitializing host data...")
# posData = zeros( Float32, (nSnapshots+1, nParticles, 2 ))
posSnapshot = zeros(Float32, (nSnapshots+1, nParticles, 2) )

pos_x_all = 0.01 * rand( nParticles ) + 0.485
pos_y_all = zeros( nParticles )
theta_all = 2*pi*rand( nParticles )
vel_x_all = cos( theta_all )
vel_y_all = sin( theta_all )
region_x_all = zeros( Int32, nParticles )
region_y_all = zeros( Int32, nParticles )
collideWith_all = -1 * ones( Int32, nParticles)
times_all = zeros( Float64, nParticles )
snapshotNumber_all = ones( Int32, nParticles )

if usingCUDA
  println("\nUsing CUDA" )
  using CUDA

  #Select a CUDA device
  # list_devices()
  dev = CuDevice(0)
  #Create a context (like a process in CPU) on the selected device
  ctx = create_context(dev)

  #Compile and load CUDA code
  println( "\nCompiling CUDA code..." )
  run(`nvcc -ptx cuda_files/billiard_kernels.cu`)
  cudaModule = CuModule("billiard_kernels.ptx")

  #Extract cuda funtions
  billiard_kernel_cuda = CuFunction( cudaModule, "billiard_kernel")
  # cuda_event_syncronize = CuFunction( cudaModule, "event_syncronize")

  #Set threadBlock and blockGrid
  cudaBlock = 512  #Number of threads per block
  div = divrem( nParticles, cudaBlock )
  cudaGrid = div[1] + 1*(div[2]>0)  #Number of blocks in the grid (Protected for nParticles non-multiple of threadsPerBlock)
  println( " Threads per block: $cudaBlock\n Blocks in grid: $cudaGrid    ( nPartcles / threadsPerBlock )")

  println("\nInitializing device data...")
  pos_x_all_d = CuArray( pos_x_all )
  pos_y_all_d = CuArray( pos_y_all )
  vel_x_all_d = CuArray( vel_x_all )
  vel_y_all_d = CuArray( vel_y_all )
  region_x_all_d = CuArray( region_x_all )
  region_y_all_d = CuArray( region_y_all )
  collideWith_all_d = CuArray( collideWith_all )
  times_all_d = CuArray( times_all )
  snapshotNumber_all_d = CuArray( snapshotNumber_all )
  circleProperties_d = CuArray( circleProperties )
  linesProperties_d = CuArray( linesProperties )
  posData_x_all = zeros( Float32, (nSnapshots, nParticles) )
  posData_y_all = zeros( Float32, (nSnapshots, nParticles) )
  posData_x_all_d = CuArray( posData_x_all )
  posData_y_all_d = CuArray( posData_y_all )

  function billiard_step_cuda( snapshotNumber )
    #Launch cuda kernel: kernel_name, blocksPerGrid, threadsPerBlock, kernelArguments
    CUDA.launch( billiard_kernel_cuda, cudaGrid,  cudaBlock,
      ( nParticles, iterPerSnapshot, Int32(nSnapshots), Float32( timePerSnapshot ),
      nCircles, circleProperties_d,
      nLines, linesProperties_d, pos_x_all_d, pos_y_all_d,
      vel_x_all_d, vel_y_all_d, region_x_all_d, region_y_all_d,
      collideWith_all_d, times_all_d, snapshotNumber_all_d,
      posData_x_all_d, posData_y_all_d ) )

    #Small data transfer to syncronize divevice with host and measure time.
    copy!( circleProperties, circleProperties_d )
  end
end

posInitial_x = map( x->Float32(x), pos_x_all )
posInitial_y = map( x->Float32(x), pos_y_all )
posSnapshot[1, :, 1] = posInitial_x
posSnapshot[1, :, 2] = posInitial_y

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
        time_current = line_module.collideTime( line, pos, vel )
        if ( ( time_current > 0 ) &&  (time_current < timeMin ) )
          timeMin = time_current
          collideWith_current = line_counter
        end
      end

      #Check for collision with circle
      if collideWith_last != 0
        time_current = circle_module.collideTime( circle, pos, vel )
        if ( time_current < timeMin ) && ( time_current > 0 )
          timeMin = time_current
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
          posSnapshot[snapshotNumber, i, 1] = Float32( pos_snap.x )
          posSnapshot[snapshotNumber, i, 2] = Float32( pos_snap.y )
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
outFileName = usingCUDA ? "data_billard_cuda.h5" : "data_billard_julia.h5"
file = h5open( outDir * outFileName, "w")

println( "\nnParticles: $nParticles \nnSnapshots: $nSnapshots \nIterations per snapshot: $iterPerSnapshot \nTime per snapshot: $timePerSnapshot \nOutput: $(outDir*outFileName)\n" )
println( "Starting $(nSnapshots*iterPerSnapshot) iterations...\n")

time_compute = 0
for stepNumber in 1:nSnapshots
  printProgress( stepNumber-1, nSnapshots, time_compute )
  if usingCUDA
    time_compute += @elapsed billiard_step_cuda( stepNumber+1 )
  else
    time_compute += @elapsed billiard_kernel()
  end
end
printProgress( nSnapshots, nSnapshots, time_compute )

println( "\n\nTotal Time: $(time_compute) secs" )
println( "Compute Time: $(time_compute) secs\n" )

if usingCUDA
  copy!(posData_x_all, posData_x_all_d )
  copy!(posData_y_all, posData_y_all_d )
  posSnapshot[2:end,:,1] = posData_x_all
  posSnapshot[2:end,:,2] = posData_y_all
end

#Write positions data to h5 file
write( file, "pos_snap", posSnapshot)
close(file)



# group = g_create(file, "")
# data = d_create( file, "pos", Float64, (nParticles,2), "chunk" , (100,2) )
# # d = d_create(g, "foo", datatype(Float64), ((10,20),(100,200)), "chunk", (1,1)))
# # data = d_c

# file = h5open("data_billard.h5", "r")
#
