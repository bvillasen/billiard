module billiard_module
export billiard_kernel

using vector2D_module
using line_module
using circle_module


function billiard_kernel(nParticles, iterPerSnapshot, nSnapshots, timePerSnapshot,
  circle, lines, pos_x_all, pos_y_all, vel_x_all, vel_y_all,
  region_x_all, region_y_all, collideWith_all, times_all, snapshotNumber_all,
  posSnapshot )
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

end
