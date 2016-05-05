#include "vector2D.cu"
#include "circle.cu"
#include "line.cu"


extern "C"{

__global__ void billiard_kernel( const int nParticles, const int iterPerSnapshot,
	         const int nSnapshots, const float timePerSnapshot,
			     const int nCircles, double *circlesProperties,
           const int nLines, double *linesProperties,
           double *pos_x_all, double *pos_y_all, double *vel_x_all, double *vel_y_all,
           int *region_x_all, int *region_y_all, int *collideWith_all,
           double *times_all, int*snapshotNumber_all,
				   float *posData_x_all, float *posData_y_all){
	//Get thread id
	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if ( tid >= nParticles ) return;

	//Initialize circle
	Circle circle( circlesProperties[0], Vector2D( circlesProperties[1], circlesProperties[2]) );

	//Initialize lines
	Line lines[4];  //Number of lines
	int i;
	for ( i=0; i<nLines; i++ ){
		lines[i] = Line( Vector2D( linesProperties[i*4], linesProperties[i*4 +1] ), Vector2D( linesProperties[i*4 +2], linesProperties[i*4 +3] )  );
	}

	//Load particle properties from global memory
	Vector2D pos( pos_x_all[tid], pos_y_all[tid] );
	Vector2D vel( vel_x_all[tid], vel_y_all[tid] );
	vel.normalize();
	Vector2D region( region_x_all[tid], region_y_all[tid] );
	int collideWith_last = collideWith_all[tid];
	double timeTotal = times_all[tid];
	int snapshotNumber = snapshotNumber_all[tid];

	Vector2D deltaPos;
	double timeMin, time_current;
	int collideWith_current, line_counter;
	for ( i=0; i<iterPerSnapshot; i++){
		timeMin = 1e5;
		collideWith_current = collideWith_last;

		//Check for collision with lines
		for ( line_counter=1; line_counter <= nLines; line_counter++){
			if (line_counter == collideWith_current) continue;
			time_current = lines[line_counter-1].collideTime( pos, vel );
			if ( time_current > 0 && time_current < timeMin ){
				timeMin = time_current;
				collideWith_current = line_counter;
			}
		}

		//Check for collision with circle
		if ( collideWith_last != 0 ){
			time_current = circle.collideTime( pos, vel );
			if ( time_current > 0 && time_current < timeMin ){
				timeMin = time_current;
				collideWith_current = 0;
			}
		}

		//Advance position and time of the particle
	 	deltaPos = vel/timeMin;
		pos = pos + deltaPos;
		timeTotal += timeMin;

		//Check if particle has passed a snapshot_time
		if ( timeTotal > snapshotNumber*timePerSnapshot ){
			deltaPos = vel / ( snapshotNumber*timePerSnapshot - timeTotal );
			deltaPos = (pos + region) + deltaPos;
			if ( snapshotNumber <= nSnapshots ){
				posData_x_all[ tid*nSnapshots + snapshotNumber-1 ] = float( deltaPos.x );
				posData_y_all[ tid*nSnapshots + snapshotNumber-1 ] = float( deltaPos.y );
				snapshotNumber += 1;
			}
		}

		//Bounce whith circle or change region with periodic line
    if ( collideWith_current == 0 ) circle.bounce( pos, vel );
		else{
			lines[collideWith_current-1].changePosPeriodic( pos, region );
			collideWith_current += 2;
			if ( collideWith_current > 4 ) collideWith_current -= 4;
		}
		collideWith_last = collideWith_current;
	}
	//Update particle properties in global memory
	pos_x_all[tid] = pos.x;
	pos_y_all[tid] = pos.y;
	vel_x_all[tid] = vel.x;
	vel_y_all[tid] = vel.y;
	region_x_all[tid] = region.x;
	region_y_all[tid] = region.y;
	collideWith_all[tid] = collideWith_last;
	times_all[tid] = timeTotal;
	snapshotNumber_all[tid] = snapshotNumber;
}



}//end extern "C"

// __global__ void event_syncronize( int *foo ){
// 	//Get thread id
// 	int tid = blockDim.x*blockIdx.x + threadIdx.x;
// 	return;}
