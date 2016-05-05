module tools

importall Base

export printProgress, writeSnapshot, findInArgument, getIntFromArgument


function findInArgument( inStr, findStr )
  range = search( inStr, findStr )
  len = length( range )
  if len == 0 return (false,-1)
  else return (true, maximum(range) )
  end
end

function getIntFromArgument( inStr, findStr )
  indx = findInArgument( inStr, findStr )[2] + 2
  valStr = inStr[indx:end]
  val = parse( Int, valStr )
  return val
end

function writeSnapshot( n, name, data, outFile; stride=1)
  snapNumber = n < 10 ? "0$n" : "$n"
  snapNumber = n < 100 ? "0$snapNumber" : snapNumber
  key = name * "_snap_" * snapNumber
  # println( key )
  # outFile[ key, "chunk", (4,4,4), "blosc", 3] = data[1:stride:end,1:stride:end,1:stride:end]
  outFile[ key ] = map( Float32, data[1:stride:end,1:stride:end,1:stride:end] )
end

function *(n::Int64, s::ASCIIString)
  r = ""
  for i in 1:n
    r = r * s
  end
  return r
end
*( s::ASCIIString, n::Int64 ) = n*s


function printProgress( current, total, time )
  terminalString = "\rProgress: "
  percent = total==0 ? 100.*current : 100.*current/total
  nDots =  round( Int, divrem(percent, 5)[1] )
  dotsString = "[" * (nDots*".") * ((20-nDots)*" ") * "]"
  percentString = "$(round(  Int, percent))%"
  if current != 0
    ETR = time*(total - current)/current
    hours = round( Int, divrem(ETR , 3600)[1] )
    minutes =  round( Int, divrem( (ETR - 3600*hours), 60 )[1] )
    seconds = round( Int, ETR - 3600*hours - 60*minutes)
    ETRstring = "  ETR= $(hours):$(minutes):$(seconds)    "
  else
    ETRstring = "  ETR=    "
  end
  ETRstring =  time < 0.0001 ? "  ETR=    " : ETRstring
  print( terminalString * dotsString * percentString * ETRstring )
#   print( current )
end

end




# def printProgressTime( current, total,  deltaTime ):
#   terminalString = "\rProgress: "
#   if total==0: total+=1
#   percent = 100.*current/total
#   nDots = int(percent/5)
#   dotsString = "[" + nDots*"." + (20-nDots)*" " + "]"
#   percentString = "{0:.0f}%".format(percent)
#   if current != 0:
#     ETR = (deltaTime*(total - current))/float(current)
#     #print ETR
#     hours = int(ETR/3600)
#     minutes = int(ETR - 3600*hours)/60
#     seconds = int(ETR - 3600*hours - 60*minutes)
#     ETRstring = "  ETR= {0}:{1:02}:{2:02}    ".format(hours, minutes, seconds)
#   else: ETRstring = "  ETR=    "
#   if deltaTime < 0.0001: ETRstring = "  ETR=    "
#   terminalString  += dotsString + percentString + ETRstring
#   sys.stdout. write(terminalString)
#   sys.stdout.flush()
