module cudaTools

importall Base, CUDA

export copyDtoD

function copyDtoD{T}( dst::CuArray{T}, src::CuArray{T} )
    if length(dst) != length(src)
        throw(ArgumentError("Inconsistent array length."))
    end
    nbytes = length(src) * sizeof(T)
    @cucall(cuMemcpyDtoD, (CUdeviceptr, CUdeviceptr, Csize_t), dst.ptr.p, src.ptr.p, nbytes)
    return dst
end

end
