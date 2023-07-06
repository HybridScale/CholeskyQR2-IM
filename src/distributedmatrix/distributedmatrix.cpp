#include "distributedmatrix.hpp"
#include <mpi.h>
#include <vector>


DistributedMatrix::DistributedMatrix(std::int64_t m, std::int64_t n, MPI_Comm Comm) : m_(m), n_(n), mpi_comm_(Comm)
{
    MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);

    counts.resize(world_size_);
    displacements.resize(world_size_);

    MPI_Type_contiguous(n, MPI_DOUBLE, &row_datatype);
    MPI_Type_commit(&row_datatype);

    localm_ = ceil( (1.0*m_)/world_size_ );

    count_and_dispplacement();
}


MPI_Datatype DistributedMatrix::get_datatype()
{
    return row_datatype;
}


std::vector<int> DistributedMatrix::get_displacements()
{
    return displacements;
}


std::vector<int> DistributedMatrix::get_counts()
{
    return counts;
}


std::int64_t DistributedMatrix::get_rank_localm()
{
    return localm_;
}


int DistributedMatrix::get_rank_displacement()
{
    return displacements[world_rank_];
}


int DistributedMatrix::get_rank_count()
{
    return counts[world_rank_];
}


void DistributedMatrix::count_and_dispplacement()
{
// calculate counts and displacements for scatterv and gattherv for matrix divided
// by rows

    int localm_tmp = localm_;

    counts[0] = localm_;
    displacements[0] = 0;

    for(int i=1; i < world_size_; ++i)
    {

        if(i == (world_size_-1))
        {
            localm_tmp   = m_ - i * localm_ ;
        }
        
        counts[i] = localm_tmp;
        displacements[i] = i * counts[i-1];
    }

    if(world_rank_ == (world_size_-1))
        localm_   = m_ - world_rank_ * localm_ ;
}

