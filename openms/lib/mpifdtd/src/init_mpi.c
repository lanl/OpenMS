#include <mpi.h>
#include "init.hpp"

int mygrid_nx, mygrid_ny, mygrid_nz;
int m_mygrid_nx, m_mygrid_ny, m_mygrid_nz;
int p_mygrid_nx, p_mygrid_ny, p_mygrid_nz;
int c_mygrid_nx, c_mygrid_ny, c_mygrid_nz;

// coordinates of each grid
float *mygrid_x, *mygrid_y, *mygrid_z;

// Initialize MPI
void init_mpi(int argc, char* argv[]) {

    MPI_Init(&argc, &argv);

    // Get the rank of the current process
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    // Get the total number of MPI processes
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // printf("Rank: %d, Total Processes: %d\n", mpi_rank, mpi_size);
}

// Finalize MPI
void finalize_mpi() {
    MPI_Finalize();
}


void setup_MPI(){
    // may move the init_mpi into setup_MPI!!!
    // configure MPI for FDTD


    myprintf("setting up MPI for FDTD simulations\n\n");

    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // CHECK THAT THE DESIRED NUMBER OF PROCESSORS HAS BEEN ALLOCATED ...
    if( (mpi_grid_proces[0]*mpi_grid_proces[1]*mpi_grid_proces[2]) != mpi_size)
    {
        myprintf("Error in init_MPI(): The product of number of processors in each dimension is not equal to the total number of processors!\n");
        myprintf("mpi_grid_proces[0] = %d\n", mpi_grid_proces[0]);
        myprintf("mpi_grid_proces[1] = %d\n", mpi_grid_proces[1]);
        myprintf("mpi_grid_proces[2] = %d\n", mpi_grid_proces[2]);
        myprintf("mpi_size           = %d\n", mpi_size);
        exit(0);
    }

    //mpi_nxprocs = mpi_grid_proces[0];
    //mpi_nyprocs = mpi_grid_proces[1];
    //mpi_nzprocs = mpi_grid_proces[2];

    int mpi_grid_id;
    int periodicity[3] = {1, 1, 1};

    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    // This function creates a new communicator (mpi_grid_comm) with a Cartesian topology.
    MPI_Cart_create(MPI_COMM_WORLD, 3, mpi_grid_proces, periodicity, 1, &mpi_grid_comm);
    MPI_Comm_rank(mpi_grid_comm, &mpi_grid_id);

    // Get the Cartesian coordinates of this process
    MPI_Cart_coords(mpi_grid_comm, mpi_grid_id, 3, mpi_grid_coords);

    //printf("mpi_grid_id = %d mpi_grid_coords = %d    %d   %d   %d\n", mpi_grid_id,
    //mpi_grid_coords[0], mpi_grid_coords[1], mpi_grid_coords[2]);

    myprintf("mpi_size = %d\n", mpi_size);
    myprintf("mpi_nxprocs = %d\n", mpi_nxprocs);

    // SET THE E FIELD TO SEND BELOW AND RECIEVE FROM ABOVE, WRAPPING AROUND ...
    int mpi_coord_send[3], mpi_coord_recv[3], mpi_proc_send[3], mpi_proc_recv[3];

    for( int i = 0; i < 3; ++i )
    {
        for( int j = 0; j < 3; ++j )
        {
            mpi_coord_send[j] = mpi_grid_coords[j];
            mpi_coord_recv[j] = mpi_grid_coords[j];
        }

        if( mpi_coord_send[i] == 0 )
        {
            mpi_coord_send[i] = mpi_grid_proces[i] - 1;
        }
        else
        {
            --mpi_coord_send[i];
        }

        if( mpi_coord_recv[i] == (mpi_grid_proces[i] - 1) )
        {
            mpi_coord_recv[i] = 0;
        }
        else
        {
            ++mpi_coord_recv[i];
        }

        MPI_Cart_rank(mpi_grid_comm, mpi_coord_send, &mpi_proc_send[i]);
        MPI_Cart_rank(mpi_grid_comm, mpi_coord_recv, &mpi_proc_recv[i]);
    }

    mpi_efield_isend = mpi_proc_send[0];
    mpi_efield_irecv = mpi_proc_recv[0];
    mpi_efield_jsend = mpi_proc_send[1];
    mpi_efield_jrecv = mpi_proc_recv[1];
    mpi_efield_ksend = mpi_proc_send[2];
    mpi_efield_krecv = mpi_proc_recv[2];
    //printf("mpi_grid_id = %d mpi_efield_isend/irecv = %d   %d\n",
    //mpi_grid_id, mpi_efield_isend, mpi_efield_irecv);

    // SET THE H FIELD TO SEND ABOVE AND RECIEVE FROM BELOW, WRAPPING AROUND ...
    for( int i = 0; i < 3; ++i )
    {
        for(int j = 0; j < 3; ++j)
        {
            mpi_coord_send[j] = mpi_grid_coords[j];
            mpi_coord_recv[j] = mpi_grid_coords[j];
        }

        if( mpi_coord_send[i] == (mpi_grid_proces[i]-1) )
        {
            mpi_coord_send[i] = 0;
        }
        else
        {
            ++mpi_coord_send[i];
        }

        if( mpi_coord_recv[i] == 0 )
        {
            mpi_coord_recv[i] = mpi_grid_proces[i] - 1;
        }
        else
        {
            --mpi_coord_recv[i];
        }

        // Converts the modified coordinates back to the rank of the corresponding
        // process in the Cartesian grid communicator. This rank is then used for
        // sending and receiving data.
        MPI_Cart_rank(mpi_grid_comm, mpi_coord_send, &mpi_proc_send[i]);
        MPI_Cart_rank(mpi_grid_comm, mpi_coord_recv, &mpi_proc_recv[i]);
    }

    mpi_hfield_isend = mpi_proc_send[0];
    mpi_hfield_irecv = mpi_proc_recv[0];
    mpi_hfield_jsend = mpi_proc_send[1];
    mpi_hfield_jrecv = mpi_proc_recv[1];
    mpi_hfield_ksend = mpi_proc_send[2];
    mpi_hfield_krecv = mpi_proc_recv[2];
}

void init_grid(){
    int grid_size[3];

    grid_size[0] = floor(0.5+(xsize*lattice_x));
    grid_size[1] = floor(0.5+(ysize*lattice_y));
    ////////////////////////////////////
    ///// for non_uniform_grid() ///////
    ////////////////////////////////////
    grid_size[2] = non_uniform_z_to_i(0.5*zsize);
    myprintf("non uniform grid_size[2] = %d\n", grid_size[2]); ////////////

#ifdef MPION
    int dn[3];

    int dn_remainder[3];
    for (int i = 0; i < 3; i++){
        dn[i] = grid_size[i] / mpi_grid_proces[i];
        dn_remainder[i] = grid_size[i] % mpi_grid_proces[i];

        if (dn_remainder[i] != 0){
           if (mpi_grid_coords[0] < dn_remainder[i]){
              mpi_grid_n1[i] = mpi_grid_coords[i] * (dn[i] + 1) + 1;
              mpi_grid_n2[i] = (mpi_grid_coords[i] + 1) * (dn[i] + 1);
           }
           else
           {
              mpi_grid_n1[i] = mpi_grid_coords[i] * (dn[i] + 1) + 1 + dn_remainder[i];
              mpi_grid_n2[i] = (mpi_grid_coords[i] + 1) * dn[i] + dn_remainder[i];
           }
        }
        else
        {
           mpi_grid_n1[i] = mpi_grid_coords[i] * dn[i] + 1;
           mpi_grid_n2[i] = (mpi_grid_coords[i] + 1) * dn[i];
        }
    }
#else
    for (int i = 0; i < 3; i++){
        mpi_grid_n1[i] = 1;
        mpi_grid_n2[i] = grid_size[i];
    }
#endif

    // new we get the grid size on each node
    mygrid_nx = mpi_grid_n2[0] - mpi_grid_n1[0] + 1;
    mygrid_ny = mpi_grid_n2[1] - mpi_grid_n1[1] + 1;
    mygrid_nz = mpi_grid_n2[2] - mpi_grid_n1[2] + 1;

    printf("number of grid (isize) in old code = %d  %d \n", (int)floor(0.5+(xsize*lattice_x)), grid_size[0]);
    printf("number of grid is %d %d %d\n", mygrid_nx, mygrid_ny, mygrid_nz);

    xcenter = xsize / 2;
    ycenter = ysize / 2;
    zcenter = zsize / 2;  // always true even with non_uniform_grid()

    m_mygrid_nx = mygrid_nx;

    p_mygrid_nx = mygrid_nx - 1;
    c_mygrid_nx = mygrid_nx / 2;

    m_mygrid_ny = mygrid_ny;
    p_mygrid_ny = mygrid_ny - 1;
    c_mygrid_ny = mygrid_ny / 2;

    m_mygrid_nz = mygrid_nz;
    p_mygrid_nz = mygrid_nz - 1;
    c_mygrid_nz = mygrid_nz / 2;  // in case of using non_uniform_grid(), this will not be used.

    printf("number of m_mygrid is %d %d %d\n", m_mygrid_nx, m_mygrid_ny, m_mygrid_nz);

    // coordinates of grids
    mygrid_x = float_1d_memory(mygrid_nx);
    mygrid_y = float_1d_memory(mygrid_ny);
    mygrid_z = float_1d_memory(mygrid_nz);

    // get coordinates of each grid
    for(int i = 1; i <= mygrid_nx; ++i)
    {
        mygrid_x[i-1] = (mpi_grid_n1[0] + i - 1)*1.0;
	printf("coordinates_x  %d   %f   %d \n", i, mygrid_x[i], mpi_grid_n1[0] + i);
    }

    for(int j = 1; j <= mygrid_ny; ++j)
    {
        mygrid_y[j-1] = (mpi_grid_n1[1] + j - 1)*1.0;
    }
    for(int k = 1; k <= mygrid_nz; ++k)
    {
        mygrid_z[k-1] = (mpi_grid_n1[2] + k - 1)*1.0;
    }
}


int findgrid_x(float x){
   int index = -1;
   float target = floor(0.5+((x+xcenter)*lattice_x));

#ifdef MPION
    for (int j = 0; j < mygrid_nx; j++) {
        float diff = fabs(mygrid_x[j] - target);

        if (diff < 0.1) {
            index = j;
        }
    }
#else
   index = floor(0.5+((x+xcenter)*lattice_x));
#endif
   return index;
}

int findgrid_y(float y){
   int index = -1;
   float target = floor(0.5+((y+ycenter)*lattice_y));

#ifdef MPION
    for (int j = 0; j < mygrid_nx; j++) {
        float diff = fabs(mygrid_y[j] - target);

        if (diff < 0.1) {
            index = j;
        }
    }
#else
   index = floor(0.5+((y+ycenter)*lattice_y));
#endif
   return index;
}

// initialization
void initialize(){

   /* release information */
   myprintf("\n=======================================\n");
   myprintf("       mpiFDTD++ ver. %2.3f ", FDTDver);
   myprintf("\n=======================================\n\n");

#ifdef MPION
   myprintf("MPI is used!\n");
   setup_MPI();
#endif

   init_grid();

}


// finalize the simulations
void finalize(){


}
