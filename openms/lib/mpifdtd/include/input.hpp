


int objectn=0;
int in_object(int n,float i,float j,float k);
float average_epsilon(float i,float j,float k);
float rand_distance_2D(float x_scan, float y_scan, float x_temp, float y_temp);
float rand_distance_3D(float x_scan, float y_scan, float z_scan, float x_temp, float y_temp, float z_temp);

float back_epsilon;
struct obj *object;


void reading_matrix(char *matrix_file, int n);
void generate_random_2D(float x_min, float x_max, float y_min, float y_max, float radius, float *xr, float *yr, int i);
void generate_random_3D(float x_min, float x_max, float y_min, float y_max, float z_min, float z_max, float radius, float *xr, float *yr, float *zr, int i);


