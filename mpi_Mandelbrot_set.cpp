#include <mpi.h>
#include <complex>
#include <ctime>
#include <cstdlib>
#include <fstream>
#include <iostream>

using namespace std;

const unsigned short max_color = 65535;
const string mandelbrot_file_name = "../../../images/mandelbrot";

// Used to store mandelbrot set into an image file.
void print_result(int *result, int iter_limit, const char *name)
{
    ofstream file;
    file.open(mandelbrot_file_name);
    file << "P3" << endl;
    file << x_resolution << " " << y_resolution << endl;
    file << max_color << endl;
    for (int i = 1; i <= y_resolution; i++)
    {
        for (int j = 0; j < x_resolution; j++)
        {
            int k = result[(i - 1) * x_resolution + j];
            file << " ";
            if (k == iter_limit)
                file << 0 << " " << 0 << " " << 0;
            else
            {
                unsigned short x = k * max_color / iter_limit;
                file << x / 4 << " ";
                file << x / 2 + max_color / 2 << " ";
                file << x;
            }
            file << " ";
        }
        file << endl;
    }
    file.close();

    string command = "pnmtojpeg -quality=100 -smooth=100 -optimize ./" + mandelbrot_file_name + " > " + mandelbrot_file_name + "_limit=" + to_string(iter_limit) + "_" + name + ".jpeg";

    if (system(command.c_str()) != 0)
        exit(1);

    system(("rm " + mandelbrot_file_name).c_str());
}

//process command arguments
void get_iter_limit(int argc, char **argv, int &iter_limit, bool &print)
{
    try
    {
        switch (argc)
        {
        case 2:
            iter_limit = stoi(argv[--argc]);
            print = false;
            return;
        case 3:
            iter_limit = stoi(argv[--argc]);
            print = stoi(argv[--argc]);
            return;
        case 1:
            cout << endl;
            cout << "Pass at least one argument" << endl;
        }
    }
    catch (exception)
    {
        cout << endl;
        cout << "Wrong arguments" << endl;
    }
    cout << "Parameters syntax after program name: [<print>] <iter_limit>" << endl;
    cout << "<print>: Whether to print the set to a file or not. This must be somethig that can be converted to boolean" << endl;
    cout << "<iter_limit>: The iteration limit. This must be a positive integer" << endl;
    cout << endl;
    exit(1);
}

void output_execution_time(double time)
{
    cout << time << endl;
}

double now()
{
    struct timespec tp;
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &tp) != 0)
        exit(1);
    return tp.tv_sec + tp.tv_nsec / (double)1000000000;
}

void run(int iter_limit, int current_processor, int processors_amount, bool print)
{
    int part_width = result_size / processors_amount;

    if (current_processor == 0)
    {
        int *result = new int[result_size];
        int *current = result;

        double first_measure = now();

        // distribute tasks
        int start, end = 0;
        for (int i = 1; i < processors_amount; i++)
        {
            start = end;
            end += part_width;
            int message[2] = {start, end};
            MPI_Send(message, 2, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
        compute_mandelbrot_subset(current + end, iter_limit, end, result_size);
        // join pieces together
        for (int i = 1; i < processors_amount; i++)
        {
            MPI_Recv(current, part_width, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            current += part_width;
        }

        output_execution_time(now() - first_measure);

        if (print)
            print_result(result, iter_limit, "MPI_send_recv");
        delete[] result;
    }
    else
    {
        int message[2];
        MPI_Recv(message, 2, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        int *partial_result = new int[part_width];
        compute_mandelbrot_subset(partial_result, iter_limit, message[0], message[1]);
        MPI_Send(partial_result, part_width, MPI_INT, 0, 0, MPI_COMM_WORLD);
        delete[] partial_result;
    }
}

int main(int argc, char **argv)
{
    int iter_limit;
    bool print;
    get_iter_limit(argc, argv, iter_limit, print);

    int current_processor, processors_amount;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &current_processor);
    MPI_Comm_size(MPI_COMM_WORLD, &processors_amount);

    run(iter_limit, current_processor, processors_amount, print);

    MPI_Finalize();
    return 0;
}

const double x_begin = -2.0, x_end = 0.6;
const double y_begin = -1.2, y_end = 1.2;
const double x_step = (x_end - x_begin) / (x_resolution - 1);
const double y_step = (y_end - y_begin) / (y_resolution - 1);

void compute_mandelbrot_subset(int *result, int iter_limit, int start, int end)
{
    int i, j;
    complex<double> c, z;

#pragma omp parallel shared(result, iter_limit, start, end) private(i, j, c, z)
#pragma omp for schedule(runtime)
    for (i = start; i < end; i++)
    {
        c = complex<double>(
            x_begin + (i % x_resolution) * x_step,
            y_begin + (i / x_resolution) * y_step);
        z = 0;
        j = 0;
        while (norm(z) <= 4 && j < iter_limit)
        {
            z = z * z + c;
            j++;
        }
        result[i - start] = j;
    }
}