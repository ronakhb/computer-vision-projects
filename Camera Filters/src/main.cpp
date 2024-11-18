/*
Ronak Bhanushali
CS 5330
Spring 2024

Project 1
*/

#include "projects/project1/imgDisplay.hpp"
#include "projects/project1/vidDisplay.hpp"

int main(int argc, char *argv[])
{
    if (argc > 1)
    {
        if (strcmp(argv[1], "task1") == 0)
        {
            std::string path_to_img = argv[2];
            displayImage(path_to_img);
        }
        else if (strcmp(argv[1], "task2") == 0)
        {
            displayVideo();
        }
    }

    return 0;
}