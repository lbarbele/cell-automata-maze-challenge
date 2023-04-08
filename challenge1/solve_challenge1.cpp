#include <iostream>
#include <filesystem>
#include <fstream>

#include <cell_maze.hpp>
#include <position.hpp>
#include <solve_maze.hpp>

int
main(
  int argc,
  char** argv
)
{
  // parse input

  if (argc != 3) {
    std::cout << "bad input" << std::endl;
    std::cout << "usage: ./solve_challenge1 path/to/input.txt path/to/output.txt" << std::endl;
    return 1;
  }

  const std::filesystem::path input_path = argv[1];
  const std::filesystem::path output_path = argv[2];

  // open the output file and check
  std::ofstream output_file(output_path);

  if (!output_file.is_open()) {
    std::cerr << "unable to open the output file " << output_path << std::endl;
    return 1;
  }

  // configuration
  const uint max_steps = 10000;
  const std::size_t drop_dist = 200;
  const bool verbose = true;


  // create the maze
  auto maze = utl::cell_maze<uint>::from_file(input_path, true);

  // set start/end positions in the maze corners
  const utl::position start_pos = {0, 0};
  const utl::position end_pos = {maze.rows()-1, maze.cols()-1};

  // solve the maze
  const auto solution = utl::solve_maze(maze, start_pos, end_pos, max_steps, drop_dist, verbose);

  // convert solution to steps
  const auto steps = utl::solution_to_string(solution);

  // print steps to the output file
  output_file << steps << std::endl;

  return 0;
}