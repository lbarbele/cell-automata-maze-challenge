#include <chrono>
#include <iostream>
#include <list>

#include "matrix.hpp"
#include "maze.hpp"

int
main(
  int argc,
  char** argv
)
{
  // using ms = std::chrono::milliseconds;

  // // build the matrix
  // auto t0 = std::chrono::high_resolution_clock::now();
  // const auto m = Matrix<uint>::from_file(argv[1]);
  // auto t = std::chrono::high_resolution_clock::now();
  // std:: cout << "matrix built in " << std::chrono::duration_cast<ms>(t - t0) << std::endl;

  // t0 = std::chrono::high_resolution_clock::now();
  // const auto mz = Maze(m);
  // t = std::chrono::high_resolution_clock::now();
  // std::cout << "maze created in " << std::chrono::duration_cast<ms>(t - t0) << std::endl;

  // first challenges
  std::vector<Position> initial_live_cells = {
    {1, 4},
    {2, 2}, {2, 4}, {2, 5},
    {3, 1}, {3, 2}, {3, 5}, {3, 6},
    {4, 2}, {4, 4}, {4, 5},
    {5, 4},
  };

  auto mtx = Matrix<uint>::full(7, 8, Maze::Cell::dead);
  for (const auto& p : initial_live_cells) {
    mtx[p] = Maze::Cell::live;
  }

  mtx[{0, 0}] = Maze::Cell::start;
  mtx[{6, 7}] = Maze::Cell::end;

  Maze mz(mtx);

  std::cout << mz.config << std::endl;
  std::cout << mz.evolve().config << std::endl;
  std::cout << mz.evolve().config << std::endl;
  std::cout << mz.evolve().config << std::endl;
  std::cout << mz.evolve().config << std::endl;

  return 0;
}