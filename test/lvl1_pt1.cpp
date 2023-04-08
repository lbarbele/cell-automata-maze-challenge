#include <exception>
#include <iostream>
#include <filesystem>

#include <cell_maze.hpp>
#include <matrix.hpp>
#include <position.hpp>
#include <solve_maze.hpp>

bool
test_evolution(
  const utl::cell_maze<uint>& input_maze,
  const std::filesystem::path data_path,
  const uint generations
)
{
  auto maze = input_maze;

  for (uint i = 0; i <= generations; ++i) {
    const auto fname = "lvl1_pt1_" + std::to_string(i) + ".txt";
    const auto true_states = utl::matrix<uint>::from_file(data_path / fname);

    if (maze.states() != true_states)
      return false;

    maze.evolve();
    maze.clear_cell(0);
    maze.clear_cell(maze.size() - 1);
  }

  return true;
}

bool
test_solution(
  const std::filesystem::path data_path,
  const utl::position_list& solution
)
{

  uint generation = 0;

  for (const auto& pos : solution) {
    const auto fname = "lvl1_pt1_" + std::to_string(generation++) + ".txt";
    const auto states = utl::matrix<uint>::from_file(data_path / fname);
    if (states[pos]) {
      return false;
    }
  }

  return true;
}

int
main(
  int argc,
  char** argv
)
{
  if (argc != 2) {
    std::cout << "bad input" << std::endl;
    std::cout << "usage: ./lvl1_pt1 <data_path>" << std::endl;
    return 1;
  }

  const utl::position_list initial_live_cells = {
    {1, 4},
    {2, 2}, {2, 4}, {2, 5},
    {3, 1}, {3, 2}, {3, 5}, {3, 6},
    {4, 2}, {4, 4}, {4, 5},
    {5, 4}
  };

  // create an empty maze and set the initial live cells
  utl::cell_maze<uint> maze(7, 8);

  for (const auto& pos : initial_live_cells) {
    maze.set_cell(pos);
  }

  // customize the cell propagation dynamics
  maze.set_underpopulation_rule(3);
  maze.set_overpopulation_rule(7);
  maze.set_reproduction_rules(2, 3);

  // test maze evolution
  const bool evolution_ok = test_evolution(maze, argv[1], 30);

  if (!evolution_ok) {
    throw std::runtime_error("maze evolution incorrect");
  }

  // test maze solution
  const auto solution = utl::solve_maze(maze, {0, 0}, {6, 7}, 100, 1000, false);

  const bool solution_ok = test_solution(argv[1], solution);

  if (!solution_ok) {
    throw std::runtime_error("solution is incorrect");
  }

  std::cout << "lvl1_pt1: all tests passed :)" << std::endl;

  return 0;
}