#include <iostream>

#include <cell_maze.hpp>
#include <position.hpp>
#include <solve_maze.hpp>

int
main()
{
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

  const auto first_level_solution = utl::solve_maze(maze, {0, 0}, {6, 7}, 100, 1000, false);

  if (!first_level_solution.empty()) {
    std::cout << first_level_solution;
    return 0;
  } else {
    return 1;
  }
}