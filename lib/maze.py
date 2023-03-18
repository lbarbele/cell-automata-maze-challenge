from __future__ import annotations
import random

import matplotlib.pyplot as plt
import numpy as np

from .cell import Cell

Position = tuple[int, int]

class Maze():
  """
  The Maze class represents a rectangular maze of cell automata and provides methods
  to analyze its dynamics. The maze is constructed in its initial state by a numpy
  matrix of cell values (compatible with the Cell enum). Then, its state can be
  evolved by successive calls to ```Maze.evolve()```.

  To find the shortest path between some initial position and the finishing point,
  two methods are provided:
  - ```Maze.solve_random()```: will try to find a solution by randomly walking through the maze
  - ```Maze.solve()```: will consider every possible path until some of them leads to the end
  """

  def __init__(self, matrix: np.ndarray):
    """
    Maze constructor: takes a cell matrix as input, whose values will be checked. The
    matrix is required to contain exactly one starting and one finishing cell. It is
    also required that all cell values are valid (i.e. can be converted to the enum Cell).
    """
    # check input
    if not isinstance(matrix, np.ndarray):
      raise RuntimeError('input matrix is invalid (expected a numpy array)')

    if len(matrix.shape) != 2:
      raise RuntimeError(f'input matrix has {len(matrix.shape)} dimensions whereas 2 were expected')

    # copy the input matrix with integer dtype and retrieve its shape information
    self.maze = np.array(matrix, dtype = int)
    self.shape = self.maze.shape
    self.rows = self.shape[0]
    self.cols = self.shape[1]

    # loop over cells to find start/end and check values
    self.start = None
    self.finish = None

    for pos, cell in np.ndenumerate(matrix):
      if cell == Cell.START and self.start is None:
        self.start = pos
      elif cell == Cell.FINISH and self.finish is None:
        self.finish = pos
      elif cell == Cell.START and not (self.start is None):
        raise RuntimeError('more than one starting point found in matrix')
      elif cell == Cell.FINISH and not (self.finish is None):
        raise RuntimeError('more than one finishing point found in matrix')
      elif cell != Cell.WHITE and cell != Cell.GREEN:
        raise RuntimeError(f'invalid cell found (value = {cell})')

    # check if start and finish were found
    if self.start is None or self.finish is None:
      raise RuntimeError('input matrix is incomplete: missing start or finish tiles')

  def clone(self) -> Maze:
    """
    Returns a deep clone of this maze.

    Arguments: none
    
    Returns: a ```Maze``` object, clone if this maze
    """
    return Maze(self.maze)
  
  def count_green_neighbours(self, position: Position) -> int:
    """
    Counts the number of green cells among the direct neighbours of the cell at the
    given position.

    Arguments:
    - ```position: tuple[int, int]```: position of the cell around which the number of green cells will
    be computed

    Returns: an integer corresponding to the count of green neighbours.
    """
    count = 0

    for ij in self.get_neighbours(position):
      if self.maze[ij] == Cell.GREEN:
        count += 1
    
    return count

  def draw(self, position: Position = None, ms: int = 20) -> None:
    """
    Draw the maze using ```matplotlib```'s ```matshow``` method. If a position is given
    as argument, a cursor is drawn at such position and all possible valid moves are
    marked in the grid.

    Arguments:
    - ```position: tuple[int, int]```: cursor position
    - ```ms: int```: cursor size

    Returns: a ```matplotlib.image.AxesImage``` object (returned from ```plt.matshow```)
    """
    axes_image = plt.matshow(self.maze, cmap = 'tab10')
    if not (position is None):
      plt.plot(position[1], position[0], 'kD', ms = ms)
      moves = np.array(self.get_valid_moves(position))
      if len(moves) > 0:
        plt.plot(moves[:, 1], moves[:, 0], 'k+')

    return axes_image
  
  def evolve(self, overwrite: bool = True) -> Maze:
    """
    Apply cell dynamics to every cell in the maze. By default, this maze is overwritten
    and returned by the method. If the parameter ```overwrite``` is disabled, the maze
    is not modified and the returned value is a new Maze after the cell evolution.

    Arguments:
    - ```overwrite: bool```: enable/disable overwrite of this Maze

    Returns: ```self``` if overwrite is enabled or a new ```Maze``` otherwise.
    """
    other = np.full(self.maze.shape, Cell.WHITE, dtype = int)

    for pos in np.ndindex(self.shape):
      other[pos] = self.get_mutation(pos)

    if overwrite:
      self.maze = other
      return self
    else:
      return Maze(other)

  def get_cell_indices(self, cell_type: Cell) -> list[Position]:
    """
    Find all positions of cells of given type.

    Arguments:
    - ```cell_type: Cell```: type of cell to be found

    Returns: a list of ```Position``` containing the positions of the given cell type.
    """
    return [pos for pos, c in np.ndenumerate(self.maze) if c == cell_type]
  
  def get_mutation(self, position: Position) -> Cell:
    """
    Apply cell dynamics to determine how the cell at the given position will evolve.

    Arguments:
    - ```position: Position```: position of the cell of interest

    Returns: a ```Cell``` corresponding to the cell evolution
    """
    type = self.maze[position]

    if type == Cell.START or type == Cell.FINISH:
      return type
    
    ngreen = self.count_green_neighbours(position)
    if type == Cell.WHITE and (1 < ngreen and ngreen < 5):
      return Cell.GREEN
    elif type == Cell.GREEN and (3 < ngreen and ngreen < 6):
      return Cell.GREEN
    else:
      return Cell.WHITE

  def get_neighbours(self, position: Position, include_diag: bool = True) -> list[Position]:
    """
    Determines all neighbour positions of a given cell, taking into account border. By default,
    searches for neighbours in all (up to) 8 neighbouring cells. By setting ```include_diag```
    to false, only horizontal and vertical neighbours are returned.

    Arguments:
    - ```position: Position```: position from where to compute neighbours
    - ```include_diag: bool```: include diagonal neighbours

    Returns: a list of ```Position``` objects corresponding to the cell neighbours
    """
    i, j = position

    if i < 0: i = self.rows + i
    if j < 0: j = self.cols + j

    # vertical neighbours
    nb = [(i+1, j), (i, j+1), (i-1, j), (i, j-1)]

    # diagonal neighbours
    if include_diag:
      nb = nb + [(i+1, j+1), (i+1, j-1), (i-1, j-1), (i-1, j+1)]

    return [(k, l) for k, l in nb if (k>=0 and l>=0 and k<self.rows and l<self.cols)]

  def get_valid_moves(self, position: Position = None) -> list[Position]:
    """
    Get a list of valid moves for a cursor at the given position. That is, computes all
    movements, starting at position, that will end up in a white cell.

    Arguments:
    - ```position: Position```: cursor position

    Returns: a list of ```Position``` objects representing valid moves
    """
    if position is None:
      position = self.start

    if self.maze[position] == Cell.GREEN:
      return []
    
    nb = self.get_neighbours(position, include_diag = False)
    valid_moves = [ij for ij in nb if self.get_mutation(ij) != Cell.GREEN]
    return valid_moves
  
  def random_walk(self, position: Position, evolve: bool = False) -> Position | None:
    """
    Takes a cursor at the given position, computes all valid moves starting at such
    position, then randomly choose one of these moves and returns it. Optionally,
    can evolve the maze to the next state. If the cursor is stuck (i.e. no move
    will lead to a white cell), the function returns None.

    Arguments:
    - ```position: Position```: cursor position
    - ```evolve: bool```: maze will be evolved to the next state, if enabled

    Returns: randomly determined next position or None, if no move is available
    """
    if position is None:
      return None
    
    moves = self.get_valid_moves(position)

    if len(moves) == 0:
      return None
    
    if evolve:
      self.evolve()

    return random.choice(moves)

  def solve(self, starting_position: Position = None, max_steps: int = 100, verbose: bool = False) -> list[Position] | None:
    """
    Solve the maze by computing all possible paths for a cursor starting at the given
    position that end up in the finishing cell. For that, a step-by-step evolution
    of the maze is computed. For every step (corresponding to a given cell state in the
    maze) every survival path is appended with all possible moves that end up in a white
    cell. Degenerate, different paths that lead to a same position at a given step are
    collapsed, since the future steps depend only on the current cell state and cursor
    position. Once the first path leading to the finishing cell is found, the function
    returns such path. If after ```max_steps``` no such path is found, the function will
    return nothing.

    Arguments:
    - ```position: Position```: starting cursor position
    - ```max_steps: int```: maximum number of steps
    - ```verbose: bool```: whether to print step-by-step information or not

    Returns: ```list[Position]```, if a path is found, otherwise returns ```None```
    """
    if starting_position is None:
      starting_position = self.start

    # in case the position is the end tile (just in case)
    if self.maze[starting_position] == Cell.FINISH:
      return True

    mz = self.clone()

    survival_paths = [[(starting_position)]]

    for i in range(1, max_steps):
      next_survival_paths = []
      ending_points = []
      for path in survival_paths:
        for move in mz.get_valid_moves(path[-1]):
          if move == self.finish:
            return path + [move]
          if not move in ending_points:
            next_survival_paths.append(path + [move])
            ending_points.append(move)
      survival_paths = next_survival_paths
      mz.evolve()

      if verbose:
        print(f'step: {i-1}, paths: {len(survival_paths)}')

    return None
  
  def solve_random(self, position: Position = None, tries: int = 100) -> list[Position]:
    """
    Naive approach to solving a maze: make a cursor walk randomly from the starting position,
    but taking into account only moves that will lead to a white cell. Eventually, if the cursor
    end up in the finishing cell, a path is returned. If, after ```tries``` attempts, no path
    leads to the finishing cell, nothing is returned.

    Arguments:
    - ```position: Position```: starting cursor position
    - ```tries: int```: maximum number of tries

    Returns: ```list[Position]```, if a path is found, otherwise returns ```None```
    """
    if position is None:
      position = self.start

    mazes = [self]

    for itry in range(tries):
      path = [position]

      while not path[-1] is None:
        istep = len(path)

        if len(mazes) < istep:
          mazes.append(mazes[-1].clone())
          mazes[-1].evolve()

        cur_maze = mazes[istep - 1]
        cur_step = cur_maze.random_walk(path[-1])
        path.append(cur_step)

        if cur_step == self.finish:
          return path
    
    return None