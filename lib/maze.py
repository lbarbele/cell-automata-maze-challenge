from __future__ import annotations
import math
import random
import typing

import matplotlib.animation as anm
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import numpy as np

from .cell import Cell

Position = tuple[int, int]

maze_cmap = clr.ListedColormap([
  [1.        , 1.        , 1.        , 1.        ],
  [0.36862745, 0.93333333, 0.38823529, 1.        ],
  [0.36862745, 0.93333333, 0.38823529, 1.        ],
  [0.36862745, 0.38039216, 0.94901961, 1.        ],
  [0.88627451, 0.32156863, 0.20784314, 1.        ]
])

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

    # check cells and search for start and finishing cells
    self.start = None
    self.finish = None

    for i, j in  zip(*np.where((self.maze != Cell.WHITE) & (self.maze != Cell.GREEN))):
      if self.maze[i, j] == Cell.START and self.start is None:
        self.start = (i, j)
      elif self.maze[i, j] == Cell.FINISH and self.finish is None:
        self.finish = (i, j)
      else:
        raise RuntimeError(f'bad cell at ({i}, {j})')

    # check if start and finish were found
    if self.start is None or self.finish is None:
      raise RuntimeError('input matrix is incomplete: missing start or finish tiles')
    
    # set the cell dynamics to be the default one (can be changed using set_dynamics)
    self.dynamics = Maze.__default_dynamics
    
  @staticmethod
  def __default_dynamics(ngreen, cell_type):
    # start/finish cells do not change
    if cell_type == Cell.START or cell_type == Cell.FINISH:
      return cell_type
    
    # dynamics of green and white cells are defined here:
    if cell_type == Cell.WHITE and (1 < ngreen and ngreen < 5):
      # white cells turn green if they have a number of adjacent green cells
      # greater than 1 and less than 5. Otherwise, they remain white.
      return Cell.GREEN
    elif cell_type == Cell.GREEN and (3 < ngreen and ngreen < 6):
      # green cells remain green if they have a number of green adjacent cells
      # greater than 3 and less than 6. Otherwise they become white.
      return Cell.GREEN
    else:
      # all other cells become or remain white
      return Cell.WHITE
    
  def __validate_position(self, position: Position) -> Position:
    # check if position is a tuple of values
    if not isinstance(position, tuple) or len(position) != 2:
      raise RuntimeError('invalid position object')

    i, j = position

    # consider negative indices
    if i < 0: i = self.rows + i
    if j < 0: j = self.cols + j

    # check if position is out of bounds
    if i < 0 or i >= self.rows or j < 0 or j >= self.cols:
      raise RuntimeError(f'position {position} is not within the maze')
    
    return (i, j)
  
  def __validate_cell(self, i: int) -> Cell:
    if i != Cell.START and i != Cell.FINISH and i != Cell.WHITE and i != Cell.GREEN:
      raise RuntimeError(f'invalid cell type {i}')
    
    return i

  def clone(self) -> Maze:
    """
    Returns a deep clone of this maze.

    Arguments: none
    
    Returns: a ```Maze``` object, clone if this maze
    """
    other = Maze(self.maze)
    other.set_dynamics(self.dynamics)
    return other
  
  def count_green_neighbours(self, position: Position) -> int:
    """
    Counts the number of green cells among the direct neighbours of the cell at the
    given position.

    Arguments:
    - ```position: tuple[int, int]```: position of the cell around which the number of green cells will
    be computed

    Returns: an integer corresponding to the count of green neighbours.
    """
    # note: input position is validated within get_neighbours, no need to check here

    count = 0

    for ij in self.get_neighbours(position):
      if self.maze[ij] == Cell.GREEN:
        count += 1
    
    return count

  def draw(self, position: Position = None, ms: int = 20, fignum: int = None) -> None:
    """
    Draw the maze using ```matplotlib```'s ```matshow``` method. If a position is given
    as argument, a cursor is drawn at such position and all possible valid moves are
    marked in the grid.

    Arguments:
    - ```position: tuple[int, int]```: cursor position
    - ```ms: int```: cursor size
    - ```fignum: int or None```: forwarded to ```matshow```

    Returns: a ```matplotlib.image.AxesImage``` object (returned from ```plt.matshow```)
    """

    # draw the maze's cell matrix using pyplot's matshow method
    axes_image = plt.matshow(self.maze, cmap = maze_cmap, fignum = fignum)

    # if a cursor position is given, draw it
    if not (position is None):
      # validate the position
      pos = self.__validate_position(position)

      # draw the cursor as a simple marker at its position
      plt.plot(pos[1], pos[0], 'kD', ms = ms)

      # compute valid moves and draw them
      moves = np.array(self.get_valid_moves(position))

      if len(moves) > 0:
        plt.plot(moves[:, 1], moves[:, 0], 'k+', ms = ms)

    # forward the object returned by matshow
    return axes_image

  def draw_animation(self, path: list[Position] = None, steps: int = None, ms: int = 20, interval: int = 200) -> anm.FuncAnimation:
    """
    Draw an animation of the maze in ```steps``` steps or wit ha cursor path, if
    ```path``` is given.

    Arguments:
    - ```path: list[Position]```: path to be animated
    - ```steps: int```: number of steps to animate
    - ```ms: int```: marker size for drawing cursor positions
    - ```interval: int```: interval between frames

    Returns: ```matplotlib.animation.FuncAnimation``` object
    """
    if path is None and steps is None:
      raise RuntimeError('at least one of steps or path must be given')
    
    steps = len(path) if path is not None else steps
    mzplt = self.clone()
    figure = plt.figure()

    def animate(i):
      pos = None if path is None else path[i]
      if i > 0: mzplt.evolve()
      mzplt.draw(position = pos, fignum = figure.number, ms = ms)

    animation = anm.FuncAnimation(
      figure,
      func = animate,
      frames = range(steps),
      repeat = False,
      interval = interval,
    )

    return animation
  
  def evolve(self, overwrite: bool = True) -> Maze:
    """
    Apply cell dynamics to every cell in the maze. By default, this maze is overwritten
    and returned by the method. If the parameter ```overwrite``` is disabled, the maze
    is not modified and the returned value is a new Maze after the cell evolution.

    Arguments:
    - ```overwrite: bool```: enable/disable overwrite of this Maze

    Returns: ```self``` if overwrite is enabled or a new ```Maze``` otherwise.
    """

    # start with a maze of white cells
    other = np.full(self.maze.shape, Cell.WHITE, dtype = int)

    # apply dynamics to every cell
    for pos in np.ndindex(self.shape):
      other[pos] = self.get_mutation(pos)

    if overwrite == True:
      # if overwrite is enabled, change this maze and return it
      self.maze = other
      return self
    elif overwrite == False:
      # if overwrite is disabled, return a new maze with the dynamics applied
      return Maze(other)
    else:
      # if overwrite is invalid
      raise RuntimeError('invalid overwrite parameter value')

  @staticmethod
  def from_file(path: str) -> Maze:
    """
    Create maze by reading a file containing cell information.

    Arguments:
    - ```path: str```: matrix of cells in text format (only numbers)

    Returns: ```Maze```
    """
    matrix = np.loadtxt(path, dtype = int)
    return Maze(matrix)

  def get_cell_indices(self, cell_type: Cell) -> list[Position]:
    """
    Find all positions of cells of given type.

    Arguments:
    - ```cell_type: Cell```: type of cell to be found

    Returns: a list of ```Position``` containing the positions of the given cell type.
    """
    ct = self.__validate_cell(cell_type)
    return [p for p, c in np.ndenumerate(self.maze) if c == ct]
  
  def get_mutation(self, position: Position) -> Cell:
    """
    Apply cell dynamics to determine how the cell at the given position will evolve.

    Arguments:
    - ```position: Position```: position of the cell of interest

    Returns: a ```Cell``` corresponding to the cell evolution
    """

    # note: position will be validated in count_green_neighbours
    ngreen = self.count_green_neighbours(position)
    cell_type = self.maze[position]

    # return value given by dynamics
    return self.dynamics(ngreen = ngreen, cell_type = cell_type)

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

    i, j = self.__validate_position(position)

    if include_diag:
      all_neighbours = [(i+1, j), (i, j+1), (i-1, j), (i, j-1), (i+1, j+1), (i+1, j-1), (i-1, j-1), (i-1, j+1)]
    else:
      all_neighbours = [(i+1, j), (i, j+1), (i-1, j), (i, j-1)]

    neighbours = []

    for k, l in all_neighbours:
      if k >= 0 and l >= 0 and k < self.rows and l < self.cols:
        neighbours += [(k, l)]

    return neighbours

  def get_valid_moves(self, position: Position = None) -> list[Position]:
    """
    Get a list of valid moves for a cursor at the given position. That is, computes all
    movements, starting at position, that will end up in a white cell.

    Arguments:
    - ```position: Position```: cursor position

    Returns: a list of ```Position``` objects representing valid moves
    """
    position = self.start if position is None else self.__validate_position(position)

    # if cursor is above a green cell, no moves are allowed
    if self.maze[position] == Cell.GREEN:
      return []
    
    # get neighbour cells
    nb = self.get_neighbours(position, include_diag = False)

    # valid moves are neighbour cells that will not evolve to green
    valid_moves = [p for p in nb if self.get_mutation(p) != Cell.GREEN]

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
    # note: position will be validated in get_valid_moves

    moves = self.get_valid_moves(position)

    if len(moves) == 0:
      return None
    
    if evolve == True:
      self.evolve()

    return random.choice(moves)
  
  def set_dynamics(self, dynamics_function: typing.Callable):
    """
    Function to define cell dynamics. The given dynamics function must be a callable
    compatible with the signature ```dynamics_function(ngreen: int, cell_type: Cell) -> Cell```.

    Arguments:
    - ```dynamics_functions: callable```: the new function describing the cell dynamics for this maze

    Returns nothing.
    """
    for ct in [Cell.START, Cell.FINISH, Cell.WHITE, Cell.GREEN]:
      for ng in range(9):
        try:
          ret = dynamics_function(ngreen = ng, cell_type = ct)
          self.__validate_cell(ret)
        except:
          raise RuntimeError('given dynamics function is invalid')

    self.dynamics = dynamics_function

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

    # assign initial position to start cell, if not given, otherwise validate it
    starting_position = self.start if starting_position is None else self.__validate_position(starting_position)

    # in case the position is already the end cell
    if self.maze[starting_position] == Cell.FINISH:
      return [starting_position]

    # get a clone of this maze, so this maze will not be modified
    mz = self.clone()

    # this is a list of all survival paths at a given step
    # that is, paths that go only throgh white cells
    survival_paths = [[(starting_position)]]

    # iterate over steps (starting at 1 because step 0 is the starting position)
    for i in range(1, max_steps):
      # list of survival paths after current step
      next_survival_paths = []

      # set of possible ending cells of all paths after the current step
      ending_points = []

      # iterate over all survival paths before the current step
      # for each path, compute every possible move from its last position.
      # if the move leads to a finishing cell, then we are done: return the path
      # otherwise, add path + move to the list of survival paths if path + move
      # leads to a cell that no other path does (because we are not interested
      # in different paths that lead to the same position at the same step)
      for path in survival_paths:
        for move in mz.get_valid_moves(path[-1]):
          if move == self.finish:
            # path to the ending cell found
            return path + [move]
          elif not move in ending_points:
            # path to new white cell found
            next_survival_paths.append(path + [move])
            ending_points.append(move)

      # update the list of survival paths and evolve the maze
      survival_paths = next_survival_paths
      mz.evolve()

      # dump information if verbose flag is enabled
      if verbose:
        shortest_dist = None
        for path in survival_paths:
          cell = path[-1]
          dist = math.hypot(cell[0] - self.finish[0], cell[1] - self.finish[1])
          if shortest_dist is None or dist < shortest_dist:
            shortest_dist = dist
        print(f'step: {i-1}, paths: {len(survival_paths)}, shortest distance to finish: {shortest_dist}')

    # in case no path was found
    return None
  
  def solve_random(self, position: Position = None, tries: int = 100, verbose: bool = False) -> list[Position]:
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

    # assign initial position to start cell, if not given, otherwise validate it
    position = self.start if position is None else self.__validate_position(position)

    # in case the position is already the end cell
    if self.maze[position] == Cell.FINISH:
      return [position]

    # get a clone of this maze, so this maze will not be modified
    mz = self.clone()

    # this is a list of all random paths at a given step, except paths
    # for which no moves are available
    random_paths = [[position] for i in range(tries)]

    # iterate over walk steps until a solution is found or no more paths are available
    istep = 1

    while len(random_paths) > 0:
      # list of random paths after current step
      next_random_paths = []

      # iterate over all random paths before the current step
      # for each path, compute a random move from its last position.
      # if the move leads to a finishing cell, then we are done: return the path
      # otherwise, add path + move to the list of survival paths if path + move
      for path in random_paths:
        move = mz.random_walk(path[-1])
        if move == self.finish:
          # path to the ending cell found
          return path + [move]
        elif move is None:
          # no moves available from path
          continue
        else:
          # random path to white cell exists
          next_random_paths.append(path + [move])

      # update the list of random paths and evolve the maze
      random_paths = next_random_paths
      mz.evolve()

      # dump information if verbose flag is enabled
      if verbose:
        shortest_dist = None
        for path in random_paths:
          cell = path[-1]
          dist = math.hypot(cell[0] - self.finish[0], cell[1] - self.finish[1])
          if shortest_dist is None or dist < shortest_dist:
            shortest_dist = dist
        print(f'step: {istep}, paths: {len(random_paths)}, shortest distance to finish: {shortest_dist}')
        istep += 1

    # in case no path was found
    return None