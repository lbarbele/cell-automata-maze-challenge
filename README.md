# Overview

This is a solution to the [Stone Automata Maze Challenge](https://sigmageek.com/challenge/stone-automata-maze-challenge) posted on SigmaGeek. This python code provides a framework to solve a rectangular maze of arbitrary dimension and of arbitrary cell dynamics.

# Notebooks

Actual solutions to the challenge (split into two questions) are given in the notebooks (under the ```notebooks```) folder:

- ```challenge-part-one.ipynb```: solves Question 1, a maze of 7 rows and 8 columns whose dynamics I inferred from solving the game.
- ```challenge-part-two.ipynb```: solves Question 2, a maze 65 rows and 85 columns (read from ```data/input.txt```) whose dynamics are described in the question. The solution is written to ```data/output.txt```.

# Library

The code responsible for describing programatically and solving the mazes is written under the ```lib``` directory. In particular, the ```Maze``` class (defined in ```lib/maze.py```) wraps all methods to generate a maze, describe its dynamics, and find a path of white cells connecting the start cell to the finishing cell. The file ```lib/cell.py``` enumerates the four cell types, mapping them into the integer codes required by the challenge.

# Solution algorithms

Two algorithms are provided to solve a maze:

- method ```Maze.solve_random```: generates $n$ random paths that go through white cells until no more movements are possible or the finishing cell is reached. If a path reaching the destination cell is found, the path is returned as a ```list``` of ```(i, j)``` tuples, describing the sequence of cells forming the path. Otherwise, if no path is found, nothing is returned. This works fine for the first question, since the maze is small.
- method ```Maze.solve```: generates all possible paths of up to ```max_steps``` steps until a path to the destination cell is found. Since the iteration is made in the number of steps, the solution is always the shortest path. To improve efficiency, degenerate paths (i.e. paths that lead to a same cell in a same number of steps) are collapsed into one single path. If no solution exists for ```max_steps```, nothing is returned.