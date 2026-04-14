/*MIT License

Copyright(c) 2020 Thibaut Vidal

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files(the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions :

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.*/

#ifndef COMMAND_LINE_H
#define COMMAND_LINE_H

#include <iostream>
#include <string>

class Commandline
{
public:

	std::string instance_name;		// Instance path
	std::string output_name;		// Output path
	bool command_ok;				// Boolean to check if the command line is valid
	int nbTrees;					// Hard limit on the number of trees (defaults to the number of trees from the input data)
	int objectiveFunction;			// 0 = Depth ; 1 = NbLeaves ; 2 = Depth then NbLeaves ; 3 = NbLeaves then Depth (not implemented) ; 4 = Heuristic ; 5 = A* NbLeaves ; 6 = GreedyExactCells ; 7 = BeamSearchExactCells
	int seed;						// Random seed (only impacts the heuristic / sampling-based -obj 4 path)
	int beamWidth;					// Beam width for objective 7 (default 5); values <= 1 delegate to greedy construction
	int beamHeuristic;				// For objective 7: 0 = default region/split scoring ; 1 = lookahead-style region priority ; 2 = balance / depth-aware (see README_GREEDY_BEAM.md)

	// argv[0]=program, argv[1]=instance, argv[2]=output, then optional flag/value pairs.
	Commandline(int argc, char** argv)
	{
		if (argc < 3 || (argc - 3) % 2 != 0)
		{
			std::cout << "ISSUE WITH THE NUMBER OF COMMANDLINE ARGUMENTS: " << argc
			          << " (expected: program instance output [flag value]...)" << std::endl;
			command_ok = false;
		}
		else
		{
			command_ok = true;
			instance_name = std::string(argv[1]);
			output_name = std::string(argv[2]);
			nbTrees = 10;
			objectiveFunction = 4;
			seed = 1;
			beamWidth = 5;
			beamHeuristic = 0;
			for (int i = 3; i < argc; i += 2)
			{
				if (i + 1 >= argc)
				{
					std::cout << "ISSUE WITH COMMANDLINE: missing value after " << std::string(argv[i]) << std::endl;
					command_ok = false;
					break;
				}
				if (std::string(argv[i]) == "-trees")
					nbTrees = atoi(argv[i + 1]);
				else if (std::string(argv[i]) == "-obj")
					objectiveFunction = atoi(argv[i + 1]);
				else if (std::string(argv[i]) == "-seed")
					seed = atoi(argv[i + 1]);
				else if (std::string(argv[i]) == "-beam")
					beamWidth = atoi(argv[i + 1]);
				else if (std::string(argv[i]) == "-bh")
					beamHeuristic = atoi(argv[i + 1]);
				else
				{
					std::cout << "----- NON RECOGNIZED ARGUMENT: " << std::string(argv[i]) << std::endl;
					command_ok = false;
				}
			}
		}
	}
};
#endif
