#include "BornAgainDecisionTree.h"
#include <queue>
#include <utility>
#include <functional>
#include <algorithm>

unsigned int BornAgainDecisionTree::dynamicProgrammingOptimizeDepth(int indexBottom, int indexTop)
{
	iterationsDP++;
	if (indexBottom == indexTop) return 0;

	int hash = fspaceFinal.keyToHash(indexBottom, indexTop);
	if (regions[indexBottom][hash] != UINT_MAX) return regions[indexBottom][hash];

	unsigned int bestLB = 0;
	unsigned int bestUB = UINT_MAX;
	for (int k = 0; k < params->nbFeatures && bestLB < bestUB; k++)
	{
		const int codeBookValue = fspaceFinal.codeBook[k];
		const int rangeLow = fspaceFinal.keyToCell(indexBottom, k);
		const int rangeUp = fspaceFinal.keyToCell(indexTop, k);
		int tempRangeLow = rangeLow;
		int tempRangeUp = rangeUp;
		while (tempRangeLow < tempRangeUp && bestLB < bestUB)
		{
			int l = (tempRangeLow+tempRangeUp) / 2;
			unsigned int leftResult = dynamicProgrammingOptimizeDepth(indexBottom, indexTop + codeBookValue * (l - rangeUp)); // Index of the new corner: z^R + e_j(l-z^R_j)
			if (leftResult > bestLB) bestLB = leftResult;

			// Small code optimization: We can stop this recursion earlier for large values of leftResult
			if (1 + leftResult >= bestUB) 
				tempRangeUp = l;
			else
			{
				unsigned int rightResult = dynamicProgrammingOptimizeDepth(indexBottom + codeBookValue * (l + 1 - rangeLow), indexTop); // Index of the new corner: z^L + e_j(l+1-z^L_j)
				if (leftResult == 0 && rightResult == 0) // Base case has been attained
				{
					if (fspaceFinal.cells[indexBottom] == fspaceFinal.cells[indexTop])
					{
						regions[indexBottom][hash] = 0;
						regionsMemorizedDP++;
						return 0;
					}
					else
					{
						regions[indexBottom][hash] = 1;
						regionsMemorizedDP++;
						return 1;
					}
				}
				if (rightResult > bestLB)     bestLB = rightResult;
				if (1 + rightResult < bestUB) bestUB = 1 + std::max<unsigned int>(leftResult, rightResult);
				if (1 + leftResult  >= bestUB) tempRangeUp = l; // Left result will only increase to the right (Theorem 4)
				if (1 + rightResult >= bestUB) tempRangeLow = l + 1; // Right result will only increase to the left (Theorem 4)
			}
		}
	}
	regions[indexBottom][hash] = bestUB;
	regionsMemorizedDP++;
	return bestUB;
}

unsigned int BornAgainDecisionTree::dynamicProgrammingOptimizeNbSplits(int indexBottom, int indexTop)
{
	iterationsDP++;
	if (indexBottom == indexTop) return 0;

	int hash = fspaceFinal.keyToHash(indexBottom, indexTop);
	if (regions[indexBottom][hash] != UINT_MAX) return regions[indexBottom][hash];

	unsigned int bestLB = 0;
	unsigned int bestUB = UINT_MAX;
	for (int k = 0; k < params->nbFeatures && bestLB < bestUB; k++)
	{
		const int codeBookValue = fspaceFinal.codeBook[k];
		const int rangeLow = fspaceFinal.keyToCell(indexBottom, k);
		const int rangeUp = fspaceFinal.keyToCell(indexTop, k);
		for (int l = rangeLow ; l < rangeUp && bestLB < bestUB; l++)
		{
			unsigned int leftResult = dynamicProgrammingOptimizeNbSplits(indexBottom, indexTop + codeBookValue * (l - rangeUp));          // Index of the new corner: z^R + e_j(l-z^R_j)
			unsigned int rightResult = dynamicProgrammingOptimizeNbSplits(indexBottom + codeBookValue * (l + 1 - rangeLow), indexTop);    // Index of the new corner: z^L + e_j(l+1-z^L_j)
			if (leftResult == 0 && rightResult == 0) // Base case has been attained
			{
				if (fspaceFinal.cells[indexBottom] == fspaceFinal.cells[indexTop])
				{
					regions[indexBottom][hash] = 0 ;
					regionsMemorizedDP++;
					return 0;
				}
				else
				{
					regions[indexBottom][hash] = 1 ;
					regionsMemorizedDP++;
					return 1;
				}
			}
			else
			{
				if (leftResult  > bestLB) bestLB = leftResult;
				if (rightResult > bestLB) bestLB = rightResult;
				if (1 + rightResult + leftResult < bestUB) bestUB = 1 + leftResult + rightResult;
			}
		}
	}
	regions[indexBottom][hash] = bestUB ;
	regionsMemorizedDP++;
	return bestUB;
}

unsigned int BornAgainDecisionTree::dynamicProgrammingOptimizeDepthThenNbSplits(int indexBottom, int indexTop)
{
	iterationsDP++;
	if (indexBottom == indexTop) return 0;

	int hash = fspaceFinal.keyToHash(indexBottom, indexTop);
	if (regions[indexBottom][hash] != UINT_MAX) return regions[indexBottom][hash];

	unsigned int bestLB = 0;
	unsigned int bestUB = UINT_MAX;
	for (int k = 0; k < params->nbFeatures && bestLB < bestUB; k++)
	{
		const int codeBookValue = fspaceFinal.codeBook[k];
		const int rangeLow = fspaceFinal.keyToCell(indexBottom, k);
		const int rangeUp = fspaceFinal.keyToCell(indexTop, k);
		for (int l = rangeLow; l < rangeUp && bestLB < bestUB; l++)
		{
			unsigned int leftResult = dynamicProgrammingOptimizeDepthThenNbSplits(indexBottom, indexTop + codeBookValue * (l - rangeUp));          // Index of the new corner: z^R + e_j(l-z^R_j)
			unsigned int rightResult = dynamicProgrammingOptimizeDepthThenNbSplits(indexBottom + codeBookValue * (l + 1 - rangeLow), indexTop);    // Index of the new corner: z^L + e_j(l+1-z^L_j)
			if (leftResult == 0 && rightResult == 0) // Base case has been attained
			{
				if (fspaceFinal.cells[indexBottom] == fspaceFinal.cells[indexTop])
				{
					regions[indexBottom][hash] = 0 ;
					regionsMemorizedDP++;
					return 0;
				}
				else
				{
					regions[indexBottom][hash] = BIG_M + 1;
					regionsMemorizedDP++;
					return BIG_M + 1;
				}
			}
			else
			{
				if (leftResult > bestLB)  bestLB = leftResult;
				if (rightResult > bestLB) bestLB = rightResult;
				unsigned int newResult = BIG_M + 1 + BIG_M * std::max<unsigned int>(leftResult/BIG_M,rightResult/BIG_M) + leftResult % BIG_M + rightResult % BIG_M;
				if (newResult < bestUB) bestUB = newResult;
			}
		}
	}
	regions[indexBottom][hash] = bestUB;
	regionsMemorizedDP++;
	return bestUB;
}

int BornAgainDecisionTree::collectResultDP(int indexBottom, int indexTop, unsigned int optValue, unsigned int currentDepth)
{
	if (optValue == 0) 
	{
		finalLeaves++;
		if (currentDepth > finalDepth) finalDepth = currentDepth;
		rebornTree.push_back(Node());
		int nodeID = (int)rebornTree.size()-1;
		rebornTree[nodeID].nodeType = Node::NODE_LEAF;
		rebornTree[nodeID].splitFeature = -1;
		rebornTree[nodeID].splitValue = -1;
		rebornTree[nodeID].classification = fspaceFinal.cells[indexBottom];
		rebornTree[nodeID].nodeID = nodeID;
		rebornTree[nodeID].depth = currentDepth;
		return nodeID;
	}
	else
	{
		// Verify which pair of subproblems was used and calling recursion on each subproblem
		for (int k = 0; k < params->nbFeatures ; k++)
		{
			const int codeBookValue = fspaceFinal.codeBook[k];
			const int rangeLow = fspaceFinal.keyToCell(indexBottom, k);
			const int rangeUp = fspaceFinal.keyToCell(indexTop, k);
			for (int l = rangeLow; l < rangeUp ; l++)
			{
				unsigned int leftResult;
				int indexTopLeft = indexTop + codeBookValue * (l - rangeUp);
				int hash1 = fspaceFinal.keyToHash(indexBottom, indexTopLeft);
				if (indexBottom == indexTopLeft) leftResult = 0; 
				else leftResult = regions[indexBottom][hash1];

				unsigned int rightResult;
				int indexBottomRight = indexBottom + codeBookValue * (l + 1 - rangeLow);
				int hash2 = fspaceFinal.keyToHash(indexBottomRight, indexTop);
				if (indexBottomRight == indexTop) rightResult = 0; 
				else rightResult = regions[indexBottomRight][hash2];

				// Found the optimal split used by the DP
				if (leftResult != UINT_MAX && rightResult != UINT_MAX &&
				   ((params->objectiveFunction == 0 && 1 + std::max<unsigned int>(leftResult, rightResult) == optValue) || 
					(params->objectiveFunction == 1 && 1 + leftResult + rightResult == optValue) ||
					(params->objectiveFunction == 2 && BIG_M + 1 + BIG_M * std::max<unsigned int>(leftResult/BIG_M,rightResult/BIG_M) + leftResult%BIG_M + rightResult%BIG_M == optValue)))
				{
					finalSplits++;
					rebornTree.push_back(Node());
					int nodeID = (int)rebornTree.size() - 1;
					rebornTree[nodeID].nodeType = Node::NODE_INTERNAL;
					rebornTree[nodeID].splitFeature = k;
					rebornTree[nodeID].splitValue = fspaceFinal.orderedHyperplaneLevels[k][l];
					rebornTree[nodeID].nodeID = nodeID;
					int myLeftID =  collectResultDP(indexBottom, indexTopLeft, leftResult, currentDepth + 1);
					int myRightID = collectResultDP(indexBottomRight, indexTop, rightResult, currentDepth + 1);
					rebornTree[nodeID].leftChild = myLeftID;
					rebornTree[nodeID].rightChild = myRightID;
					rebornTree[nodeID].depth = currentDepth;
					return nodeID;
				}
			}
		}
		// Just for safety: This point is never attained since the optimal result is found from one of the subproblems
		throw std::string("Error extraction DP results");
	}
}

void BornAgainDecisionTree::buildOptimal()
{
	iterationsDP = 0;
	regionsMemorizedDP = 0;
	finalSplits = 0;
	finalLeaves = 0;
	finalDepth = 0;

	// Initialize the cells structures and keep useful hyperplanes
	fspaceOriginal.initializeCells(randomForest->getHyperplanes(),false);
	fspaceFinal.initializeCells(fspaceOriginal.exportUsefulHyperplanes(),true);
	
	// Initialize the memory to store the DP results on the regions
	regions = std::vector<std::vector<unsigned int>>(fspaceFinal.nbCells);
	for (int index = 0; index < fspaceFinal.nbCells; index++)
		regions[index] = std::vector<unsigned int>(fspaceFinal.keyToHash(index,fspaceFinal.nbCells-1)+1,UINT_MAX);

	// Launch optimization algorithm
	// std::cout << "----- START OF OPTIMIZATION " << std::endl;
	if (params->objectiveFunction == 0)
	{
		finalObjective = dynamicProgrammingOptimizeDepth(0, (int)fspaceFinal.nbCells - 1);
		collectResultDP(0, (int)fspaceFinal.nbCells - 1, finalObjective, 0);
	}
	else if (params->objectiveFunction == 1)
	{
		finalObjective = dynamicProgrammingOptimizeNbSplits(0, (int)fspaceFinal.nbCells - 1);
		collectResultDP(0, (int)fspaceFinal.nbCells - 1, finalObjective, 0);
	}
	else if (params->objectiveFunction == 2)
	{
		finalObjective = dynamicProgrammingOptimizeDepthThenNbSplits(0, (int)fspaceFinal.nbCells - 1);
		collectResultDP(0, (int)fspaceFinal.nbCells - 1, finalObjective, 0);
	}
	else if (params->objectiveFunction == 5)
	{
		// Use A* to compute an optimal objective value w.r.t. number of splits,
		// then rely on the DP memory + collectResultDP to reconstruct the tree.
		finalObjective = aStarOptimizeNbSplits();

		// Rebuild the DP table for objective 1 to extract the structure of an optimal tree.
		iterationsDP = 0;
		regionsMemorizedDP = 0;
		regions = std::vector<std::vector<unsigned int>>(fspaceFinal.nbCells);
		for (int index = 0; index < fspaceFinal.nbCells; index++)
			regions[index] = std::vector<unsigned int>(fspaceFinal.keyToHash(index, fspaceFinal.nbCells - 1) + 1, UINT_MAX);

		unsigned int dpObjective = dynamicProgrammingOptimizeNbSplits(0, (int)fspaceFinal.nbCells - 1);
		if (dpObjective != finalObjective)
			throw std::string("Inconsistent objectives between A* and DP");
		collectResultDP(0, (int)fspaceFinal.nbCells - 1, dpObjective, 0);
	}
	else
	{
		throw std::string("NON RECOGNIZED OBJECTIVE");
	}
}

void BornAgainDecisionTree::displayRunStatistics()
{
	std::vector<std::string> objectives = {
		"Depth",
		"NbLeaves",
		"Depth then NbLeaves",
		"NbLeaves then Depth",
		"Heuristic",
		"A* NbLeaves",
		"GreedyExactCells",
		"BeamSearchExactCells"
	};
	std::cout << "----- OPTIMAL SOLUTION FOUND                      " << std::endl;
	std::cout << "----- OBJECTIVE:                                  " << objectives[params->objectiveFunction] << std::endl;
	std::cout << "----- CPU TIME(s):                                " << (double)(params->stopTime - params->startTime) / (double)CLOCKS_PER_SEC << std::endl;
	std::cout << "----- ORIGINAL CELLS:                             " << fspaceOriginal.nbCells << std::endl;
	std::cout << "----- FILTERED CELLS:                             " << fspaceFinal.nbCells << std::endl;
	std::cout << "----- NB SUBPROBLEMS:                             " << (double)regionsMemorizedDP << std::endl;
	std::cout << "----- NB RECURSIVE CALLS:                         " << (double)iterationsDP << std::endl;
	std::cout << "----- BA TREE DEPTH:                              " << finalDepth << std::endl;
	std::cout << "----- BA TREE LEAVES:                             " << finalLeaves << std::endl;
}

void BornAgainDecisionTree::exportRunStatistics(std::string fileName)
{
	std::ofstream myfile;
	std::cout << "----- EXPORTING STATISTICS in " << fileName << std::endl;
	myfile.open(fileName.data());
	if (myfile.is_open())
	{
		myfile << params->datasetName << ",";
		myfile << params->ensembleMethod << ",";
		myfile << params->nbTrees << ",";
		myfile << params->nbFeatures << ",";
		myfile << params->nbClasses << ",";
		myfile << params->objectiveFunction << ",";
		myfile << finalDepth << ",";
		myfile << finalSplits << ",";
		myfile << finalLeaves << ",";
		myfile << 1 << ","; // Only one execution loop has been done
		myfile << (double)(params->stopTime - params->startTime) / (double)CLOCKS_PER_SEC << ",";
		myfile << fspaceFinal.nbCells << ",";
		myfile << regionsMemorizedDP << ",";
		myfile << iterationsDP << std::endl;
		myfile.close();
	}
	else
		std::cout << "PROBLEM OPENING FILE " << fileName << std::endl;
}

void BornAgainDecisionTree::exportBATree(std::string fileName)
{
	std::ofstream myfile;
	std::cout << "----- EXPORTING BA TREE in " << fileName << std::endl;
	myfile.open(fileName.data());
	if (myfile.is_open())
	{
		myfile << "DATASET_NAME: " << params->datasetName << std::endl;
		myfile << "ENSEMBLE: BA" << std::endl;
		myfile << "NB_TREES: " << 1 << std::endl;
		myfile << "NB_FEATURES: " << params->nbFeatures << std::endl;
		myfile << "NB_CLASSES: " << params->nbClasses << std::endl;
		myfile << "MAX_TREE_DEPTH: " << finalDepth << std::endl;
		myfile << "Format: node / node type(LN - leave node, IN - internal node) left child / right child / feature / threshold / node_depth / majority class (starts with index 0)" << std::endl;
		myfile << std::endl;
		myfile << "[TREE 0]" << std::endl;
		myfile << "NB_NODES: " << finalSplits+ finalLeaves << std::endl;

		for (int i = 0; i < (int)rebornTree.size(); i++)
		{
			if (rebornTree[i].nodeType == Node::NODE_INTERNAL)
				myfile << i << " IN " << rebornTree[i].leftChild << " " << rebornTree[i].rightChild << " " << rebornTree[i].splitFeature << " " << rebornTree[i].splitValue << " " << rebornTree[i].depth << " -1" << std::endl;
			else if (rebornTree[i].nodeType == Node::NODE_LEAF)
				myfile << i << " LN -1 -1 -1 -1 " << rebornTree[i].depth << " " << rebornTree[i].classification << std::endl;
			else throw std::string("Error: Unexpected node type");
		}
	}
	else
		std::cout << "PROBLEM OPENING FILE  " << fileName << std::endl;
}

void BornAgainDecisionTree::computeClassCountsRegion(int indexBottom, int indexTop, std::vector<int> & counts)
{
	std::fill(counts.begin(), counts.end(), 0);

	std::vector<int> low(params->nbFeatures);
	std::vector<int> up(params->nbFeatures);
	for (int k = 0; k < params->nbFeatures; k++)
	{
		low[k] = fspaceFinal.keyToCell(indexBottom, k);
		up[k] = fspaceFinal.keyToCell(indexTop, k);
	}

	std::function<void(int,int)> recurse = [&](int k, int keyPrefix)
	{
		if (k == params->nbFeatures)
		{
			int cls = fspaceFinal.cells[keyPrefix];
			if (cls >= 0 && cls < params->nbClasses)
				counts[cls]++;
			return;
		}
		for (int i = low[k]; i <= up[k]; i++)
		{
			int nextKey = keyPrefix + i * fspaceFinal.codeBook[k];
			recurse(k + 1, nextKey);
		}
	};

	recurse(0, 0);
}

int BornAgainDecisionTree::greedyBuildRegion(int indexBottom, int indexTop, unsigned int currentDepth)
{
	std::vector<int> counts(params->nbClasses, 0);
	computeClassCountsRegion(indexBottom, indexTop, counts);

	int total = 0;
	int majorityClass = -1;
	int majorityCount = 0;
	int nonZeroClasses = 0;
	for (int c = 0; c < params->nbClasses; c++)
	{
		total += counts[c];
		if (counts[c] > 0)
		{
			nonZeroClasses++;
			if (counts[c] > majorityCount)
			{
				majorityCount = counts[c];
				majorityClass = c;
			}
		}
	}

	// If region is pure or empty, create a leaf
	if (nonZeroClasses <= 1 || total == 0)
	{
		finalLeaves++;
		if (currentDepth > finalDepth) finalDepth = currentDepth;
		rebornTree.push_back(Node());
		int nodeID = (int)rebornTree.size() - 1;
		rebornTree[nodeID].nodeType = Node::NODE_LEAF;
		rebornTree[nodeID].splitFeature = -1;
		rebornTree[nodeID].splitValue = -1;
		rebornTree[nodeID].classification = (majorityClass >= 0) ? majorityClass : 0;
		rebornTree[nodeID].nodeID = nodeID;
		rebornTree[nodeID].depth = currentDepth;
		return nodeID;
	}

	// Helper to compute Gini impurity from counts and total
	auto giniFromCounts = [&](const std::vector<int> & cnts, int tot) -> double
	{
		if (tot == 0) return 0.0;
		double sumSq = 0.0;
		for (int c = 0; c < params->nbClasses; c++)
		{
			if (cnts[c] > 0)
			{
				double p = (double)cnts[c] / (double)tot;
				sumSq += p * p;
			}
		}
		return 1.0 - sumSq;
	};

	double bestScore = 1.e30;
	int bestFeature = -1;
	int bestLevel = -1;
	int bestIndexTopLeft = -1;
	int bestIndexBottomRight = -1;

	for (int k = 0; k < params->nbFeatures; k++)
	{
		const int codeBookValue = fspaceFinal.codeBook[k];
		const int rangeLow = fspaceFinal.keyToCell(indexBottom, k);
		const int rangeUp = fspaceFinal.keyToCell(indexTop, k);
		if (rangeLow == rangeUp) continue;

		for (int l = rangeLow; l < rangeUp; l++)
		{
			int indexTopLeft = indexTop + codeBookValue * (l - rangeUp);
			int indexBottomRight = indexBottom + codeBookValue * (l + 1 - rangeLow);

			std::vector<int> leftCounts(params->nbClasses, 0);
			std::vector<int> rightCounts(params->nbClasses, 0);
			computeClassCountsRegion(indexBottom, indexTopLeft, leftCounts);
			computeClassCountsRegion(indexBottomRight, indexTop, rightCounts);

			int leftTotal = 0;
			int rightTotal = 0;
			for (int c = 0; c < params->nbClasses; c++)
			{
				leftTotal += leftCounts[c];
				rightTotal += rightCounts[c];
			}

			if (leftTotal == 0 || rightTotal == 0) continue;

			double giniLeft = giniFromCounts(leftCounts, leftTotal);
			double giniRight = giniFromCounts(rightCounts, rightTotal);
			double weighted = ((double)leftTotal * giniLeft + (double)rightTotal * giniRight) / (double)(leftTotal + rightTotal);

			if (weighted < bestScore)
			{
				bestScore = weighted;
				bestFeature = k;
				bestLevel = l;
				bestIndexTopLeft = indexTopLeft;
				bestIndexBottomRight = indexBottomRight;
			}
		}
	}

	// If no valid split found, create a majority leaf
	if (bestFeature == -1)
	{
		finalLeaves++;
		if (currentDepth > finalDepth) finalDepth = currentDepth;
		rebornTree.push_back(Node());
		int nodeID = (int)rebornTree.size() - 1;
		rebornTree[nodeID].nodeType = Node::NODE_LEAF;
		rebornTree[nodeID].splitFeature = -1;
		rebornTree[nodeID].splitValue = -1;
		rebornTree[nodeID].classification = (majorityClass >= 0) ? majorityClass : 0;
		rebornTree[nodeID].nodeID = nodeID;
		rebornTree[nodeID].depth = currentDepth;
		return nodeID;
	}

	// Apply best split and recurse
	finalSplits++;
	rebornTree.push_back(Node());
	int nodeID = (int)rebornTree.size() - 1;
	rebornTree[nodeID].nodeType = Node::NODE_INTERNAL;
	rebornTree[nodeID].splitFeature = bestFeature;
	rebornTree[nodeID].splitValue = fspaceFinal.orderedHyperplaneLevels[bestFeature][bestLevel];
	rebornTree[nodeID].nodeID = nodeID;
	rebornTree[nodeID].depth = currentDepth;

	int leftID = greedyBuildRegion(indexBottom, bestIndexTopLeft, currentDepth + 1);
	int rightID = greedyBuildRegion(bestIndexBottomRight, indexTop, currentDepth + 1);

	rebornTree[nodeID].leftChild = leftID;
	rebornTree[nodeID].rightChild = rightID;

	return nodeID;
}

void BornAgainDecisionTree::buildGreedyExact()
{
	finalSplits = 0;
	finalLeaves = 0;
	finalDepth = 0;
	iterationsDP = 0;
	regionsMemorizedDP = 0;

	// Initialize the cells structures and keep useful hyperplanes
	fspaceOriginal.initializeCells(randomForest->getHyperplanes(), false);
	fspaceFinal.initializeCells(fspaceOriginal.exportUsefulHyperplanes(), true);

	rebornTree.clear();
	greedyBuildRegion(0, (int)fspaceFinal.nbCells - 1, 0);
}

// --- BEAM SEARCH HEURISTIC HELPERS ---

// Heuristic 1: Lookahead Lower-Bound
int BornAgainDecisionTree::countRemainingSplitsLB(const BeamState & s) {
    int totalLB = 0;
    for (const auto & reg : s.regions) {
        std::vector<int> counts(params->nbClasses, 0);
        // Uses the existing method to count class occurrences in a region
        computeClassCountsRegion(reg.first, reg.second, counts); 
        
        int uniqueClasses = 0;
        for(int c : counts) {
            if(c > 0) uniqueClasses++;
        }
        // Lower bound: if a region has C classes, it needs at least C-1 more splits
        if (uniqueClasses > 1) totalLB += (uniqueClasses - 1);
    }
    return totalLB;
}

// Heuristic 2: Depth Tracker
int BornAgainDecisionTree::getMaxTreeDepth(const BeamState & s) {
    int maxD = 0;
    for (const auto & node : s.tree) {
        if (node.depth > maxD) maxD = node.depth;
    }
    return maxD;
}

void BornAgainDecisionTree::buildBeamExact()
{
	finalSplits = 0;
	finalLeaves = 0;
	finalDepth = 0;
	iterationsDP = 0;
	regionsMemorizedDP = 0;

	// Initialize the cells structures and keep useful hyperplanes
	fspaceOriginal.initializeCells(randomForest->getHyperplanes(), false);
	fspaceFinal.initializeCells(fspaceOriginal.exportUsefulHyperplanes(), true);

	auto regionImpurity = [&](int indexBottom, int indexTop) -> double
	{
		std::vector<int> counts(params->nbClasses, 0);
		computeClassCountsRegion(indexBottom, indexTop, counts);
		int total = 0;
		for (int c = 0; c < params->nbClasses; c++) total += counts[c];
		if (total == 0) return 0.0;
		double sumSq = 0.0;
		for (int c = 0; c < params->nbClasses; c++)
		{
			if (counts[c] > 0)
			{
				double p = (double)counts[c] / (double)total;
				sumSq += p * p;
			}
		}
		return 1.0 - sumSq;
	};

	// 3 Different Beam Heuristics 
	auto stateScore = [&](const BeamState & s) -> double
	{
		// Default impurity score used as a base for all heuristics
		double sc = 0.0;
		for (const auto & reg : s.regions)
			sc += regionImpurity(reg.first, reg.second);

		// Heuristic 1: Focus on Class Diversity (Lower Bound)
		if (params->beamHeuristic == 1) {
			return (double)countRemainingSplitsLB(s) + (double)s.splits;
		} 
		
		// Heuristic 2: Focus on Balance (Depth Penalty)
		if (params->beamHeuristic == 2) {
			// We add a heavy penalty for depth to avoid 'skinny' trees
			return sc + (0.5 * (double)s.splits) + (2.0 * (double)getMaxTreeDepth(s));
		}

		// Default: Impurity + Splits
		return sc + (double)s.splits;
	};

	// Initialize beam with a single root leaf covering the whole space
	BeamState root;
	root.tree.clear();
	root.regions.clear();
	root.regionNodeIDs.clear();
	root.splits = 0;

	// Create root leaf with majority class over entire space
	std::vector<int> rootCounts(params->nbClasses, 0);
	computeClassCountsRegion(0, (int)fspaceFinal.nbCells - 1, rootCounts);
	int majorityClass = 0;
	int majorityCount = -1;
	for (int c = 0; c < params->nbClasses; c++)
	{
		if (rootCounts[c] > majorityCount)
		{
			majorityCount = rootCounts[c];
			majorityClass = c;
		}
	}

	root.tree.push_back(Node());
	root.tree[0].nodeType = Node::NODE_LEAF;
	root.tree[0].splitFeature = -1;
	root.tree[0].splitValue = -1;
	root.tree[0].classification = majorityClass;
	root.tree[0].nodeID = 0;
	root.tree[0].depth = 0;

	root.regions.push_back(std::make_pair(0, (int)fspaceFinal.nbCells - 1));
	root.regionNodeIDs.push_back(0);
	root.score = stateScore(root);

	std::vector<BeamState> beam;
	beam.push_back(root);

	const int BEAM_WIDTH = 5;
	const int MAX_ITERS = 100000;
	int iter = 0;

	while (!beam.empty() && iter < MAX_ITERS)
	{
		iter++;

		// Check if all states in beam are fully pure
		bool allPure = true;
		for (const auto & s : beam)
		{
			for (const auto & reg : s.regions)
			{
				if (regionImpurity(reg.first, reg.second) > 0.0)
				{
					allPure = false;
					break;
				}
			}
			if (!allPure) break;
		}

		if (allPure)
			break;

		std::vector<BeamState> newBeam;

		for (const auto & s : beam)
		{
			// Find the most impure region in this state
			int bestRegionIdx = -1;
			double worstImp = 0.0;
			for (int i = 0; i < (int)s.regions.size(); i++)
			{
				double imp = regionImpurity(s.regions[i].first, s.regions[i].second);
				if (imp > worstImp)
				{
					worstImp = imp;
					bestRegionIdx = i;
				}
			}

			if (bestRegionIdx == -1 || worstImp == 0.0)
			{
				// Nothing to expand in this state
				newBeam.push_back(s);
				continue;
			}

			int indexBottom = s.regions[bestRegionIdx].first;
			int indexTop = s.regions[bestRegionIdx].second;
			int parentNodeID = s.regionNodeIDs[bestRegionIdx];

			// Enumerate candidate splits and keep the best few
			struct Cand
			{
				int feature;
				int level;
				int indexTopLeft;
				int indexBottomRight;
				double score;
				std::vector<int> leftCounts;
				std::vector<int> rightCounts;
			};

			std::vector<Cand> candidates;

			for (int k = 0; k < params->nbFeatures; k++)
			{
				const int codeBookValue = fspaceFinal.codeBook[k];
				const int rangeLow = fspaceFinal.keyToCell(indexBottom, k);
				const int rangeUp = fspaceFinal.keyToCell(indexTop, k);
				if (rangeLow == rangeUp) continue;

				for (int l = rangeLow; l < rangeUp; l++)
				{
					int indexTopLeft = indexTop + codeBookValue * (l - rangeUp);
					int indexBottomRight = indexBottom + codeBookValue * (l + 1 - rangeLow);

					std::vector<int> leftCounts(params->nbClasses, 0);
					std::vector<int> rightCounts(params->nbClasses, 0);
					computeClassCountsRegion(indexBottom, indexTopLeft, leftCounts);
					computeClassCountsRegion(indexBottomRight, indexTop, rightCounts);

					int leftTotal = 0;
					int rightTotal = 0;
					for (int c = 0; c < params->nbClasses; c++)
					{
						leftTotal += leftCounts[c];
						rightTotal += rightCounts[c];
					}

					if (leftTotal == 0 || rightTotal == 0) continue;

					double giniLeft = 0.0;
					double giniRight = 0.0;
					for (int c = 0; c < params->nbClasses; c++)
					{
						if (leftCounts[c] > 0)
						{
							double p = (double)leftCounts[c] / (double)leftTotal;
							giniLeft += p * p;
						}
						if (rightCounts[c] > 0)
						{
							double p = (double)rightCounts[c] / (double)rightTotal;
							giniRight += p * p;
						}
					}
					giniLeft = 1.0 - giniLeft;
					giniRight = 1.0 - giniRight;

					double weighted = ((double)leftTotal * giniLeft + (double)rightTotal * giniRight) / (double)(leftTotal + rightTotal);

					Cand cnd;
					cnd.feature = k;
					cnd.level = l;
					cnd.indexTopLeft = indexTopLeft;
					cnd.indexBottomRight = indexBottomRight;
					cnd.score = weighted;
					cnd.leftCounts = leftCounts;
					cnd.rightCounts = rightCounts;
					candidates.push_back(std::move(cnd));
				}
			}

			if (candidates.empty())
			{
				// No valid split; keep state as is
				newBeam.push_back(s);
				continue;
			}

			std::sort(candidates.begin(), candidates.end(),
			          [](const Cand & a, const Cand & b) { return a.score < b.score; });

			int maxChildren = std::min((int)candidates.size(), 3); // expand top 3 splits per state

			for (int ci = 0; ci < maxChildren; ci++)
			{
				const Cand & cnd = candidates[ci];
				BeamState child = s;
				child.splits++;

				// Turn parent node into internal node
				Node & parent = child.tree[parentNodeID];
				parent.nodeType = Node::NODE_INTERNAL;
				parent.splitFeature = cnd.feature;
				parent.splitValue = fspaceFinal.orderedHyperplaneLevels[cnd.feature][cnd.level];

				int parentDepth = parent.depth;

				// Left child
				child.tree.push_back(Node());
				int leftID = (int)child.tree.size() - 1;
				child.tree[leftID].nodeType = Node::NODE_LEAF;
				child.tree[leftID].splitFeature = -1;
				child.tree[leftID].splitValue = -1;
				int leftMajority = 0, leftMax = -1;
				for (int c = 0; c < params->nbClasses; c++)
				{
					if (cnd.leftCounts[c] > leftMax)
					{
						leftMax = cnd.leftCounts[c];
						leftMajority = c;
					}
				}
				child.tree[leftID].classification = leftMajority;
				child.tree[leftID].nodeID = leftID;
				child.tree[leftID].depth = parentDepth + 1;

				// Right child
				child.tree.push_back(Node());
				int rightID = (int)child.tree.size() - 1;
				child.tree[rightID].nodeType = Node::NODE_LEAF;
				child.tree[rightID].splitFeature = -1;
				child.tree[rightID].splitValue = -1;
				int rightMajority = 0, rightMax = -1;
				for (int c = 0; c < params->nbClasses; c++)
				{
					if (cnd.rightCounts[c] > rightMax)
					{
						rightMax = cnd.rightCounts[c];
						rightMajority = c;
					}
				}
				child.tree[rightID].classification = rightMajority;
				child.tree[rightID].nodeID = rightID;
				child.tree[rightID].depth = parentDepth + 1;

				parent.leftChild = leftID;
				parent.rightChild = rightID;

				// Update regions
				child.regions.erase(child.regions.begin() + bestRegionIdx);
				child.regionNodeIDs.erase(child.regionNodeIDs.begin() + bestRegionIdx);

				child.regions.push_back(std::make_pair(indexBottom, cnd.indexTopLeft));
				child.regionNodeIDs.push_back(leftID);
				child.regions.push_back(std::make_pair(cnd.indexBottomRight, indexTop));
				child.regionNodeIDs.push_back(rightID);

				child.score = stateScore(child);

				newBeam.push_back(std::move(child));
			}
		}

		if (newBeam.empty())
			break;

		std::sort(newBeam.begin(), newBeam.end(),
		          [](const BeamState & a, const BeamState & b) { return a.score < b.score; });

		if ((int)newBeam.size() > BEAM_WIDTH)
			newBeam.resize(BEAM_WIDTH);

		beam.swap(newBeam);
	}

	// Choose best state from beam and copy its tree into rebornTree
	if (!beam.empty())
	{
		const BeamState * best = &beam[0];
		for (const auto & s : beam)
		{
			if (s.score < best->score)
				best = &s;
		}

		rebornTree = best->tree;
		finalSplits = 0;
		finalLeaves = 0;
		finalDepth = 0;
		for (const auto & n : rebornTree)
		{
			if (n.nodeType == Node::NODE_INTERNAL) finalSplits++;
			else if (n.nodeType == Node::NODE_LEAF) finalLeaves++;
			if ((unsigned int)n.depth > finalDepth) finalDepth = n.depth;
		}
	}
}

// Simple purity test used by the A* search: a region is considered impure
// whenever there exists at least one pair of cells with different labels.
static bool isRegionImpureAStar(const FSpace & fspace, int indexBottom, int indexTop)
{
	if (indexBottom == indexTop) return false;
	if (indexBottom < 0 || indexTop < 0 || indexBottom >= fspace.nbCells || indexTop >= fspace.nbCells) return true;
	int cls = fspace.cells[indexBottom];
	for (int idx = indexBottom + 1; idx <= indexTop; ++idx)
		if (fspace.cells[idx] != cls) return true;
	return false;
}

// A* search over regions to minimize the number of splits (equivalent to minimizing
// the number of leaves). This function only computes the optimal objective value;
// tree reconstruction is handled separately via the DP memory.
unsigned int BornAgainDecisionTree::aStarOptimizeNbSplits()
{
	struct AStarState
	{
		std::vector<std::pair<int,int>> pending;
		unsigned int g;
		unsigned int h;
		unsigned int f() const { return g + h; }
	};

	struct AStarCompare
	{
		bool operator()(const AStarState & a, const AStarState & b) const
		{
			return a.f() > b.f();
		}
	};

	auto computeHeuristic = [&](const std::vector<std::pair<int,int>> & pending) -> unsigned int
	{
		unsigned int h = 0;
		for (const auto & reg : pending)
		{
			if (isRegionImpureAStar(fspaceFinal, reg.first, reg.second))
				h += 1;
		}
		return h;
	};

	AStarState start;
	start.pending.clear();
	start.pending.push_back(std::make_pair(0, (int)fspaceFinal.nbCells - 1));
	start.g = 0;
	start.h = computeHeuristic(start.pending);

	std::priority_queue<AStarState, std::vector<AStarState>, AStarCompare> open;
	open.push(start);
	// Count initial generated state
	regionsMemorizedDP++;

	while (!open.empty())
	{
		AStarState current = open.top();
		open.pop();

		// Count state expansions
		iterationsDP++;

		// Goal test: all remaining regions are pure (no impure region left)
		if (current.h == 0)
		{
			return current.g;
		}

		// Select first impure region to expand
		int regionIdx = -1;
		for (int i = 0; i < (int)current.pending.size(); i++)
		{
			if (isRegionImpureAStar(fspaceFinal, current.pending[i].first, current.pending[i].second))
			{
				regionIdx = i;
				break;
			}
		}

		if (regionIdx == -1)
			continue;

		int indexBottom = current.pending[regionIdx].first;
		int indexTop = current.pending[regionIdx].second;

		// Enumerate all possible splits on this region
		for (int k = 0; k < params->nbFeatures; k++)
		{
			const int codeBookValue = fspaceFinal.codeBook[k];
			const int rangeLow = fspaceFinal.keyToCell(indexBottom, k);
			const int rangeUp = fspaceFinal.keyToCell(indexTop, k);
			for (int l = rangeLow; l < rangeUp; l++)
			{
				int indexTopLeft = indexTop + codeBookValue * (l - rangeUp);
				int indexBottomRight = indexBottom + codeBookValue * (l + 1 - rangeLow);

				AStarState succ;
				succ.pending = current.pending;
				succ.pending.erase(succ.pending.begin() + regionIdx);
				succ.pending.push_back(std::make_pair(indexBottom, indexTopLeft));
				succ.pending.push_back(std::make_pair(indexBottomRight, indexTop));

				succ.g = current.g + 1;
				succ.h = computeHeuristic(succ.pending);

				open.push(succ);
				regionsMemorizedDP++;
			}
		}
	}

	throw std::string("A* search failed to find a solution");
}

int BornAgainDecisionTree::recursiveHelperHeuristic(unsigned int currentDepth)
{
	// PICK A RANDOM SAMPLE OF CELLS WITHIN THIS REGION
	for (int k : nonTrivialFeatures)
	{
		std::uniform_int_distribution<int> distribution(0, topRightCell[k] - bottomLeftCell[k]);
		for (int s = 0; s < params->nbCellsSampled; s++)
		{
			sampledCellsIndices[s][k] = bottomLeftCell[k] + distribution(params->generator);
			sampledCellsCoords[s][k] = orderedHyperplaneLevelsHeuristic[k][sampledCellsIndices[s][k]];
		}
	}

	// COLLECT SOME STATISTICS
	int nbClassesRepresented = 0;
	std::vector<int> nbSamplesClass = std::vector<int>(params->nbClasses, 0);
	for (int s = 0; s < params->nbCellsSampled; s++)
	{
		classSampledCells[s] = randomForest->majorityClass(sampledCellsCoords[s]);
		if (nbSamplesClass[classSampledCells[s]] == 0) nbClassesRepresented++;
		nbSamplesClass[classSampledCells[s]]++;
	}

	if (nbClassesRepresented == 0)
		throw std::string("ISSUE: NUMBER OF CLASSES REPRESENTED SHOULD BE STRICTLY POSITIVE");

	bool regionIsPure = true;
	if (nbClassesRepresented > 1)
		regionIsPure = false;
#ifdef USING_CPLEX
	else // nbClassesRepresented == 1
	{
		// CALL MIP CERTIFICATE TO VERIFY PURITY (only when the code is linked to CPLEX)
		int pureClassID = classSampledCells[0];
		for (int c = 0; c < params->nbClasses; c++)
		{
			if (c != pureClassID
				&& !myMIPcertificate->preFilterMinMax(nonTrivialFeaturesBeforeSplits, orderedHyperplaneLevelsHeuristic, bottomLeftCell, topRightCell, pureClassID, c)  // Use a pre-filter
				&& !myMIPcertificate->buildAndRunMIP(nonTrivialFeaturesBeforeSplits, nonTrivialFeatures, orderedHyperplaneLevelsHeuristic, bottomLeftCell, topRightCell, pureClassID, c))  // Do an exact check
			{
				regionIsPure = false;
				// cout << "IMPURITY DETECTED BY MIP BUT SAMPLES WERE PURE" << endl;
				break;
			}
		}
	}
#endif

	if (regionIsPure)
	{
		// CREATE A LEAF
		finalLeaves++;
		if (currentDepth > finalDepth) finalDepth = currentDepth;
		rebornTree.push_back(Node());
		int nodeID = (int)rebornTree.size() - 1;
		rebornTree[nodeID].nodeType = Node::NODE_LEAF;
		rebornTree[nodeID].splitFeature = -1;
		rebornTree[nodeID].splitValue = -1;
		rebornTree[nodeID].classification = classSampledCells[0];
		rebornTree[nodeID].nodeID = nodeID;
		rebornTree[nodeID].depth = currentDepth;
		return nodeID;
	}
	else
	{
		// OTHERWISE LOOK FOR THE BEST SPLIT BASED ON THIS RANDOM SAMPLE OF CELLS USING AN INFORMATION GAIN CRITERION
		double bestEntropySub = 1.e30;
		int bestSplitFeature = -1;
		unsigned short int bestSplitLevel = 10000;
		std::uniform_int_distribution<int> distribution(0, params->nbFeatures - 1);
		int kInit = distribution(params->generator); // Avoids possible bias due to the index order when breaking ties
		for (int kk = 0; kk < params->nbFeatures; kk++)
		{
			int k = (kInit + kk) % params->nbFeatures;
			// For each feature which has more than one level
			if (bottomLeftCell[k] != topRightCell[k])
			{
				std::vector<std::pair<unsigned short int, unsigned short int>> orderedSamples;
				for (int s = 0; s < params->nbCellsSampled; s++)
					orderedSamples.push_back(std::pair<unsigned short int, unsigned short int>(sampledCellsIndices[s][k], classSampledCells[s]));
				std::sort(orderedSamples.begin(), orderedSamples.end());

				// Initially all samples are on the right
				std::vector <int> nbSamplesClassLeft = std::vector<int>(params->nbClasses, 0);
				std::vector <int> nbSamplesClassRight = nbSamplesClass;
				int indexSample = 0;
				for (unsigned short int attributeValue = bottomLeftCell[k]; attributeValue < topRightCell[k]; attributeValue++)
				{
					// Iterate on all samples with this attributeValue and switch them to the left
					while (indexSample < params->nbCellsSampled && orderedSamples[indexSample].first <= attributeValue)
					{
						nbSamplesClassLeft[orderedSamples[indexSample].second]++;
						nbSamplesClassRight[orderedSamples[indexSample].second]--;
						indexSample++;
					}

					// No need to consider the case in which all samples have been switched to the left
					if (indexSample != params->nbCellsSampled)
					{
						// Evaluate entropy of the two resulting sample sets
						double entropyLeft = 0.0;
						double entropyRight = 0.0;
						for (int c = 0; c < params->nbClasses; c++)
						{
							// Remark that indexSample contains at this stage the number of samples in the left
							if (nbSamplesClassLeft[c] > 0)
							{
								double fracLeft = (double)nbSamplesClassLeft[c] / (double)(indexSample);
								entropyLeft -= fracLeft * log2(fracLeft);
							}
							if (nbSamplesClassRight[c] > 0)
							{
								double fracRight = (double)nbSamplesClassRight[c] / (double)(params->nbCellsSampled - indexSample);
								entropyRight -= fracRight * log2(fracRight);
							}
						}

						// Evaluate the information gain and store if this is the best option found until now
						double entropySub = ((double)indexSample*entropyLeft + (double)(params->nbCellsSampled - indexSample)*entropyRight) / (double)params->nbCellsSampled;
						if (entropySub < bestEntropySub)
						{
							bestEntropySub = entropySub;
							bestSplitFeature = k;
							bestSplitLevel = attributeValue;
						}
					}
				}
			}
		}

		/* APPLY THE SPLIT AND PERFORM RECURSIVE CALLS */
		finalSplits++;
		rebornTree.push_back(Node());
		int nodeID = (int)rebornTree.size() - 1;
		rebornTree[nodeID].nodeType = Node::NODE_INTERNAL;
		rebornTree[nodeID].splitFeature = bestSplitFeature;
		rebornTree[nodeID].splitValue = orderedHyperplaneLevelsHeuristic[bestSplitFeature][bestSplitLevel];
		rebornTree[nodeID].nodeID = nodeID;

		int tempR = topRightCell[bestSplitFeature];
		topRightCell[bestSplitFeature] = bestSplitLevel;
		// Possibly eliminate this feature from the list of features which need to be considered
		if (bottomLeftCell[bestSplitFeature] == topRightCell[bestSplitFeature])
		{
			nonTrivialFeatures.erase(bestSplitFeature);
			for (int s = 0; s < params->nbCellsSampled; s++)
			{
				sampledCellsIndices[s][bestSplitFeature] = bottomLeftCell[bestSplitFeature];
				sampledCellsCoords[s][bestSplitFeature] = orderedHyperplaneLevelsHeuristic[bestSplitFeature][sampledCellsIndices[s][bestSplitFeature]];
			}
		}
		int myLeftID = recursiveHelperHeuristic(currentDepth + 1);
		topRightCell[bestSplitFeature] = tempR;
		nonTrivialFeatures.insert(bestSplitFeature);

		int tempL = bottomLeftCell[bestSplitFeature];
		bottomLeftCell[bestSplitFeature] = bestSplitLevel + 1;
		// Possibly eliminate this feature from the list of features which need to be considered
		if (bottomLeftCell[bestSplitFeature] == topRightCell[bestSplitFeature])
		{
			nonTrivialFeatures.erase(bestSplitFeature);
			for (int s = 0; s < params->nbCellsSampled; s++)
			{
				sampledCellsIndices[s][bestSplitFeature] = bottomLeftCell[bestSplitFeature];
				sampledCellsCoords[s][bestSplitFeature] = orderedHyperplaneLevelsHeuristic[bestSplitFeature][sampledCellsIndices[s][bestSplitFeature]];
			}
		}
		int myRightID = recursiveHelperHeuristic(currentDepth + 1);
		bottomLeftCell[bestSplitFeature] = tempL;
		nonTrivialFeatures.insert(bestSplitFeature);

		rebornTree[nodeID].leftChild = myLeftID;
		rebornTree[nodeID].rightChild = myRightID;
		rebornTree[nodeID].depth = currentDepth;
		return nodeID;
	}
}

void BornAgainDecisionTree::buildHeuristic()
{
	finalSplits = 0;
	finalLeaves = 0;
	finalDepth = 0;

	// Get the ordered Hyperplanes
	orderedHyperplaneLevelsHeuristic = randomForest->getHyperplanes();

	// Initialize the solver
#ifdef USING_CPLEX
	myMIPcertificate = new MIPCertificate(params, randomForest);
#endif
	
	// Initialize the original region
	for (int k = 0; k < params->nbFeatures; k++)
	{
		bottomLeftCell.push_back(0); // Initializing the bottom left cell
		topRightCell.push_back((int)orderedHyperplaneLevelsHeuristic[k].size()-1); // Initializing the top right cell
		if (bottomLeftCell[k] != topRightCell[k]) nonTrivialFeatures.insert(k); // Keeping track of the features which are non-trivial
	}
	nonTrivialFeaturesBeforeSplits = nonTrivialFeatures;

	// Initialize other data structures and call the recursive construction procedure
	sampledCellsIndices = std::vector<std::vector<unsigned short int>>(params->nbCellsSampled, std::vector<unsigned short int>(params->nbFeatures, 0));
	sampledCellsCoords = std::vector<std::vector<double>>(params->nbCellsSampled, std::vector<double>(params->nbFeatures, 1.e30));
	classSampledCells = std::vector<unsigned short int>(params->nbCellsSampled);
	recursiveHelperHeuristic(0);

#ifdef USING_CPLEX
	delete myMIPcertificate;
#endif
}