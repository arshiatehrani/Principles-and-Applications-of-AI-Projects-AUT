from Agents import Agent
import util
import random


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.index = 0  # your agent always has index 0

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        It takes a GameState and returns a tuple representing a position on the game board.
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions(self.index)

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current and proposed successor
        GameStates (Game.py) and returns a number, where higher numbers are better.
        You can try and change this evaluation function if you want but it is not necessary.
        """
        nextGameState = currentGameState.generateSuccessor(self.index, action)
        return nextGameState.getScore(self.index) - currentGameState.getScore(
            self.index
        )


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    Every player's score is the number of pieces they have placed on the board.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore(0)


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (Agents.py)
    is another abstract class.
    """

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2", **kwargs):
        self.index = 0  # your agent always has index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent which extends MultiAgentSearchAgent and is supposed to be implementing a minimax tree with a certain depth.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(**kwargs)

    def getAction(self, state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """

        # Function
        def minimax(state, depth, agentIndex):
            # Check if terminal state or reached maximum depth
            if state.isGameFinished() or depth == 0:
                return self.evaluationFunction(state), None

            # Get legal actions for the current agent
            legalActions = state.getLegalActions(agentIndex)

            # If current agent is the maximizing agent (self)
            if agentIndex == 0:
                maxEval = float("-inf")
                bestAction = None

                # Evaluate each action for the current agent
                for action in legalActions:
                    nextState = state.generateSuccessor(agentIndex, action)
                    eval, _ = minimax(
                        nextState, depth - 1, (agentIndex + 1) % state.getNumAgents()
                    )

                    if eval > maxEval:
                        maxEval = eval
                        bestAction = action

                return maxEval, bestAction

            # If current agent is minimizing agent (opponent)
            else:
                minEval = float("inf")
                bestAction = None

                # Evaluate each action for the opponent agent
                for action in legalActions:
                    nextState = state.generateSuccessor(agentIndex, action)
                    eval, _ = minimax(
                        nextState, depth - 1, (agentIndex + 1) % state.getNumAgents()
                    )

                    if eval < minEval:
                        minEval = eval
                        bestAction = action

                return minEval, bestAction

        _, bestAction = minimax(state, self.depth, 0)  # Start with agent index 0 (self)
        return bestAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning. It is very similar to the MinimaxAgent but you need to implement the alpha-beta pruning algorithm too.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(**kwargs)

    def getAction(self, state):
        """
        Returns the minimax action using alpha-beta pruning with self.depth and self.evaluationFunction
        """

        def alpha_beta_pruning(state, depth, alpha, beta, agentIndex):
            if state.isGameFinished() or depth == 0:
                return self.evaluationFunction(state), None

            legalActions = state.getLegalActions(agentIndex)

            if agentIndex == 0:  # Maximizing agent (self)
                maxEval = float("-inf")
                bestAction = None

                for action in legalActions:
                    nextState = state.generateSuccessor(agentIndex, action)
                    eval, _ = alpha_beta_pruning(
                        nextState,
                        depth - 1,
                        alpha,
                        beta,
                        (agentIndex + 1) % state.getNumAgents(),
                    )

                    if eval > maxEval:
                        maxEval = eval
                        bestAction = action

                    alpha = max(alpha, maxEval)
                    if beta <= alpha:
                        break  # Beta cutoff

                return maxEval, bestAction

            else:  # Minimizing agent (opponent)
                minEval = float("inf")
                bestAction = None

                for action in legalActions:
                    nextState = state.generateSuccessor(agentIndex, action)
                    eval, _ = alpha_beta_pruning(
                        nextState,
                        depth - 1,
                        alpha,
                        beta,
                        (agentIndex + 1) % state.getNumAgents(),
                    )

                    if eval < minEval:
                        minEval = eval
                        bestAction = action

                    beta = min(beta, minEval)
                    if beta <= alpha:
                        break  # Alpha cutoff

                return minEval, bestAction

        _, bestAction = alpha_beta_pruning(
            state, self.depth, float("-inf"), float("inf"), 0
        )
        return bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent which has a max node for your agent but every other node is a chance node.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(**kwargs)

    def getAction(self, state):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All opponents should be modeled as choosing uniformly at random from their
        legal moves.
        """

        def expectimax(state, depth, agentIndex):
            if state.isGameFinished() or depth == 0:
                return self.evaluationFunction(state), None

            legalActions = state.getLegalActions(agentIndex)

            if agentIndex == 0:  # Maximizing agent (self)
                maxEval = float("-inf")
                bestAction = None

                for action in legalActions:
                    nextState = state.generateSuccessor(agentIndex, action)
                    eval, _ = expectimax(
                        nextState, depth - 1, (agentIndex + 1) % state.getNumAgents()
                    )

                    if eval > maxEval:
                        maxEval = eval
                        bestAction = action

                return maxEval, bestAction

            else:  # Chance node (opponent)
                avgEval = 0.0
                numActions = len(legalActions)

                for action in legalActions:
                    nextState = state.generateSuccessor(agentIndex, action)
                    eval, _ = expectimax(
                        nextState, depth - 1, (agentIndex + 1) % state.getNumAgents()
                    )
                    avgEval += eval

                return avgEval / numActions, None

        _, bestAction = expectimax(state, self.depth, 0)
        return bestAction


def betterEvaluationFunction(currentGameState):
    """
    Your evaluation function for the Rollit game.
    """

    def Parity(currentGameState):
        # Get scores for both players
        maxScore = currentGameState.getScore(0)
        minScore = currentGameState.getScore(1)

        # Calculate parity heuristic
        scoreDifference = maxScore - minScore
        totalScore = maxScore + minScore

        # Avoid division by zero
        if totalScore == 0:
            return 0

        parity = scoreDifference / totalScore
        return parity

    def Mobility(currentGameState):
        # Calculate mobility for both players
        maxMobility = len(currentGameState.getLegalActions(0))
        minMobility = len(currentGameState.getLegalActions(1))

        # Calculate mobility heuristic
        mobilityDifference = maxMobility - minMobility
        totalMobility = maxMobility + minMobility

        # Avoid division by zero
        if totalMobility != 0:
            mobilityHeuristic = 100 * (mobilityDifference / totalMobility)
        else:
            mobilityHeuristic = 0

        return mobilityHeuristic

    # def Corners(currentGameState):
    #     corners = currentGameState.getCorners()
    #     playerCorners = [
    #         corners[i] for i in range(4) if currentGameState.getPieces(0)[i]
    #     ]
    #     return len(playerCorners)

    #     # Get corner positions for both players
    def Corners(currentGameState):
        corners = currentGameState.getCorners()  # Get corner positions
        playerPieces = currentGameState.getPieces(0)  # Get player's pieces

        # we iterate through each corner and check if
        # it's occupied by a player's piece,
        # incrementing playerCorners accordingly.
        # This approach avoids the index out of range error
        # and should correctly count
        # the number of corners occupied by the player.
        playerCorners = 0
        for corner in corners:
            if corner in playerPieces:
                playerCorners += 1

        return playerCorners

    def isStable(currentGameState, playerIndex, piece):
        # True if the piece is stable, False otherwise.
        # Add your stability criteria here
        # For example, a piece is stable if it is surrounded by friendly pieces

        # Check if the piece is in a corner (always stable)
        if piece in currentGameState.getCorners():
            return True
        return False

    def Stability(currentGameState):
        playerPieces = currentGameState.getPieces(0)
        stabilityValue = 0

        # Check each piece for stability
        for piece in playerPieces:
            if isStable(currentGameState, 0, piece):
                stabilityValue += 1
            else:
                stabilityValue -= 1

        return stabilityValue

    # Combine the heuristic values using appropriate weights (ones are only for the test, based on the performance, the weights could be improved)
    ParityWeight = 1
    MobilityWeight = 1
    CornersWeight = 1
    StabilityWeight = 1

    evalValue = (
        Parity(currentGameState) * ParityWeight
        + Mobility(currentGameState) * MobilityWeight
        + Corners(currentGameState) * CornersWeight
        + Stability(currentGameState) * StabilityWeight
    )

    return evalValue


# Abbreviation
better = betterEvaluationFunction
