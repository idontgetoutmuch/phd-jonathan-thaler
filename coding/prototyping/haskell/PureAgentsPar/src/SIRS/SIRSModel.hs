{-# LANGUAGE DeriveGeneric, DeriveAnyClass #-}

module SIRS.SIRSModel where

import System.Random
import Control.DeepSeq
import GHC.Generics (Generic)

import qualified PureAgentsPar as PA

data SIRSState = Susceptible | Infected | Recovered deriving (Eq, Show, Generic, NFData)
data SIRSMsg = Contact SIRSState deriving (Generic, NFData)

data SIRSAgentState = SIRSAgentState {
    sirState :: SIRSState,
    timeInState :: Double,
    rng :: StdGen
} deriving (Show, Generic)

-- NOTE: need to provide an instance-implementation for NFData when using Par-Monad as it reduces to normal-form
-- NOTE: use separate implementation because ommiting StdGen
instance NFData SIRSAgentState where
    rnf (SIRSAgentState s t r) = rnf s `seq` rnf t

type SIRSEnvironment = ()
type SIRSAgent = PA.Agent SIRSMsg SIRSAgentState SIRSEnvironment
type SIRSTransformer = PA.AgentTransformer SIRSMsg SIRSAgentState SIRSEnvironment
type SIRSSimHandle = PA.SimHandle SIRSMsg SIRSAgentState SIRSEnvironment

infectedDuration :: Double
infectedDuration = 7.0

immuneDuration :: Double
immuneDuration = 3.0

infectionProbability :: Double
infectionProbability = 0.3

is :: SIRSAgent -> SIRSState -> Bool
is a ss = (sirState s) == ss
    where
        s = PA.state a

sirsTransformer :: SIRSTransformer
sirsTransformer a (_, PA.Dt dt) = sirsDt a dt
sirsTransformer a (_, PA.Domain m) = sirsMsg a m

sirsMsg :: SIRSAgent -> SIRSMsg -> SIRSAgent
-- MESSAGE-CASE: Contact with Infected -> infect with given probability if agent is susceptibel
sirsMsg a (Contact Infected)               -- NOTE: ignore sender
    | is a Susceptible = infectAgent a
    | otherwise = a
-- MESSAGE-CASE: Contact with Recovered or Susceptible -> nothing happens
sirsMsg a (Contact _) = a           -- NOTE: ignore sender

sirsDt :: SIRSAgent -> Double -> SIRSAgent
sirsDt a dt
    | is a Susceptible = a
    | is a Infected = handleInfectedAgent a dt
    | is a Recovered = handleRecoveredAgent a dt

infectAgent :: SIRSAgent -> SIRSAgent
infectAgent a
    | infect = PA.updateState a (\sOld -> sOld { sirState = Infected, timeInState = 0.0, rng = g' } )
    | otherwise = PA.updateState a (\sOld -> sOld { rng = g' } )
    where
        g = (rng (PA.state a))
        (infect, g') = randomThresh g infectionProbability

handleInfectedAgent :: SIRSAgent -> Double -> SIRSAgent
handleInfectedAgent a dt = if t' >= infectedDuration then
                                recoveredAgent           -- NOTE: agent has just recovered, don't send infection-contact to others
                                else
                                    randomContact gettingBetterAgent

    where
        t = (timeInState (PA.state a))
        t' = t + dt
        recoveredAgent = PA.updateState a (\sOld -> sOld { sirState = Recovered, timeInState = 0.0 } )
        gettingBetterAgent = PA.updateState a (\sOld -> sOld { timeInState = t' } )

handleRecoveredAgent :: SIRSAgent -> Double -> SIRSAgent
handleRecoveredAgent a dt = if t' >= immuneDuration then
                                susceptibleAgent
                                else
                                    immuneReducedAgent
    where
        t = (timeInState (PA.state a))
        t' = t + dt
        susceptibleAgent = PA.updateState a (\sOld -> sOld { sirState = Susceptible, timeInState = 0.0 } )
        immuneReducedAgent = PA.updateState a (\sOld -> sOld { timeInState = t' } )


randomContact :: SIRSAgent -> SIRSAgent
randomContact a = PA.updateState a' (\sOld -> sOld { rng = g' } )
    where
        s = PA.state a
        (a', g') = PA.sendMsgToRandomNeighbour a (Contact Infected) (rng s)

createRandomSIRSAgents :: StdGen -> (Int, Int) -> Double -> ([SIRSAgent], StdGen)
createRandomSIRSAgents gInit cells@(x,y) p = (as', g')
    where
        n = x * y
        (randStates, g') = createRandomStates gInit n p
        as = map (\idx -> PA.createAgent idx (randStates !! idx) sirsTransformer) [0..n-1]
        --as' = map (\a -> PA.addNeighbours a (filter (\a' -> (PA.agentId a') /= (PA.agentId a)) as) ) as
        as' = map (\a -> PA.addNeighbours a (agentNeighbours a as cells) ) as

        createRandomStates :: StdGen -> Int -> Double -> ([SIRSAgentState], StdGen)
        createRandomStates g 0 p = ([], g)
        createRandomStates g n p = (rands, g'')
            where
              (randState, g') = randomAgentState g p
              (ras, g'') = createRandomStates g' (n-1) p
              rands = randState : ras

randomAgentState :: StdGen -> Double -> (SIRSAgentState, StdGen)
randomAgentState g p = (SIRSAgentState{ sirState = s, timeInState = 0.0, rng = g'' }, g')
    where
        (isInfected, g') = randomThresh g p
        (g'', _) = split g'
        s = if isInfected then
                Infected
                else
                    Susceptible

randomThresh :: StdGen -> Double -> (Bool, StdGen)
randomThresh g p = (flag, g')
    where
        (thresh, g') = randomR(0.0, 1.0) g
        flag = thresh <= p


agentNeighbours :: SIRSAgent -> [SIRSAgent] -> (Int, Int) -> [SIRSAgent]
agentNeighbours a as cells = filter (\a' -> any (==(agentToCell a' cells)) neighbourCells ) as
    where
        aCell = agentToCell a cells
        neighbourCells = neighbours aCell

agentToCell :: SIRSAgent -> (Int, Int) -> (Int, Int)
agentToCell a (xCells, yCells) = (ax, ay)
     where
        aid = PA.agentId a
        ax = mod aid yCells
        ay = floor((fromIntegral aid) / (fromIntegral xCells))


neighbourhood :: [(Int, Int)]
neighbourhood = [topLeft, top, topRight,
                 left, right,
                 bottomLeft, bottom, bottomRight]
    where
        topLeft = (-1, -1)
        top = (0, -1)
        topRight = (1, -1)
        left = (-1, 0)
        right = (1, 0)
        bottomLeft = (-1, 1)
        bottom = (0, 1)
        bottomRight = (1, 1)

neighbours :: (Int, Int) -> [(Int, Int)]
neighbours (x,y) = map (\(x', y') -> (x+x', y+y')) neighbourhood
