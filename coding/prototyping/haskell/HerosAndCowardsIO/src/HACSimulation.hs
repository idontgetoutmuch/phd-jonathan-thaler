module HACSimulation (
    SimIn (..),
    SimOut (..),
    SimulationIO,
    SimulationStep
  ) where

import qualified HACAgent as Agent

data SimIn = SimIn {
    simInWorldType :: Agent.WorldType,
    simInInitAgents :: [Agent.AgentState]
}

data SimOut = SimOut {
    simOutAgents :: [Agent.AgentOut]
}

{- NOTE: Drives the simulation by IO e.g. through rendering with the last output as the result -}
type SimulationIO = SimIn -> (SimOut -> IO (Bool, Double)) -> IO ()
{- NOTE: Drives the simulation by iterating a given amount of steps and returning all results where the final result is the last element of the list -}
type SimulationStep = SimIn -> Double -> Int -> IO [SimOut]

