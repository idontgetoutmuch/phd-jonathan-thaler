module SpacialGameMsg.RunSGMsg where

import SpacialGameMsg.SGModelMsg
import qualified PureAgents2DDiscrete as Front
import qualified Graphics.Gloss as GLO
import Graphics.Gloss.Interface.IO.Simulate

import qualified PureAgentsPar as PA

import System.Random
import Data.Maybe
import Data.List

winSize = (1000, 1000)
winTitle = "Spatial Game ABS"

runSGMsgWithRendering :: IO ()
runSGMsgWithRendering = do
                            let dt = 1.0
                            let dims = (9, 9)
                            let rngSeed = 42
                            let defectorsRatio = 0.0
                            let g = mkStdGen rngSeed

                            let (as, g') = createRandomSGAgents g dims defectorsRatio
                            let asWithDefector = setDefector as (4, 4) dims

                            let env = sgEnvironmentFromAgents asWithDefector

                            let hdl = PA.initStepSimulation asWithDefector env
                            stepWithRendering dims hdl dt

runSGMsgStepsAndRender :: IO ()
runSGMsgStepsAndRender = do
                            let dt = 1.0
                            let dims = (99, 99)
                            let rngSeed = 42
                            let steps = 60
                            let defectorsRatio = 0.0
                            let g = mkStdGen rngSeed

                            let (as, g') = createRandomSGAgents g dims defectorsRatio
                            let asWithDefector = setDefector as (49, 49) dims
                            let env = sgEnvironmentFromAgents asWithDefector

                            let (as', _) = PA.stepSimulation asWithDefector env dt steps

                            let observableAgentStates = map (sgAgentToRenderCell dims) as'
                            let frameRender = (Front.renderFrame observableAgentStates winSize dims)
                            GLO.display (Front.display winTitle winSize) GLO.white frameRender
                            return ()



stepWithRendering :: (Int, Int) -> SGSimHandle -> Double -> IO ()
stepWithRendering dims hdl dt = simulateIO (Front.display winTitle winSize)
                                GLO.white
                                2
                                hdl
                                (modelToPicture dims)
                                (stepIteration dt)

-- A function to convert the model to a picture.
modelToPicture :: (Int, Int) -> SGSimHandle -> IO GLO.Picture
modelToPicture dims hdl = do
                            let as = PA.extractHdlAgents hdl
                            let cells = map (sgAgentToRenderCell dims) as
                            return (Front.renderFrame cells winSize dims)

sgAgentToRenderCell :: (Int, Int) -> SGAgent -> Front.RenderCell
sgAgentToRenderCell (xDim, yDim) a = Front.RenderCell { Front.renderCellCoord = (ax, ay),
                                                        Front.renderCellColor = ss }
    where
        id = PA.agentId a
        s = PA.state a
        ax = mod id yDim
        ay = floor((fromIntegral id) / (fromIntegral xDim))
        curr = sgCurrState s
        prev = sgPrevState s
        ss = sgAgentStateToColor prev curr

-- NOTE: read it the following way: "the agent was in state X following another one Y" => first parameter is prev, second is curr
sgAgentStateToColor :: SGState -> SGState -> (Double, Double, Double)
sgAgentStateToColor Cooperator Cooperator = blueC
sgAgentStateToColor Defector Defector = redC
sgAgentStateToColor Defector Cooperator = greenC
sgAgentStateToColor Cooperator Defector = yellowC

blueC :: (Double, Double, Double)
blueC = (0.0, 0.0, 0.7)

greenC :: (Double, Double, Double)
greenC = (0.0, 0.4, 0.0)

redC :: (Double, Double, Double)
redC = (0.7, 0.0, 0.0)

yellowC :: (Double, Double, Double)
yellowC = (1.0, 0.9, 0.0)

-- A function to step the model one iteration. It is passed the current viewport and the amount of time for this simulation step (in seconds)
-- NOTE: atomically is VERY important, if it is not there there then the STM-transactions would not occur!
--       NOTE: this is actually wrong, we can avoid atomically as long as we are running always on the same thread.
--             atomically would commit the changes and make them visible to other threads
stepIteration :: Double -> ViewPort -> Float -> SGSimHandle -> IO SGSimHandle
stepIteration fixedDt viewport dtRendering hdl = return (PA.advanceSimulation hdl fixedDt)
--------------------------------------------------------------------------------------------------------------------------------------------------