module WildFire.WildFireModelDynamic where

import Control.Monad.STM

import System.Random
import Debug.Trace
import Data.List
import Data.Maybe

import qualified Data.HashMap as Map
import qualified PureAgentsSTM as PA

type WFCellCoord = (Int, Int)
data WFCellState = Living | Burning | Dead deriving (Eq, Show)
type WFMsg = ()

data WFCell = WFCell {
    cellIdx :: Int,
    coord :: WFCellCoord,
    burnable :: Double,
    cellState :: WFCellState
} deriving (Show)

data WFAgentState = WFAgentState {
    cell :: WFCell,
    rng :: StdGen
} deriving (Show)

type WFCellContainer = Map.Map Int WFCell

data WFEnvironment = WFEnvironment {
    cells :: WFCellContainer,
    cellLimits :: WFCellCoord
}

type WFAgent = PA.Agent WFMsg WFAgentState WFEnvironment
type WFMsgHandler = PA.MsgHandler WFMsg WFAgentState WFEnvironment
type WFUpdtHandler = PA.UpdateHandler WFMsg WFAgentState WFEnvironment
type WFSimHandle = PA.SimHandle WFMsg WFAgentState WFEnvironment

burnPerTimeUnit :: Double
burnPerTimeUnit = 0.4

-- NOTE: in this case no messages are sent between agents
wfMsgHandler :: WFMsgHandler
wfMsgHandler a _ _ = return a

-- NOTE: an active agent is always burning: it can be understood as the process of a burning cell
wfUpdtHandler :: WFUpdtHandler
wfUpdtHandler a dt = if burnableLeft <= 0.0 then
                        killCellAndAgent a c

                        else
                            do
                                burnCell a burningCell

                                -- TODO: gosh is this ugly code, refactor it to something more readable!
                                -- TODO: the following must run in one transaction because we search the env and then change it, it could be changed in the mean-time when running in parallel
                                env <- PA.readEnv a
                                let g = (rng (PA.state a))
                                let (randCoord, g') = randomNeighbourCoord g (coord burningCell)
                                let randCellMaybe = cellByCoord env randCoord
                                if isJust randCellMaybe then
                                    do
                                        let randCell = fromJust randCellMaybe
                                        if ( (cellState randCell) == Living) then
                                            do
                                                (aNew, g'') <- igniteCell g' randCell
                                                let a' = PA.newAgent a aNew
                                                return (PA.updateState a' (\sOld -> sOld { rng = g'', cell = burningCell } ))
                                            else
                                                return (PA.updateState a (\sOld -> sOld { rng = g', cell = burningCell } ))
                                        else
                                            return (PA.updateState a (\sOld -> sOld { rng = g', cell = burningCell } ))

                        where
                            c = (cell (PA.state a))
                            b = (burnable c)
                            burnableLeft = b - (burnPerTimeUnit * dt)
                            -- NOTE: always set cell to ignite because can't do that initially
                            burningCell = c { burnable = burnableLeft, cellState = Burning }


burnCell :: WFAgent -> WFCell -> STM ()
burnCell a c = PA.changeEnv a (\e -> replaceCell e c )

killCellAndAgent :: WFAgent -> WFCell -> STM WFAgent
killCellAndAgent a c = do
                        let c' = c { burnable = 0.0, cellState = Dead }
                        PA.changeEnv a (\e -> replaceCell e c' )
                        return (PA.kill a)

neighbourhood :: [WFCellCoord]
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


randomNeighbourCoord :: StdGen -> WFCellCoord -> (WFCellCoord, StdGen)
randomNeighbourCoord g (cx, cy) = (randC, g')
    where
        nsCells = map (\(nx, ny) -> (cx + nx, cy + ny) :: WFCellCoord) neighbourhood
        (randIdx, g') = randomR (0, (length nsCells) - 1) g
        randC = nsCells !! randIdx

igniteCell :: StdGen -> WFCell -> STM (WFAgent, StdGen)
igniteCell g c = do
                    let (g', g'') = split g
                    let aState = WFAgentState { cell = c, rng = g' }
                    let id = (cellIdx c)
                    a <- PA.createAgent id aState wfMsgHandler wfUpdtHandler
                    -- NOTE: don't need any neighbours because no messaging!
                    return (a, g'')

replaceCell :: WFEnvironment -> WFCell -> WFEnvironment
replaceCell env c = env { cells = Map.insert idx c cs }
    where
        idx = cellIdx c
        cs = cells env

cellByCoord :: WFEnvironment -> WFCellCoord -> Maybe WFCell
cellByCoord env co = Map.lookup idx cs
    where
        limits = cellLimits env
        cs = cells env
        idx = idxByCoord co limits

createEnvironment :: (Int, Int) -> WFEnvironment
createEnvironment mcs@(maxX, maxY) = WFEnvironment { cells = csMaped, cellLimits = mcs }
    where
        cs = [ WFCell { cellIdx = (y*maxX) + x,
                            coord = (x, y),
                            burnable = 1.0,
                            cellState = Living } | y <- [0..maxY-1], x <- [0..maxX-1] ]
        csMaped = foldl (\acc c -> Map.insert (cellIdx c) c acc ) Map.empty cs

idxByCoord :: WFCellCoord -> (Int, Int) -> Int
idxByCoord (x, y) (maxX, maxY) = (y*maxX) + x