module SIRHelpers (
   aggregateStates,
   appendLine,
   writeSimulationUntil,
   visualiseSimulation,
   renderAgent,
   ctxToPic,
   transformToWindow,
   animation,
   runSimulationUntil,
   decodeCSV,
   getResults,
   addToList,
   scatterAxis2,
   main1,
   mkSimCtx,
   SimCtx,
   SIRState(..),
   agentGridSize,
   runStepCtx,
   simEnv,
   cellWidth,
   evaluateCtxs
) where

import           Data.IORef
import           System.IO
import           Text.Printf

import           Control.Monad.Random
import           Control.Monad.Reader
import           Control.Monad.Writer
import           Control.Monad.Trans.MSF.Random

import           Data.Array.IArray
import           FRP.BearRiver
import qualified Graphics.Gloss as GLO
import qualified Graphics.Gloss.Interface.IO.Animate as GLOAnim
import           Data.MonadicStreamFunction.InternalCore
import           Data.List (unfoldr)
import           Graphics.Gloss.Export

import qualified Data.ByteString.Lazy as LBS
import qualified Data.Csv             as Csv
import Plots.Axis  (Axis, r2Axis)

import Diagrams.Backend.Cairo (B)
import IHaskell.Display.Diagrams ()
import Plots (scatterPlot, key, r2AxisMain, Plotable, addPlotable', linePlot')
import Diagrams.Prelude hiding (Time, (^/), (^+^), trace, coords, (*^))
import System.Environment

import qualified Data.ByteString.Lazy as BL
import Data.Csv
import qualified Data.Vector as V

import Control.Monad.State

import Debug.Trace

data SIRState = Susceptible | Infected | Recovered deriving (Show, Eq)

data SimCtx g = SimCtx
  { simSf    :: !(SimSF g)
  , simEnv   :: !SIREnv
  , simRng   :: g
  , simSteps :: !Integer
  , simTime  :: !Time
  }

type Disc2dCoord  = (Int, Int)

type SimSF g = SF (Rand g) () SIREnv

type SIREnv = Array Disc2dCoord SIRState

winSize :: (Int, Int)
winSize = (800, 800)

cx, cy, wx, wy :: Int
(cx, cy)   = agentGridSize
(wx, wy)   = winSize

cellWidth, cellHeight :: Double
cellWidth  = (fromIntegral wx / fromIntegral cx)
cellHeight = (fromIntegral wy / fromIntegral cy)

winTitle :: String
winTitle = "Agent-Based SIR on 2D Grid"

agentGridSize :: (Int, Int)
agentGridSize = (27, 28)

mkSimCtx :: RandomGen g
         => SimSF g
         -> SIREnv
         -> g
         -> Integer
         -> Time
         -> SimCtx g
-- creating the specific SimCtx data struct to return
mkSimCtx sf env g steps t = SimCtx {
    simSf    = sf
  , simEnv   = env
  , simRng   = g
  , simSteps = steps
  , simTime  = t
  }

runStepCtx :: RandomGen g
           => DTime
           -> SimCtx g
           -> SimCtx g
runStepCtx dt ctx = ctx'
  where
    g   = simRng ctx
    sf  = simSf ctx

    sfReader            = unMSF sf ()
    sfRand              = runReaderT sfReader dt
    ((env, simSf'), g') = runRand sfRand g

    steps = simSteps ctx + 1
    t     = simTime ctx + dt
    ctx'  = mkSimCtx simSf' env g' steps t

aggregateStates :: [SIRState] -> (Double, Double, Double)
aggregateStates as = (susceptibleCount, infectedCount, recoveredCount)
  where
    susceptibleCount = fromIntegral $ length $ filter (Susceptible==) as
    infectedCount = fromIntegral $ length $ filter (Infected==) as
    recoveredCount = fromIntegral $ length $ filter (Recovered==) as

appendLine :: Csv.ToRecord a => Handle -> a -> IO ()
appendLine hndl line = LBS.hPut hndl (Csv.encode [Csv.toRecord line])

writeSimulationUntil :: RandomGen g
                     => Time
                     -> DTime
                     -> SimCtx g
                     -> String
                     -> IO ()
writeSimulationUntil tMax dt ctx0 fileName = do
    fileHdl <- openFile fileName WriteMode
    appendLine fileHdl ("Susceptible", "Infected", "Recovered")
    writeSimulationUntilAux 0 ctx0 fileHdl
    hClose fileHdl
  where
    writeSimulationUntilAux :: RandomGen g
                            => Time
                            -> SimCtx g
                            -> Handle
                            -> IO ()
    writeSimulationUntilAux t ctx fileHdl
        | t >= tMax = return ()
        | otherwise = do
          let env  = simEnv ctx
              aggr = aggregateStates $ elems env

              t'   = t + dt
              ctx' = runStepCtx dt ctx

          appendLine fileHdl aggr

          writeSimulationUntilAux t' ctx' fileHdl


visualiseSimulation :: RandomGen g
                    => DTime
                    -> SimCtx g
                    -> IO ()
visualiseSimulation dt ctx0 = do
    ctxRef <- newIORef ctx0

    GLOAnim.animateIO

      (GLO.InWindow winTitle winSize (0, 0))
      GLO.white
      (nextFrame ctxRef)
      (const $ return ())

  where
    (cx, cy)   = agentGridSize
    (wx, wy)   = winSize
    cellWidth  = (fromIntegral wx / fromIntegral cx) :: Double
    cellHeight = (fromIntegral wy / fromIntegral cy) :: Double

    nextFrame :: RandomGen g
              => IORef (SimCtx g)
              -> Float
              -> IO GLO.Picture
    nextFrame ctxRef _ = do
      ctx <- readIORef ctxRef

      let ctx' = runStepCtx dt ctx
      writeIORef ctxRef ctx'

      return $ ctxToPic ctx

ctxToPic :: RandomGen g
             => SimCtx g
             -> GLO.Picture
ctxToPic ctx = GLO.Pictures $ aps ++ [timeStepTxt]
      where
          env = simEnv ctx
          as  = assocs env
          aps = map renderAgent as
          t   = simTime ctx

          (tcx, tcy)  = transformToWindow (-7, 10)
          timeTxt     = printf "%0.1f" t
          timeStepTxt = GLO.color GLO.black $ GLO.translate tcx tcy $ GLO.scale 0.5 0.5 $ GLO.Text timeTxt

renderAgent :: (Disc2dCoord, SIRState) -> GLO.Picture
renderAgent (coord, Susceptible)
    = GLO.color (GLO.makeColor 0.0 0.0 0.7 1.0) $ GLO.translate x y $ GLO.Circle (realToFrac cellWidth / 2)
  where
    (x, y) = transformToWindow coord
renderAgent (coord, Infected)
    = GLO.color (GLO.makeColor 0.7 0.0 0.0 1.0) $ GLO.translate x y $ GLO.ThickCircle 0 (realToFrac cellWidth)
  where
    (x, y) = transformToWindow coord
renderAgent (coord, Recovered)
    = GLO.color (GLO.makeColor 0.0 0.70 0.0 1.0) $ GLO.translate x y $ GLO.ThickCircle 0 (realToFrac cellWidth)
  where
    (x, y) = transformToWindow coord

transformToWindow :: Disc2dCoord -> (Float, Float)
transformToWindow (x, y) = (x', y')
      where
        rw = cellWidth
        rh = cellHeight

        halfXSize = fromRational (toRational wx / 2.0)
        halfYSize = fromRational (toRational wy / 2.0)

        x' = fromRational (toRational (fromIntegral x * rw)) - halfXSize
        y' = fromRational (toRational (fromIntegral y * rh)) - halfYSize


animation :: RandomGen g => [SimCtx g] -> DTime -> Time -> SimCtx g
-- bc of the gif time to pictures conversion. Lists of context -> time to context
animation ctxs dt t = ctxs !! floor (t / dt)

runSimulationUntil :: RandomGen g
                   => Time
                   -> DTime
                   -> SimCtx g
                   -> [(Double, Double, Double)]
-- With the max time, time step and initial context, run simulation via the Aux function
runSimulationUntil tMax dt ctx0 = runSimulationAux 0 ctx0 []
  where
    runSimulationAux :: RandomGen g
                      => Time
                      -> SimCtx g
                      -> [(Double, Double, Double)]
                      -> [(Double, Double, Double)]
    runSimulationAux t ctx acc
        | t >= tMax = acc -- if time step is greater than tmax,
        | otherwise = runSimulationAux t' ctx' acc'
      where
        env  = simEnv ctx --
        aggr = aggregateStates $ elems env

        t'   = t + dt -- increase time by timestep
        ctx' = runStepCtx dt ctx -- get new step context
        acc' = aggr : acc


decodeCSV :: BL.ByteString -> Either String (V.Vector (Int, Int, Int))
decodeCSV  = decode NoHeader

getResults :: IO ([(Int, Int)], [(Int, Int)], [(Int, Int)])
getResults = do
    csvData <- BL.readFile "myFile.csv"
    let x = decodeCSV csvData
    case decode NoHeader csvData of
        Left err -> error err
        Right y -> do let (a,b,c) = addToList y
                      let d = zip [1..length a] a --data that I want to use for plotting (trace 1 - S)
                      let e = zip [1..length b] b --data for plotting (line 2)
                      let f = zip [1..length c] c --data for plotting (line 3)
                      pure (d, e, f)

addToList :: V.Vector (Int, Int, Int) -> ([Int], [Int], [Int])
addToList v = unzip3 (V.toList v)


scatterAxis2 :: ([(Int, Int)], [(Int, Int)], [(Int, Int)]) -> Axis B V2 Double
scatterAxis2 (d,e,f)= r2Axis &~ do
    scatterPlot (map (\(x , y ) -> (fromIntegral x , fromIntegral y )) d ) $ key "S"
    scatterPlot (map (\(x , y ) -> (fromIntegral x , fromIntegral y )) e ) $ key "I"
    scatterPlot (map (\(x , y ) -> (fromIntegral x , fromIntegral y )) f ) $ key "R"


main1 :: IO ()
main1 = do
    (d , e , f ) <- getResults
    withArgs [ "-odiagrams/BoardingSchool78.png" ] (r2AxisMain $ scatterAxis2 (d , e , f ))

evaluateCtxs :: RandomGen g => Int -> DTime -> SimCtx g -> [SimCtx g]
evaluateCtxs n dt initCtx = unfoldr g (initCtx, n)
  where
    g (c, m) | m < 0 = Nothing
                   | otherwise = Just (c, (runStepCtx dt c, m - 1))
