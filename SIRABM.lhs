\documentclass{article}
%include polycode.fmt

\usepackage[colorlinks]{hyperref}
\usepackage[pdftex,dvipsnames]{xcolor}
\usepackage{xargs}

\usepackage[colorinlistoftodos,prependcaption,textsize=tiny]{todonotes}
\newcommandx{\unsure}[2][1=]{\todo[linecolor=red,backgroundcolor=red!25,bordercolor=red,#1]{#2}}
\newcommandx{\change}[2][1=]{\todo[linecolor=blue,backgroundcolor=blue!25,bordercolor=blue,#1]{#2}}
\newcommandx{\info}[2][1=]{\todo[linecolor=OliveGreen,backgroundcolor=OliveGreen!25,bordercolor=OliveGreen,#1]{#2}}
\newcommandx{\improvement}[2][1=]{\todo[linecolor=Plum,backgroundcolor=Plum!25,bordercolor=Plum,#1]{#2}}

\setlength{\parskip}{\medskipamount}
\setlength{\parindent}{0pt}

\author{Hrishika Alajpur}

\title{Agent based SIR modelling in Haskell}

\begin{document}

\maketitle

In 1978, anonymous authors sent a note to the British Medical Journal
reporting an influenza outbreak in a boarding school in the north of
England (\cite{bmj-influenza}). The chart below shows the solution of the
SIR (Susceptible, Infected, Record) model with parameters which give
roughly the results observed in the school.

\begin{figure}[h]
    \centering
    \includegraphics[width=0.8\textwidth]{diagrams/Sir.png}
    \caption{Influenza Outbreak in Boarding School (1978)}
    \label{fig:boardingSchool1978}
\end{figure}

\cite{RSPSA1927:115} give a simple model of the spread of an infectious
disease using ordinary differential equations. Individuals move from
being susceptible ($S$) to infected ($I$) to recovered ($R$).

An alternative to modelling via differential equations is to use an
agent-based model (ABM), a computational model which uses a bottom-up
approach to simulate the actions and interactions of the system's
constituent units (agents) to capture the global system
behaviour.\improvement{I wonder if we could get Adrianne to give us a
few words about why ABMs are now popular? I can guess but she really
is an expert.} The autonomous agents are assigned characteristics
(e.g. age) and follow various rules to ensure viable system
dynamics. Spatiality can also be added to simulations.

Traditionally, ABM simulations are implemented as object-oriented
programs due to the intuitive mapping of objects to agents. However,
Functional Reactive Programming can also be used as it can guarantee the
reproducibility of the simulation at compile time and avoid specific
run-time bugs - which cannot be achieved in traditional object-oriented
languages such as C\#.

The goal of this blogpost is to give an example of Agent Based
Modelling in Haskell. It is heavily based on this \cite{thaler} and
the associated \href{https://github.com/thalerjonathan/phd}{github
repo}.

\section{What is Functional Reactive Programming?}

Functional reactive programming (FRP) provides a method of implementing
systems with continuous and discrete time-semantics in pure functional
languages. FRP successfully applied in many domains, such as robotics or
user interfacing.

The central abstraction in FRP is a signal - which is a value that
varies over time. These signals can be composed directly or by composing
signal functions. FRP libaries generally use arrows to implement
functionality rather than monads due to greater efficiency and
modularity. Arrows are a superset of monads, hence have very similar
uses but are more restrictive (see
\href{https://pdf.sciencedirectassets.com/272990/1-s2.0-S1571066106X02438/1-s2.0-S1571066106001666/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEJ7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIFDg9QZeU9kXNykNtK1Y3q77oHuug09p5ti63It1vccTAiBFKp5vMEkDr9v8O6lU%2FGJEBt1WPAktubolt%2FM7Ft0eRyqzBQhXEAUaDDA1OTAwMzU0Njg2NSIMm1WFiIp%2BgTN43MxSKpAF4WtU%2BhIjuCHa7o1OJ%2FJ5MHrwqLRo%2BnhAJ7pK%2BeBHF6xEGRXFSUKaELPIEgO6CAoqLitoqxUvWasKIWiBouge9w%2FZhcoQKARlX6lw6dYnQktYC%2BzakAwobpJa1s2LEWugYNdL9j2bfEZZJ3AtIqbUlwtvbBhqAEyKxkEQHlolc9cHyaq17Vv6%2FgfPDeE%2BqM6cM5I025Q4nQcAK4cQMzb66%2BxiIVpcAGVR8gSprc7GfKIpQlVO17l7aPJ33%2FDrrNMG28QydoXVKseJN2bkGk6%2BnjlP0TItMeFC5r%2FEtx%2BpgYTS%2BpnS%2FPxWeCDUMq4HfTNB9qVMipypeP98oeGMR92%2BNjp2uhF9X2a%2BHJbdOOQVV3IxKsT5JjSJ%2FzJhp6ieJ%2BPM7%2BCSIsCQ7Udf%2FaDo0KeX6KsGtCNFh0zJFwNyel6sx1b5yPXG8t7rgC2KAGV1kaYpggNZKMoGHFDt0HUOQCIan0hiP9vW8GvX0HjI1%2FH3Kajx0qVKnbky%2FbNfGmj9gMCJBfxYpRNvO0as%2FtWBngW7QQHIQR2opY400yJxE%2BdVFCgfCDaYBuDl1sckh7O%2FbDKT%2FQtEHmRlUdYkg8BIHYsqC%2FE3nt5Sqo83JGpaL7%2F0R0TyioO8EazXMwWVDK%2B%2Fbn4AhmoxIEPuAhw%2FgkfAUtn9o7a3iZF%2BHT%2Fv6gGxKLJkI40OzkM3nTlHmd93Yceamm%2Bj%2BHUcMzkqyUxyVUGQLEuTFWJ8DrFGsLqE%2F0h9A4y1sGV48%2BYXiiRipqitUyx7fW%2F1JRuY8sPsJmLc4CnIihYXIhIhzm06xqmouOaOEuSlAFvnAuvy6f09mD8VCC8H2E%2BtqkmUciVlg4fVM3mMRHdePbtNJ8HxyaN6143%2BMD8GVVkwzdqloAY6sgFkA1yE%2BQykIX6iO2s8d%2Bsr4dc1A4L23Kw4TymTEusts%2BVxCXWhF2CU23C3MxIddMHinkuiv8u6J6tcehBM05NTGXotQgTLeL4IASQ6%2BNdj6YWRjHZ5A77E8qBMap%2FWzGzpDXLAJ4hDZWktDQ4KMpW446reCDlw0ViRPJTzaSPsCcuElkjGI34WpARqHLN149%2FSs%2FcIFWBqwhG0%2F93p3zgWt5gkv9u5lbg6ZOYZ1Bjgzm9z&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20230309T062929Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY3TG3MPND%2F20230309%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=ae795e67238960dc7bcdc97d21a32e3c73d229d94f6ccf87eecdd2eb4cb17ccb&hash=6109688cb98c98a108b74ea3b2556b59fcada4fd8ea4d0942973369e63ce2e0e&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1571066106001666&tid=spdf-0f923ac0-927a-479a-84e6-8376df53a474&sid=749e13d39ace604f5e2bba26f52d6131b5ccgxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=04125300565756535e54&rr=7a514386fa4faafb&cc=au}{here}
for more on arrows vs monads).

The central concept of arrowised FRP is the Signal Function (SF). The SF
represents a process overtime which maps an input signal to an output
signal. Thus, signifying that SFs have an awareness of the passing of
time through the timestep of the system\unsure{I don't know what you are trying to say here}. This concept of using
time-varying functions as a method of handling agent-based models is
essential due to their time-dependent nature (see more
[here](https://www.cs.yale.edu/publications/techreports/tr1049.pdf)).

Futher reading on FRP concepts can be found \href{https://ivanperez.io/papers/2016-HaskellSymposium-Perez-Barenz-Nilsson-FRPRefactored-short.pdf}{here}.

\section{SIR model using Functional Reactive Programming}

The goal is to use a Functional Reactive Programming (FRP) approach in
Haskell to simulate a SIR (Susceptible, Infected, Recovered) model -
which is a simple compartmental model. The general result within such a
SIR model is as below:\change{I was unable to get these PNGs. Can we store them in the repo?}

\begin{figure}[h]
    \centering
    \includegraphics[width=0.8\textwidth]{diagrams/Temp1.png}
    \caption{Temporary Chart 1}
    \label{fig:temp1}
\end{figure}

%% [Screen%20Shot%202023-02-15%20at%203.49.56%20pm.png](attachment:Screen%20Shot%202023-02-15%20at%203.49.56%20pm.png)

The following sections describe the approach taken to achieve the above
trend using a FRP approach in Haskell.

\section{Setting up SIR simulation using Haskell}

\subsection{Language Extensions}

Language extensions are used to enable certain features in Haskell. In
this case, we require the following features: - `Arrows` : to support
the arrow notations - `Strict` : to allows functions parameters to be
evaluated before calls - `FlexibleContexts` : loosens restrictions on
what constraints can be used in a typeclass

\begin{code}
{-# LANGUAGE Arrows #-}
{-# LANGUAGE Strict #-}
{-# LANGUAGE FlexibleContexts #-}
\end{code}

\subsection{Loading Modules}

Various modules required for the different aspect of simulation must be
loaded.

\begin{code}
import           Data.IORef
import           System.IO
import           Text.Printf

import           Control.Monad.Random
import           Control.Monad.Reader
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
import Plots.Axis.Render (renderAxis)

import Control.Lens ((&~))
import Diagrams.Backend.Cairo (B)
import Diagrams.TwoD.Types (V2)
import IHaskell.Display.Diagrams hiding (animation)
import Plots (scatterPlot, key)

import qualified Data.ByteString.Lazy as BL
import Data.Csv
import qualified Data.Vector as V
\end{code}

\subsection{SIR States as a Algebraic Data Type}

The main governing property of the agents within the SIR simulation is
their state - which can be either Susceptible, Infected or Recovered.
Here, these possible states are described as a Algebraic data type (ADT)
to ensure ease of assignment later.

Additionally, a deriving (Show, Eq) clause is used to ensure the
complier automatically generates instances of the Eq and Show classes
for the ADT.

\begin{code}
data SIRState = Susceptible | Infected | Recovered deriving (Show, Eq)
\end{code}

Define 2D environment

In this model, a N x A grid is used to define the spatial aspect of the
model. Here, the discrete 2D environment is defined with a tuple.

\begin{code}
type Disc2dCoord  = (Int, Int)
\end{code}

Define types to store agent's environment

During the simulation, the location and state of each agent must be
retrievable. Hence, the type `SIREnv` is defined to allow agents to
store their location coordinate and their state in an array.

\begin{code}
type SIREnv       = Array Disc2dCoord SIRState
\end{code}

Next, to aid readability, we define a type synonym for `Rand` monad
which allows us to obtain random numbers parameterised over the random
number generator `g`

\begin{code}
type SIRMonad g   = Rand g
\end{code}

Define the SIR Agent

`SIRAgent` is used to define Agents as Signal functions which receives
the SIR states of all agents as input and outputs the current SIR state
of the agent. It also returns the output in a `SIRMonad` context - which
is building the monad over a rand monad.

\begin{code}
type SIRAgent g   = SF (SIRMonad g) SIREnv SIRState
\end{code}

`SimSF` type is similar to `SIRagent` above, however it does not receive
any inputs and just outputs the current location and state of the agent.
This is useful in instances where states of neighbours must be accessed
by an agent to determine whether infection occurs.

\begin{code}
type SimSF g = SF (SIRMonad g) () SIREnv
\end{code}

Defining simulation context

A simulation context data struct which contains the various parameters
(e.g. simulation time) is defined in terms of their type. The ! is used
to specific it is a strictness declaration.

\begin{code}
data SimCtx g = SimCtx
  { simSf    :: !(SimSF g)
  , simEnv   :: !SIREnv
  , simRng   :: g
  , simSteps :: !Integer
  , simTime  :: !Time
  }
\end{code}

Simulation Rules

In this simulation, the rules are: - There is a population of size N in
which all agents are either Susceptible, Infected or Recovered at a
particular time. - Initially, there is at least one infected person in
the population. - People interact with each other on average with a
given rate of β per time-unit. - People become infected with a given
probability γ when interacting with an infected person. - When infected,
a person recovers on average after δ time-units. - An infected person is
immune to further infections (no reinfection). - The 2D environment has
either Moore or von Neumann neighbourhood. - Agents are either static or
can move freely around the cells. - The cells allow either single or
multiple occupants. - Agents can read the states of all their neighbours
which tells them if a neighbour is infected or not.

Simulation Parameters set up

To enforce the simulation rules, various simulation parameters are
defined.

\begin{code}
-- contact rate or β
contactRate :: Double
contactRate = 5.0

-- Infection rate or γ
infectivity :: Double
infectivity = 0.05

-- Recovery rate or δ
illnessDuration :: Double
illnessDuration = 15.0

-- 2D grid size
agentGridSize :: (Int, Int)
agentGridSize = (51, 51)
\end{code}

The outputs of the simulation will be a CSV file containing the data and
an animation, hence parameters relevant to these are set.

\begin{code}
-- window size for animation
winSize :: (Int, Int)
winSize = (800, 800)

-- parameters for gif
cx, cy, wx, wy :: Int
(cx, cy)   = agentGridSize
(wx, wy)   = winSize

-- parameters for rendering agents
cellWidth, cellHeight :: Double
cellWidth  = (fromIntegral wx / fromIntegral cx)
cellHeight = (fromIntegral wy / fromIntegral cy)


-- window title for animation
winTitle :: String
winTitle = "Agent-Based SIR on 2D Grid"
\end{code}

Defining helper functions for the simulation

To ensure ease of understanding and seperation of functionality, various
helper functions are defined for the different aspects of simulation
(e.g. for generating the animation).

Functions related to the simulation context

`mkSimCtx` and `runStepCtx` are used to initalise the simulation context
and update the simulation context respectively.

\begin{code}
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
\end{code}

`evaluateCtxs` takes the initial context and creates a list of new
contexts by running the simulation step.

\begin{code}
evaluateCtxs :: RandomGen g => Int -> DTime -> SimCtx g -> [SimCtx g]
evaluateCtxs n dt initCtx = unfoldr g (initCtx, n)
  where
    g (c, m) | m < 0 = Nothing
                   | otherwise = Just (c, (runStepCtx dt c, m - 1))
\end{code}

Functions related to Boundary Conditions

Either Neumann or Moore boundary conditions (BC) are used within
simulation. The areas surveyed in these is as represented in the
following diagram:
![Screen%20Shot%202023-02-22%20at%205.28.13%20pm.png](attachment:Screen%20Shot%202023-02-22%20at%205.28.13%20pm.png)

The following code is related to enforcing these BC:

\begin{code}
-- Neumann BC
neumann :: [Disc2dCoord]
neumann = [ topDelta, leftDelta, rightDelta, bottomDelta ]

-- Moore BC
moore :: [Disc2dCoord]
moore = [ topLeftDelta,    topDelta,     topRightDelta,
          leftDelta,                     rightDelta,
          bottomLeftDelta, bottomDelta,  bottomRightDelta ]

-- different Delta values for the BC
topLeftDelta :: Disc2dCoord
topLeftDelta      = (-1, -1)
topDelta :: Disc2dCoord
topDelta          = ( 0, -1)
topRightDelta :: Disc2dCoord
topRightDelta     = ( 1, -1)
leftDelta :: Disc2dCoord
leftDelta         = (-1,  0)
rightDelta :: Disc2dCoord
rightDelta        = ( 1,  0)
bottomLeftDelta :: Disc2dCoord
bottomLeftDelta   = (-1,  1)
bottomDelta :: Disc2dCoord
bottomDelta       = ( 0,  1)
bottomRightDelta :: Disc2dCoord
bottomRightDelta  = ( 1,  1)
\end{code}

Functions related to using RNG elements

`drawRandomElemS` and `randomBoolM` are related to use of RNG elements
in simulation. A specific example is `RandomBoolM` being used to draw a
random Boolean to determine whether an infection occurs in the
susceptibleAgent function.

\begin{code}
randomBoolM :: RandomGen g => Double -> Rand g Bool
randomBoolM p = getRandomR (0, 1) >>= (\r -> return $ r <= p)

drawRandomElemS :: MonadRandom m => SF m [a] a
drawRandomElemS = proc as -> do
  r <- getRandomRS ((0, 1) :: (Double, Double)) -< ()
  let len = length as
  let idx = (fromIntegral len * r)
  let a =  as !! (floor idx)
  returnA -< a
\end{code}

Functions related to defining/governing agent behaviour

Aspects related to the behaviour of agents are defined via the various
functions below.

`initAgentsEnv` initialises the simulation environment.

\begin{code}
initAgentsEnv :: (Int, Int) -> ([(Disc2dCoord, SIRState)], SIREnv)
initAgentsEnv (xd, yd) = (as, e)
  where
    xCenter = floor $ fromIntegral xd * (0.5 :: Double)
    yCenter = floor $ fromIntegral yd * (0.5 :: Double)
    -- populating the grid with susceptible agents everywhere except the centre
    sus = [ ((x, y), Susceptible) | x <- [0..xd-1],
                                    y <- [0..yd-1],
                                    x /= xCenter ||
                                    y /= yCenter ]
    -- populating the infected agent at the center
    inf = ((xCenter, yCenter), Infected)
    -- list of infected and susceptible agent locations
    as = inf : sus
    -- array of min and max grid size and the list of agent locations
    e = array ((0, 0), (xd - 1, yd - 1)) as
\end{code}

In order to observe the general SIR trend, infection via observing the
neighbours must be viable. The neighbours function takes the coordinates
and returns the surround SIRstates. `allNeighbours` returns the state of
all neighbours or agents.

\begin{code}
neighbours :: SIREnv
           -> Disc2dCoord
           -> Disc2dCoord
           -> [Disc2dCoord]
           -> [SIRState]
neighbours e (x, y) (dx, dy) n = map (e !) nCoords'
  where
    nCoords  = map (\(x', y') -> (x + x', y + y')) n
    nCoords' = filter (\(nx, ny) -> nx >= 0 &&
                                    ny >= 0 &&
                                    nx <= (dx - 1) &&
                                  ny <= (dy - 1)) nCoords


allNeighbours :: SIREnv -> [SIRState]
allNeighbours = elems
\end{code}

Unlike the other states, the recovered state does not generate any event
and rather just acts a sink which constantly returns Recovered.

\begin{code}
recoveredAgent :: RandomGen g => SIRAgent g
recoveredAgent = arr (const Recovered)
\end{code}

The function below describes the behaviour of an infected agent. This
behaviour is governed by either recovering on average after delta time
units or staying infected within a timestep.

\begin{code}
infectedAgent :: RandomGen g => SIRAgent g
infectedAgent
    = switch
      -- delay the switching by 1 step, otherwise could
      -- make the transition from Susceptible to Recovered within time-step
      (infected >>> iPre (Infected, NoEvent))
      (const recoveredAgent)
  where
    infected :: RandomGen g => SF (SIRMonad g) SIREnv (SIRState, Event ())
    infected = proc _ -> do
      -- illness duration = infected agent
      recovered <- occasionally illnessDuration () -< ()
      if isEvent recovered
        -- otherwise recovered agent as the updated state
        then returnA -< (Recovered, Event ())
        else returnA -< (Infected, NoEvent)
\end{code}

`susceptibleAgent` describes the behaviour of a susceptible agent -
which is governed by querying the surrounding neighbours and either
getting infected based on the parameter γ (event generated) or staying
susceptible (no event)

\begin{code}
susceptibleAgent :: RandomGen g => Disc2dCoord -> SIRAgent g
susceptibleAgent _coord
    = switch
      -- delay the switching by 1 step, otherwise could
      -- make the transition from Susceptible to Recovered within time-step
      (susceptible >>> iPre (Susceptible, NoEvent))
      (const infectedAgent)
  where
    susceptible :: RandomGen g
                => SF (SIRMonad g) SIREnv (SIRState, Event ())
    susceptible = proc env -> do
      -- use occasionally to make contact on average
      makeContact <- occasionally (1 / contactRate) () -< ()

      if not $ isEvent makeContact
        then returnA -< (Susceptible, NoEvent)
        else (do
          -- take env, the dimensions of grid and neighbourhood info
          --let ns = neighbours env coord agentGridSize moore
          -- queries the environemtn for its neighbours - in this case appears to be all neighbours
          let ns = allNeighbours env
          s <- drawRandomElemS -< ns -- randomly selects one
          case s of
            Infected -> do
              infected <- arrM (const (lift $ randomBoolM infectivity)) -< ()
              -- upon infection,
              if infected
                -- event returned which returns in switching into the infected agent SF (to behave as such)
                then returnA -< (Infected, Event ())
                else returnA -< (Susceptible, NoEvent)
            _       -> returnA -< (Susceptible, NoEvent))
\end{code}

`sirAgent` defines the behaviour of the agent depending on the initial
state. Only the suspectible agent receives the coordinates as the
infected and recovered agents do not require this information.

\begin{code}
sirAgent :: RandomGen g => Disc2dCoord -> SIRState -> SIRAgent g
sirAgent coord Susceptible = susceptibleAgent coord
sirAgent _     Infected    = infectedAgent
sirAgent _     Recovered   = recoveredAgent -- recovered agent ignores gen bc they stay immune
\end{code}

The simulationStep function is a closed feedback loop which takes the
current signal functions and returns the new agent states.

\begin{code}
simulationStep :: RandomGen g
               => [(SIRAgent g, Disc2dCoord)]
               -> SIREnv
               -> SF (SIRMonad g) () SIREnv
simulationStep sfsCoords env = MSF $ \_ -> do
    let (sfs, coords) = unzip sfsCoords

    -- run all agents sequentially but keep the environment
    -- read-only: it is shared as input with all agents
    -- and thus cannot be changed by the agents themselves
    -- run agents sequentially but with shared, read-only environment
    ret <- mapM (`unMSF` env) sfs
    -- construct new environment from all agent outputs for next step
    let (as, sfs') = unzip ret
        env' = foldr (\(coord, a) envAcc -> updateCell coord a envAcc) env (zip coords as)

        sfsCoords' = zip sfs' coords
        cont       = simulationStep sfsCoords' env'
    return (env', cont)
  where
    updateCell :: Disc2dCoord -> SIRState -> SIREnv -> SIREnv
    updateCell c s e = e // [(c, s)]
\end{code}

Functions related to collating simulation data

`aggregateStates` is used to collate the number of susceptible, infected
and recovered agents within the simulation. This function is used both
for the animation and the plot.

\begin{code}
aggregateStates :: [SIRState] -> (Double, Double, Double)
aggregateStates as = (susceptibleCount, infectedCount, recoveredCount)
  where
    susceptibleCount = fromIntegral $ length $ filter (Susceptible==) as
    infectedCount = fromIntegral $ length $ filter (Infected==) as
    recoveredCount = fromIntegral $ length $ filter (Recovered==) as
\end{code}

Functions for generating CSV file

`appendLine` is a helper function used to write the counts of the S, I,
R states into the CSV file.

\begin{code}
appendLine :: Csv.ToRecord a => Handle -> a -> IO ()
appendLine hndl line = LBS.hPut hndl (Csv.encode [Csv.toRecord line])
\end{code}

`writeSimulationUntil` uses the above auxilliary functions to generate
the overall CSV file

\begin{code}
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
\end{code}

Functions for generating the animation

`visualiseSimulation` generates and updates the animation.

\begin{code}
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
\end{code}

`animation` is used to map a list of contexts to time - which is
required for the function that produces the gif.

\begin{code}
animation :: RandomGen g => [SimCtx g] -> DTime -> Time -> SimCtx g
-- bc of the gif time to pictures conversion. Lists of context -> time to context
animation ctxs dt t = ctxs !! floor (t / dt)
\end{code}

Functions for running simulation

`runSimulationUntil` runs the overall simulation for the desired
duration.

\begin{code}
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
\end{code}

Main Function

The `main` function below sets up the general simulation via various
steps and enables a method of getting output either through an animation
or through a CSV file output and a GIF of the animation.

\begin{code}
main :: IO ()
main = do
  hSetBuffering stdout NoBuffering

  let visualise = False
      t         = 100
      dt        = 0.1
      seed      = 123 -- 42 leads to recovery without any infection

      g         = mkStdGen seed
      (as, env) = initAgentsEnv agentGridSize
      sfs       = map (\(coord, a) -> (sirAgent coord a, coord)) as
      sf        = simulationStep sfs env
      ctx       = mkSimCtx sf env g 0 0

  if visualise
    then visualiseSimulation dt ctx
    else do
      let ts = [0.0, dt .. t]
      let ctxs = evaluateCtxs (length ts) dt ctx
      exportPicturesToGif 10 LoopingForever (800, 800) GLO.white "SIR.gif" ((ctxToPic . (animation ctxs dt)) . uncurry encodeFloat . decodeFloat) (map (uncurry encodeFloat . decodeFloat) ts)
      writeSimulationUntil t dt ctx "SIR_DUNAI_dt001.csv"
\end{code}

Results

Chart to be finalised. Further details on the results to be added after
this.

\begin{code}
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

-- (d,e,f) <- getResults

scatterAxis2 :: ([(Int, Int)], [(Int, Int)], [(Int, Int)]) -> Axis B V2 Double
scatterAxis2 (d, e, f) = r2Axis &~ do
    scatterPlot (map (\(x,y) -> (fromIntegral x, fromIntegral y)) d) $ key "S"
    scatterPlot (map (\(x,y) -> (fromIntegral x, fromIntegral y)) e) $ key "I"
    scatterPlot (map (\(x,y) -> (fromIntegral x, fromIntegral y)) f) $ key "R"
--
-- scatterExample2 = renderAxis scatterAxis2
-- diagram scatterExample2
main1 = do
  (d,e,f) <- getResults
  let scatterExample2 = renderAxis $ scatterAxis2 (d, e, f)
  let diagram2 = diagram scatterExample2
  return ()
\end{code}

![svg](output_82_0.svg)

The GIF below shows the spread of infection overtime

![SegmentLocal](SIR.gif "segment")

\section{Bibliography}

\bibliographystyle{plain}
\bibliography{SIRABM}

\end{document}
