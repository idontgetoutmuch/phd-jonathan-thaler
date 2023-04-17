\documentclass{article}
%include polycode.fmt

%format -< = "\leftbroom"
%format >>> = "\ggg"
%format <<< = "\lll"
%format contactRate = "\beta"
%format infectivity = "\gamma"
%format illnessDuration = "\delta"

\usepackage[colorlinks]{hyperref}
\usepackage[pdftex,dvipsnames]{xcolor}
\usepackage{xargs}
\usepackage{halloweenmath}

\usepackage{biblatex}
\addbibresource{SIRABM.bib}

\input{answer.tex}

\usepackage[obeyDraft,colorinlistoftodos,prependcaption,textsize=tiny]{todonotes}
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

\listoftodos

%if style == newcode
\begin{code}
{-# LANGUAGE Arrows #-}
{-# LANGUAGE Strict #-}
{-# LANGUAGE FlexibleContexts #-}

module Main (main, main1, main2, main3, decodeCSV, runSimulationUntil, moore, neumann) where

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
import Diagrams.Prelude hiding (Time, (^/), (^+^), trace, coords)
import System.Environment

import qualified Data.ByteString.Lazy as BL
import Data.Csv
import qualified Data.Vector as V

import Control.Monad.State

import Debug.Trace

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
\end{code}
%endif

\section{Introduction}

In 1978, anonymous authors sent a
note,\citetitle{bmj-influenza}~\cite{bmj-influenza}, to the British
Medical Journal reporting an influenza outbreak in a boarding school
in the north of England . The chart below shows the solution of the
SIR (Susceptible, Infected, Record) model with parameters which give
roughly the results observed in the school.\change{This should be the
actual data not the output of our model}

\begin{figure}[h]
    \centering
    \includegraphics[width=0.8\textwidth]{diagrams/BoardingSchool78.png}
    \caption{Influenza Outbreak in Boarding School (1978)}
    \label{fig:boardingSchool1978}
\end{figure}

In \cite{RSPSA1927:115},
\citeauthor{RSPSA1927:115}~(\citedate{RSPSA1927:115}) give a simple
model of the spread of an infectious disease using ordinary
differential equations. Individuals move from being susceptible ($S$)
to infected ($I$) to recovered ($R$).

An alternative to modelling via differential equations is to use an
agent-based model (ABM), a computational model which uses a bottom-up
approach to simulate the actions and interactions of the system's
constituent units (agents) to capture the global system
behaviour.\improvement[inline]{I wonder if we could get Adrianne to give us a
few words about why ABMs are now popular? I can guess but she really
is an expert.} The autonomous agents are assigned characteristics
(e.g. age) and follow various rules to ensure viable system
dynamics. Spatiality can also be added to simulations.

Traditionally, ABM simulations are implemented as object-oriented
programs due to the intuitive mapping of objects to agents. However,
Functional Reactive Programming (which we explain below) can also be
used as it can guarantee the reproducibility of the simulation at
compile time and avoid specific run-time bugs - which cannot be
achieved in traditional object-oriented languages such as C\#.\unsure{This doesn't sound quite right}

The goal of this blogpost is to give an example of Agent Based
Modelling in Haskell. It is heavily based on \citetitle{thaler} by
\citeauthor{thaler}~\cite{thaler} and the associated
\href{https://github.com/thalerjonathan/phd}{github repo}. It is
targetted at the reader experienced in ABM but with no knowledge of
FRP or Haskell.

\section{What is Functional Reactive Programming?}

In many models, we are interested in how they vary over time. Handling
time explicitly has proven to be difficult and
error-prone.\change{citation needed}. In Functional Reactive
Programming (FRP), instead of so doing, we move from the domain of
data and functions on that data, to the domain of data and
{\it{time-varying}} functions on that data {\it{together with}}
methods of operating on these time-varying functions. These operators
are usually referred to as combinators and need to be sufficiently
rich to allow models of interest to be captured in such a framework.

FRP successfully applied in many domains, such as robotics or user interfacing.\improvement{We could do with a few more examples with a bit more detail}

As already hinted, the central abstraction in FRP is a signal, which
is a value that varies over time: |Time -> value|. We need to be able
to lift an ordinary function to be a time-varying function. This is
done using the |arr| operator. Figure~\ref{fig-arrOperator} depicts a
function |f| being lifted to be a time-varying process. We can look at
the specification of |arr|: |arr :: Monad m => (b -> c) -> MSF m b c|
which tells us that |arr| takes a function from |b| to |c|, |f :: b ->
c| and returns a time-varying value denoted |MSF m b c|. The reader
can safely ignore the |m| and the constraint |Monad m| on the left
hand side of the |=>|.\improvement{Why are they there though?}

These signals can be composed directly or by composing
signal functions. FRP libaries generally use arrows to implement
functionality rather than monads due to greater efficiency and
modularity. Arrows are a superset of monads, hence have very similar
uses but are more restrictive (see~\cite{HEUNEN2006219} for more on arrows vs monads).

The central concept of arrowised FRP is the Signal Function (SF). The SF
represents a process overtime which maps an input signal to an output
signal. Thus, signifying that SFs have an awareness of the passing of
time through the timestep of the system\unsure[inline]{I don't know what you are trying to say here}. This concept of using
time-varying functions as a method of handling agent-based models is
essential due to their time-dependent nature (see more
\href{https://www.cs.yale.edu/publications/techreports/tr1049.pdf}{here}.

Futher reading on FRP concepts can be found \href{https://ivanperez.io/papers/2016-HaskellSymposium-Perez-Barenz-Nilsson-FRPRefactored-short.pdf}{here}.

\begin{figure}
\[
\input{arrOperator.tex}
\]
\caption{|arr| combinator}
\label{fig-arrOperator}
\end{figure}

These time-varying process can be composed together using the |>>>| operator: |(>>>) :: Monad m => MSF m a b -> MSF m b c -> MSF m a c|. This is an infix function which takes two time-varying process and creates a new one that takes the output from the first process and makes it the iput of the second process.

\begin{figure}
\[
\input{functionAsArrow.tex}
\]
\caption{|>>>| combinator}
\label{fig-rararaOperator}
\end{figure}

An example should clarify how such combinators can be used to describe an agent. A falling ball is an object which falls under gravity starting parameterised by a starting position and velocity. This time-varying process takes no input (the only value of type |()| is |()|) and produces a signal (a time-varying value) of the ball's position and velocity.

We can read the code below as take the accelaration (here
gravitational constant $g$), integrate it and add the starting
velocity then integrate the velocity and add the starting velocity.

\begin{code}
type Pos = Double
type Vel = Double

fallingBall1 :: MonadState Int m => Pos -> Vel -> SF m () (Pos, Vel)
fallingBall1 p0 v0 = proc () -> do
  let g = -9.81
  v <- arr (\x -> v0 + x) <<< integral -< g
  p <- arr (\y -> p0 + y) <<< integral -< v
  arrM_ (lift (modify (+1))) -< ()
  returnA -< (p,v)
\end{code}

\begin{figure}[h]
    \centering
    \includegraphics[width=0.8\textwidth]{diagrams/fallingBall.png}
    \caption{Falling Ball}
    \label{fig:fallingBall}
\end{figure}

Figure~\ref{fig:fallingBall} shows the execution of such a process.

%if style == newcode
\begin{code}

main3 :: IO ()
main3 = do
  withArgs ["-odiagrams/fallingBall.png"] (r2AxisMain (jSaxis (fallingBall1 10 10)))
  withArgs ["-odiagrams/bouncingBall.png"] (r2AxisMain (jSaxis (bouncingBall 10 10)))

addPoint :: (Plotable (Diagram B) b, MonadState (Axis b V2 Double) m) =>
            Double -> (Double, Double) -> m ()
addPoint o (x, y) = addPlotable'
                    ((circle 1e0 :: Diagram B) #
                     fc brown #
                     opacity o #
                     translate (r2 (x, y)))

jSaxis :: SF (StateT Int (WriterT [(Double, Double)] Identity)) () (Double, Double)
       -> Axis B V2 Double
jSaxis msf = r2Axis &~ do
  let preMorePts = runMSFDet' 0 msf
  let morePts = map p2 $ preMorePts
  let l = length preMorePts
  let os = [0.05,0.1..]
  let ps = take (l `div` 4) [0,4..]
  zipWithM_ addPoint os (map (preMorePts!!) ps)
  linePlot' $ map unp2 $ take 200 morePts

arrM_ :: Monad m => m b -> MSF m a b
arrM_ = arrM . const

fallingBall :: MonadState Int m => Pos -> Vel -> SF m () (Pos, Vel)
fallingBall p0 v0 = proc () -> do
  let g = -9.81
  v <- arr (\x -> v0 + x) <<< integral -< g
  p <- arr (\y -> p0 + y) <<< integralLinear v0 -< v
  arrM_ (lift (modify (+1))) -< ()
  returnA -< (p,v)

integralLinear :: Monad m => Double -> SF m Double Double
integralLinear initial = average >>> integral
  where
    average = (arr id &&& iPre initial) >>^ (\(x, y) -> (x ^+^ y) ^/ 2)

runMSFDet' :: Int ->
              SF (StateT Int (WriterT [(Double, Double)] Identity)) () (Double, Double) ->
              [(Double, Double)]
runMSFDet' s msf = snd $ runWriter (runMSFDetAux s msf)
  where
    runMSFDetAux s0 msf0 = do
      let msfReaderT = unMSF msf0 ()
          msfStateT  = runReaderT msfReaderT 0.1
          msfRand    = runStateT msfStateT s0
      ((p, msf'), s') <- msfRand
      tell [p]
      when (s' <= 100) (runMSFDetAux s' msf')

runStep :: DTime ->
           ((Double, SF (StateT Int IO) () Double), Int) ->
           IO ((Double, SF (StateT Int IO) () Double), Int)
runStep dt ((_t, msf), n) = msfRand
  where
    sfReader = unMSF msf ()
    sfRand   = runReaderT sfReader dt
    msfRand  =  runStateT sfRand n

visualiseSimulation2 :: DTime
                     -> ((Double, SF (StateT Int IO) () Double), Int)
                     -> IO ()
visualiseSimulation2 dt ((p, msf), n) = do
  ctxRef <- newIORef ((p, msf), n)
  GLOAnim.animateIO (GLO.InWindow "foo" (800, 800) (0, 0))
                    GLO.blue
                    (nextFrame ctxRef)
                    (const $ return ())
  where
    nextFrame :: IORef ((Double, SF (StateT Int IO) () Double), Int)
              -> Float
              -> IO GLO.Picture
    nextFrame ctxRef _ = do
      ((p0, msf0), n0) <- readIORef ctxRef
      putStrLn $ show p0
      ctx' <- runStep dt ((p0, msf0), n0)
      writeIORef ctxRef ctx'
      return $ GLOAnim.translate 0.0 (fromRational $ toRational p) (GLOAnim.thickCircle 10.0 100.0)

main :: IO ()
main = visualiseSimulation2 0.1 ((10.0, (bouncingBall 100.0 0.0 >>> arr fst)), 0)
\end{code}
%endif

We will need a few more combinators to show how to create an ABM, the first of which is |switch :: Monad m => SF m a (b, Event c) -> (c -> SF m a b) -> SF m a b|. An |Event| is defined as |data Event a = NoEvent || Event a|. |switch| takes

\begin{enumerate}

\item
A time-varying process, the output of which is pair with the second element of this pair either returning |NoEvent| or returning |Event x| (for some |x| of type |c|).

\item
A function which takes a value of type |c| and returns a time-varying process.

\end{enumerate}

In the event (pun intended) of |NoEvent|, |switch| returns the first time-varying process (without the event information); if there is an event then |switch| applies the function that is the second argument and returns the time-varying process so created.

Let us modify our example of a falling object to create a bouncing ball.

\begin{code}
bouncingBall :: MonadState Int m => Double -> Double -> SF m () (Double, Double)
bouncingBall p0 v0 =
  switch (fallingBall p0 v0 >>> (arr id &&& hitFloor))
         (\(p,v) -> bouncingBall p (-v))

hitFloor :: Monad m => SF m (Double,Double) (Event (Double,Double))
hitFloor = arr $ \(p,v) ->
  if p < 0 && v < 0 then Event (p,v) else noEvent
\end{code}

\begin{figure}[h]
    \centering
    \includegraphics[width=0.8\textwidth]{diagrams/bouncingBall.png}
    \caption{Bouncing Ball}
    \label{fig:bouncingBall}
\end{figure}

Figure~\ref{fig:bouncingBall} shows the execution of such a process.

\section{SIR model using Functional Reactive Programming}

The goal is to use a Functional Reactive Programming (FRP) approach in
Haskell to simulate a SIR (Susceptible, Infected, Recovered) model -
which is a simple compartmental model. The general result within such a
SIR model is as below:\change[inline]{I was unable to get these PNGs. Can we store them in the repo?}

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

\subsection{SIR States as a Algebraic Data Type}

The main governing property of the agents within the SIR simulation is
their state - which can be either |Susceptible|, |Infected| or |Recovered|.
Here, these possible states are described as a Algebraic data type (ADT)
to ensure ease of assignment later.

Additionally, a deriving |(Show, Eq)| clause is used to ensure the
complier automatically generates instances of the Eq and Show classes
for the ADT.

\begin{code}
data SIRState = Susceptible | Infected | Recovered deriving (Show, Eq)
\end{code}

\subsection{Define 2D environment}

For a bit more a challenge than a simple compartmental model, let us
model each agent as being at fixed co-ordinates on a grid. By
modifying how other agents on the grid could be infected, we can
simulate how an epidemic could spread through a population spread over
the grid. For example, we might model that agents can only be infected
if they are in physical contact with an already infected agent.

A $N \times M$ grid is used to define the spatial aspect of the
model. Here, the discrete 2D environment is defined with a tuple. Were
this to be modelled via differential equations, we would have to use
partial differential equations. With one time dimension and two
spatial dimensions, such differential equations are amenable to
numerical methods but if the dimensions were significantly higher then
ABMs start to show their advantage.

\begin{code}
type Disc2dCoord  = (Int, Int)
\end{code}

\subsection{Define types to store agent's environment}

During the simulation, the location and state of each agent must be
retrievable. Hence, the type `SIREnv` is defined to allow agents to
store their location coordinate and their state in an array.\unsure[inline]{I think there must be an assumption that every position contains an agent and this agent is either Susceptible, Infected or Recovered. Do you agree? And if so, can we make this explicit?}

\begin{code}
type SIREnv = Array Disc2dCoord SIRState
\end{code}

\subsection{Define the SIR Agent}

\textit{SIRAgent} is used to define Agents as Signal functions which receives
the SIR states of all agents as input and outputs the current SIR state
of the agent. It also returns the output in a \textit{Rand} context - which
is building the monad over a rand monad.

\begin{code}
type SIRAgent g = SF (Rand g) SIREnv SIRState
\end{code}

\textit{SimSF} type is similar to \textit{SIRagent} above, however it does not receive
any inputs and just outputs the current location and state of the agent.
This is useful in instances where states of neighbours must be accessed
by an agent to determine whether infection occurs.

\begin{code}
type SimSF g = SF (Rand g) () SIREnv
\end{code}

\subsection{Defining simulation context}

A simulation context data struct which contains the various parameters
(e.g. simulation time) is defined in terms of their type. The \textit{!} is used
to specific it is a strictness declaration.\change[inline]{I probably should have said this earlier but the reason for using strictness annotations is to avoid space leaks - I can explain this when we meet}

\begin{code}
data SimCtx g = SimCtx
  { simSf    :: !(SimSF g)
  , simEnv   :: !SIREnv
  , simRng   :: g
  , simSteps :: !Integer
  , simTime  :: !Time
  }
\end{code}

\subsection{Simulation Rules}

In this simulation, the rules are:

\begin{itemize}

\item There is a population of size $N$ in which all agents are either
Susceptible, Infected or Recovered at a particular time.

\item Initially, there is at least one infected person in the
population.

\item People interact with each other on average with a given rate of
|contactRate| per time-unit.

\item People become infected with a given probability |infectivity| when
interacting with an infected person.

\item When infected, a person recovers on average after |illnessDuration|
time-units.

\item An infected person is immune to further infections (no
reinfection).

\item The 2D environment has either Moore or von Neumann neighbourhood.

\item Agents are either static or can move freely around the cells.

\item The cells allow either single or multiple occupants.

\item Agents can read the states of all their neighbours which tells
them if a neighbour is infected or not.

\end{itemize}

\subsection{Agents}

\info[inline]{I know we have to have this here in Python notebook but I think it would be better to explain how the agents get updated first - maybe we can even do this in the notebook by defining things but not running them}

\subsubsection{Susceptible Agents}

|susceptibleAgent| describes the behaviour of a susceptible agent ---
which is governed by querying the surrounding neighbours and either
getting infected based on the parameter |infectivity| (event generated) or staying
susceptible (no event).

\improvement[inline]{Now that I have thought about it for a bit longer, maybe we should just leave the code as it is but with a footnote or aside to say that we should really use a Poisson distribution}

\change[inline]{I removed this comment: -- use occasionally to make contact on average}

We use |occasionally :: MonadRandom m => Time -> b -> SF m a (Event
b)| to determine whether an agent could have contacted another agent
within the average time period |1 / contactRate|. The constraint
|MonadRandom m| on the left hand side of |=>| can be ignored but we
note that it is now |MonadRandom| rather than |Monad| which further
restricts what |m| can be (|MonadRandom| is sub-class of
|Monad|). |occasionally| then takes a time and a value and returns a
time-varying process of type |SF m () Event ()|. So this process takes
no input and returns something that is either |NoEvent| or after some
(random) time |Event ()|. If the output is |NoEvent| then we return
that this agent is still |Susceptible|.

On the other hand if the agent did make contact then we select another
agent randomly from its neighbours: if this other agent is infected
then the original agent may become infected depending on a Bernoulli
random variabl.

Note that we are not using the Gillespie
algorithm~\cite{doi:10.1021/j100540a008} or, as is often used, a
Poisson distribution to determine the number of other agents that
would be infected. We allow this agent to be potentially infected
depending on its proximity to a possibly infected agent.

Functions related to using RNG elements

`drawRandomElemS` and `randomBoolM` are related to use of RNG elements
in simulation. A specific example is `RandomBoolM` being used to draw a
random Boolean to determine whether an infection occurs in the
susceptibleAgent function.

\begin{code}
randomBoolSF :: RandomGen g => SF (Rand g) () Bool
randomBoolSF = arrM (const (lift $ randomBoolM infectivity))
  where
    randomBoolM p = do r <- getRandomR (0, 1)
                       return $ r <= p

drawRandomElemS :: MonadRandom m => SF m [a] a
drawRandomElemS = proc as -> do
  r <- getRandomRS ((0, 1) :: (Double, Double)) -< ()
  let len = length as
  let idx = (fromIntegral len * r)
  let a =  as !! (floor idx)
  returnA -< a
\end{code}

\change[inline]{I removed these comments -- take env, the dimensions of grid and neighbourhood info -- let ns = neighbours env coord agentGridSize moore -- queries the environemtn for its neighbours - in this case appears to be all neighbours -- randomly selects one -- upon infection -- event returned which returns in switching into the infected agent SF (to behave as such)
}

\begin{code}
susceptible :: Disc2dCoord -> SF (Rand StdGen) SIREnv (SIRState, Event ())
susceptible coord = proc env -> do
  makeContact <- occasionally (1 / contactRate) () -< ()
  if not (isEvent makeContact)
    then returnA -< (Susceptible, NoEvent)
    else do
      let ns = neighbours env coord agentGridSize Nothing
      s <- drawRandomElemS -< ns
      case s of
        Infected -> do
          isInfected <- randomBoolSF -< ()
          if isInfected
            then returnA -< (Infected, Event ())
            else returnA -< (Susceptible, NoEvent)
        _ -> returnA -< (Susceptible, NoEvent)
\end{code}

We delay the switching by 1 step, otherwise the agent could make the
transition from Susceptible to Recovered in one time step, not
something we want to have in our model.

\begin{code}
susceptibleAgent :: Disc2dCoord -> SF (Rand StdGen) SIREnv SIRState
susceptibleAgent coord
    = switch
      (susceptible coord >>> iPre (Susceptible, NoEvent))
      (const infectedAgent)
\end{code}

\subsubsection{Infected Agents}

The function below describes the behaviour of an infected agent. This
behaviour is governed by either recovering on average after delta time
units or staying infected within a timestep.

\begin{code}
infected :: ABMSF SIREnv (SIRState, Event ())
infected = proc _ -> do
  recovered <- occasionally illnessDuration () -< ()
  if isEvent recovered
    then returnA -< (Recovered, Event ())
    else returnA -< (Infected, NoEvent)
\end{code}

Again we delay the switching by 1 step, otherwise could make the
transition from Susceptible to Recovered within time-step.

\begin{code}
type ABMSF a b = SF (Rand StdGen) a b

infectedAgent :: ABMSF SIREnv SIRState
infectedAgent
    = switch
      (infected >>> iPre (Infected, NoEvent))
      (const recoveredAgent)
\end{code}

\subsubsection{Recovered Agents}

Unlike the other states, the recovered state does not generate any event
and rather just acts a sink which constantly returns Recovered.

\begin{code}
recoveredAgent :: ABMSF SIREnv SIRState
recoveredAgent = arr (const Recovered)
\end{code}

\section{Arrowised FRP}

An Arrow is a generalisation of a monad which represents a process
that takes as input a value of type |input| and outputs a
value of type |output|.

\input{answerPiccies.tex}

\section{Whatever}

To enforce the simulation rules, various simulation parameters are
defined: the contact rate |contactRate|, the infection rate |infectivity|,
recovery rate |illnessDuration| and the grid size.

\begin{code}
contactRate :: Double
contactRate = 5.0

infectivity :: Double
infectivity = 0.10

illnessDuration :: Double
illnessDuration = 15.0

agentGridSize :: (Int, Int)
agentGridSize = (27, 28)
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
           -> Maybe [Disc2dCoord]
           -> [SIRState]
neighbours e _ _ Nothing = elems e
neighbours e (x, y) (dx, dy) (Just n) = map (e !) nCoords'
  where
    nCoords  = map (\(x', y') -> (x + x', y + y')) n
    nCoords' = filter (\(nx, ny) -> nx >= 0 &&
                                    ny >= 0 &&
                                    nx <= (dx - 1) &&
                                  ny <= (dy - 1)) nCoords
\end{code}

`sirAgent` defines the behaviour of the agent depending on the initial
state. Only the suspectible agent receives the coordinates as the
infected and recovered agents do not require this information.

 -- recovered agent ignores gen bc they stay immune

\begin{code}
sirAgent :: Disc2dCoord -> SIRState -> SF (Rand StdGen) SIREnv SIRState
sirAgent coord Susceptible = susceptibleAgent coord
sirAgent _     Infected    = infectedAgent
sirAgent _     Recovered   = recoveredAgent
\end{code}

The simulationStep function is a closed feedback loop which takes the
current signal functions and returns the new agent states.

|unMSF :: MSF m a b -> a -> m (b, MSF m a b)| executes one step of a
simulation, and produces an output in a monadic context, and a
continuation to be used for future steps.

We run all agents sequentially keeping the environment read-only; it
is shared as input with all agents and thus cannot be changed by the
agents themselves.

We then construct new environment from all agent outputs for next step
using |(\\)| which takes an array and a list of pairs and returns an
array identical to the left argument except that it has been updated
by the associations in the right argument.

\begin{code}
simulationStep :: RandomGen g
               => [(SIRAgent g, Disc2dCoord)]
               -> SIREnv
               -> SF (Rand g) () SIREnv
simulationStep sfsCoords env = MSF $ \_ -> do
  let (sfs, coords) = unzip sfsCoords
  ret <- mapM (`unMSF` env) sfs
  let (as, sfs') = unzip ret
      env' = foldr (\(coord, a) envAcc -> envAcc // [(coord, a)]) env (zip coords as)
      sfsCoords' = zip sfs' coords
      cont       = simulationStep sfsCoords' env'
  return (env', cont)
\end{code}

Functions related to collating simulation data

`aggregateStates` is used to collate the number of susceptible, infected
and recovered agents within the simulation. This function is used both
for the animation and the plot.

\begin{code}
aggregateStates :: [SIRState] -> (Int, Int, Int)
aggregateStates as = (susceptibleCount, infectedCount, recoveredCount)
  where
    susceptibleCount = length $ filter (Susceptible==) as
    infectedCount    = length $ filter (Infected==) as
    recoveredCount   = length $ filter (Recovered==) as
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
    -- (cx, cy)   = agentGridSize
    -- (wx, wy)   = winSize
    -- cellWidth  = (fromIntegral wx / fromIntegral cx) :: Double
    -- cellHeight = (fromIntegral wy / fromIntegral cy) :: Double

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
          timeStepTxt = GLO.color GLO.black $ GLO.translate tcx tcy $
                        GLO.scale 0.5 0.5 $ GLO.Text timeTxt

renderAgent :: (Disc2dCoord, SIRState) -> GLO.Picture
renderAgent (coord, Susceptible)
    = GLO.color (GLO.makeColor 0.0 0.0 0.7 1.0) $
      GLO.translate x y $ GLO.Circle (realToFrac cellWidth / 2)
  where
    (x, y) = transformToWindow coord
renderAgent (coord, Infected)
    = GLO.color (GLO.makeColor 0.7 0.0 0.0 1.0) $
      GLO.translate x y $
      GLO.ThickCircle 0 (realToFrac cellWidth)
  where
    (x, y) = transformToWindow coord
renderAgent (coord, Recovered)
    = GLO.color (GLO.makeColor 0.0 0.70 0.0 1.0) $
      GLO.translate x y $
      GLO.ThickCircle 0 (realToFrac cellWidth)
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
                   -> [(Int, Int, Int)]
-- With the max time, time step and initial context, run simulation via the Aux function
runSimulationUntil tMax dt ctx0 = runSimulationAux 0 ctx0 []
  where
    runSimulationAux :: RandomGen g
                      => Time
                      -> SimCtx g
                      -> [(Int, Int, Int)]
                      -> [(Int, Int, Int)]
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
main2 :: IO ()
main2 = do
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
    csvData <- BL.readFile "SIR_DUNAI_dt001.csv"
    case decode HasHeader csvData of
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

main1 :: IO ()
main1 = do
  (d,e,f) <- getResults
  withArgs ["-odiagrams/BoardingSchool78.png"] (r2AxisMain $ scatterAxis2 (d, e, f))
\end{code}

![svg](output\_82\_0.svg)

The GIF below shows the spread of infection overtime

![SegmentLocal](SIR.gif "segment")

\section{Bibliography}

\printbibliography

\end{document}
