{-# LANGUAGE Arrows #-}
module ABSData
  (
    AgentId
  , DataFlow
  , AgentIn (..)
  , AgentOut (..)
  , SIRMsg (..)
  , SIRAgentIn
  , SIRAgentOut
  , SIRDataFlow
  , SIRAgent
  
  , runABS

  , susceptibleAgent
  , infectedAgent

  , agentIn
  , agentOut
  ) where

import System.IO

import Control.Monad.Random
import qualified Data.Map as Map
import FRP.Yampa

import SIR

type AgentId     = Int
type DataFlow d = (AgentId, d)

data AgentIn d = AgentIn
  { aiId    :: !AgentId
  , aiData  :: ![DataFlow d]
  } deriving (Show)

data AgentOut o d = AgentOut
  { aoData        :: ![DataFlow d]
  , aoObservable  :: !o
  } deriving (Show)

type Agent o d    = SF (AgentIn d) (AgentOut o d)

data SIRMsg       = Contact SIRState deriving (Show, Eq)
type SIRAgentIn   = AgentIn SIRMsg
type SIRAgentOut  = AgentOut SIRState SIRMsg
type SIRAgent     = Agent SIRState SIRMsg
type SIRDataFlow  = DataFlow SIRMsg

runABS :: RandomGen g 
       => g 
       -> Int
       -> Int
       -> Double
       -> Double
       -> Double 
       -> Time 
       -> DTime 
       -> [(Double, Double, Double)]
runABS g populationSize infectedCount contactRate infectivity illnessDuration t dt
    = aggregateAllStates $ runSimulation g contactRate infectivity illnessDuration t dt as
  where
    as = initAgents populationSize infectedCount

runSimulation :: RandomGen g 
              => g 
              -> Double
              -> Double
              -> Double 
              -> Time 
              -> DTime 
              -> [(AgentId, SIRState)] 
              -> [[SIRState]]
runSimulation g contactRate infectivity illnessDuration t dt as 
    = map (map aoObservable) aoss
  where
    steps = floor $ t / dt
    dts = replicate steps (dt, Nothing) -- keep input the same as initial one, will be ignored anyway
    n = length as

    (rngs, _) = rngSplits g n []
    ais = map fst as
    sfs = map (\(g', (_, s)) -> sirAgent g' ais s) (zip rngs as)
    ains = map (\(aid, _) -> agentIn aid) as

    aoss = embed (stepSimulation sfs ains) ((), dts)

    rngSplits :: RandomGen g => g -> Int -> [g] -> ([g], g)
    rngSplits g 0 acc = (acc, g)
    rngSplits g n acc = rngSplits g'' (n-1) (g' : acc)
      where
        (g', g'') = split g

    stepSimulation :: [SIRAgent] -> [SIRAgentIn] -> SF () [SIRAgentOut]
    stepSimulation sfs ains =
        dpSwitch
          (\_ sfs' -> (zip ains sfs'))
          sfs
          (switchingEvt >>> notYet) -- at time = 0, if we switch immediately we end up in endless switching, so always wait for 'next'
          stepSimulation

      where
        switchingEvt :: SF ((), [SIRAgentOut]) (Event [SIRAgentIn])
        switchingEvt = proc (_, aos) -> do
          let ais      = map aiId ains
              aios     = zip ais aos
              nextAins = distributeData aios
          returnA -< Event nextAins

    sirAgent :: RandomGen g 
            => g 
            -> [AgentId] 
            -> SIRState 
            -> SIRAgent
    sirAgent g ais Susceptible = susceptibleAgent g contactRate infectivity illnessDuration ais
    sirAgent g _   Infected    = infectedAgent g illnessDuration
    sirAgent _ _   Recovered   = recoveredAgent

susceptibleAgent :: RandomGen g 
                 => g
                 -> Double
                 -> Double
                 -> Double
                 -> [AgentId]
                 -> SIRAgent
susceptibleAgent g contactRate infectivity illnessDuration ais = 
    switch 
      (susceptible g) 
      (const $ infectedAgent g illnessDuration)
  where
    susceptible :: RandomGen g 
                  => g 
                  -> SF SIRAgentIn (SIRAgentOut, Event ())
    susceptible g0 = proc ain -> do
      -- todo: use feedback
      rec
        g <- iPre g0 -< g'
        let (infected, g') = runRand (gotInfected infectivity ain) g

      if infected 
        then returnA -< (agentOut Infected, Event ())
        else (do
          makeContact <- occasionally g (1 / contactRate) ()  -< ()
          contactId   <- drawRandomElemSF g                   -< ais
          let ao = agentOut Susceptible
          if isEvent makeContact
            then returnA -< (dataFlow (contactId, Contact Susceptible) ao, NoEvent)
            else returnA -< (ao, NoEvent))

infectedAgent :: RandomGen g 
              => g 
              -> Double
              -> SIRAgent
infectedAgent g illnessDuration = 
    switch
    infected 
      (const recoveredAgent)
  where
    infected :: SF SIRAgentIn (SIRAgentOut, Event ())
    infected = proc ain -> do
      recEvt <- occasionally g illnessDuration () -< ()
      let a = event Infected (const Recovered) recEvt
      -- note that at the moment of recovery the agent can still infect others
      -- because it will still reply with Infected
      let ao = respondToContactWith Infected ain (agentOut a)
      returnA -< (ao, recEvt)

recoveredAgent :: SIRAgent
recoveredAgent = arr (const $ agentOut Recovered)

drawRandomElemSF :: (RandomGen g, Show a) => g -> SF [a] a
drawRandomElemSF g = proc as -> do
  r <- noiseR ((0, 1) :: (Double, Double)) g -< ()
  let len = length as
  let idx = fromIntegral len * r
  let a =  as !! floor idx
  returnA -< a

initAgents :: Int -> Int -> [(AgentId, SIRState)]
initAgents n i = sus ++ inf
  where
    sus = map (\ai -> (ai, Susceptible)) [0..n-i-1]
    inf = map (\ai -> (ai, Infected)) [n-i..n-1]

dataFlow :: DataFlow d -> AgentOut o d -> AgentOut o d
dataFlow df ao = ao { aoData = df : aoData ao }

onDataM :: (Monad m) 
        => (acc -> DataFlow d -> m acc) 
        -> AgentIn d 
        -> acc 
        -> m acc
onDataM dHdl ai acc = foldM dHdl acc ds
  where
    ds = aiData ai

onData :: (DataFlow d -> acc -> acc) -> AgentIn d -> acc -> acc
onData df ai a = foldr df a (aiData ai)

gotInfected :: RandomGen g => Double -> SIRAgentIn -> Rand g Bool
gotInfected infectionProb ain = onDataM gotInfectedAux ain False
  where
    gotInfectedAux :: RandomGen g => Bool -> DataFlow SIRMsg -> Rand g Bool
    gotInfectedAux False (_, Contact Infected) = randomBoolM infectionProb
    gotInfectedAux x _ = return x

respondToContactWith :: SIRState -> SIRAgentIn -> SIRAgentOut -> SIRAgentOut
respondToContactWith state = onData respondToContactWithAux
  where
    respondToContactWithAux :: DataFlow SIRMsg -> SIRAgentOut -> SIRAgentOut
    respondToContactWithAux (senderId, Contact _) = dataFlow (senderId, Contact state)

distributeData :: [(AgentId, AgentOut o d)] -> [AgentIn d]
distributeData aouts = map (distributeDataAux allMsgs) ains -- NOTE: speedup by running in parallel (if +RTS -Nx)
  where
    allMsgs = collectAllData aouts
    ains = map (\(ai, _) -> agentIn ai) aouts 

    distributeDataAux :: Map.Map AgentId [DataFlow d]
                      -> AgentIn d
                      -> AgentIn d
    distributeDataAux allMsgs ain = ain'
      where
        receiverId = aiId ain
        msgs = aiData ain -- NOTE: ain may have already messages, they would be overridden if not incorporating them

        mayReceiverMsgs = Map.lookup receiverId allMsgs
        msgsEvt = maybe msgs (++ msgs) mayReceiverMsgs

        ain' = ain { aiData = msgsEvt }

    collectAllData :: [(AgentId, AgentOut o d)] -> Map.Map AgentId [DataFlow d]
    collectAllData = foldr collectAllDataAux Map.empty
      where
        collectAllDataAux :: (AgentId, AgentOut o d)
                              -> Map.Map AgentId [DataFlow d]
                              -> Map.Map AgentId [DataFlow d]
        collectAllDataAux (senderId, ao) accMsgs 
            | not $ null msgs = foldr collectAllDataAuxAux accMsgs msgs
            | otherwise = accMsgs
          where
            msgs = aoData ao

            collectAllDataAuxAux :: DataFlow d
                                 -> Map.Map AgentId [DataFlow d]
                                 -> Map.Map AgentId [DataFlow d]
            collectAllDataAuxAux (receiverId, m) accMsgs = accMsgs'
              where
                msg = (senderId, m)
                mayReceiverMsgs = Map.lookup receiverId accMsgs
                newMsgs = maybe [msg] (\receiverMsgs -> msg : receiverMsgs) mayReceiverMsgs

                -- NOTE: force evaluation of messages, will reduce memory-overhead EXTREMELY
                accMsgs' = seq newMsgs (Map.insert receiverId newMsgs accMsgs)

agentIn :: AgentId -> AgentIn d
agentIn aid = AgentIn {
    aiId    = aid
  , aiData  = []
  }

agentOut :: o -> AgentOut o d
agentOut o = AgentOut {
    aoData        = []
  , aoObservable  = o
  }

randomBoolM :: RandomGen g => Double -> Rand g Bool
randomBoolM p = getRandomR (0, 1) >>= (\r -> return $ r <= p)