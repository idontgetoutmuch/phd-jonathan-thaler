{-# LANGUAGE FlexibleContexts #-}
module SugarScape.Agent.Common
  ( SugarScapeAgent
  , AgentLocalMonad
  , EventHandler

  , absStateLift
  , envLift
  , randLift
  , myId
  , scenario
  , agentState
  , updateAgentState
  , agentProperty
  , sendEvents
  , broadcastEvent
  , sendEventTo
  , kill
  , newAgent
  
  , neighbourAgentIds
  
  , agentWelfare
  , agentWelfareM
  , agentWelfareState
  , agentWelfareChange
  , agentWelfareChangeM
  , agentWelfareChangeState
  , mrs
  , mrsM
  , mrsState
  , mrsStateChange

  , occupier
  , occupierLocal
  , siteOccupier
  , siteUnoccupied
  , siteOccupied

  , unoccupyPosition
  , updateSiteOccupied
  , agentCellOnCoord
  
  , randomAgent

  , tagToTribe

  , changeToRedTribe
  , changeToBlueTribe
  , sugObservableFromState
  ) where

import Data.Maybe

import Control.Monad.State.Strict
import Control.Monad.Random
import Control.Monad.Reader
import Control.Monad.Writer.Strict
import Data.MonadicStreamFunction

import SugarScape.Agent.Interface
import SugarScape.Core.Common
import SugarScape.Core.Discrete
import SugarScape.Core.Model
import SugarScape.Core.Random
import SugarScape.Core.Scenario

type SugarScapeAgent g = SugarScapeScenario -> AgentId -> SugAgentState -> SugAgentMSF g

type AgentLocalMonad g = WriterT (SugAgentOut g) (ReaderT (SugarScapeScenario, AgentId) (StateT SugAgentState (SugAgentMonadT g)))
type EventHandler g    = MSF (AgentLocalMonad g) (ABSEvent SugEvent) ()

absStateLift :: (StateT ABSState (StateT SugEnvironment (Rand g))) a -> AgentLocalMonad g a
absStateLift = lift . lift . lift

envLift :: StateT SugEnvironment (Rand g) a -> AgentLocalMonad g a
envLift = lift . lift . lift . lift

randLift :: Rand g a -> AgentLocalMonad g a
randLift = lift . lift . lift . lift . lift

myId :: AgentLocalMonad g AgentId
myId = lift $ asks snd

scenario :: AgentLocalMonad g SugarScapeScenario
scenario = lift $ asks fst

updateAgentState :: (SugAgentState -> SugAgentState)
                 -> AgentLocalMonad g ()
updateAgentState = lift . lift . modify

agentProperty :: (SugAgentState -> p)
              -> AgentLocalMonad g p
agentProperty = lift . lift . gets

agentState :: AgentLocalMonad g SugAgentState
agentState = lift $ lift get

broadcastEvent :: [AgentId]
               -> SugEvent
               -> AgentLocalMonad g ()
broadcastEvent rs e = tell $ broadcastEventAo rs e

sendEvents :: [(AgentId, SugEvent)] -> AgentLocalMonad g ()
sendEvents es = tell $ sendEventsAo es

sendEventTo :: AgentId
            -> SugEvent
            -> AgentLocalMonad g ()
sendEventTo receiver e = tell $ sendEventToAo receiver e

kill :: AgentLocalMonad g ()
kill = tell killAo

newAgent :: SugAgentDef g -> AgentLocalMonad g ()
newAgent adef = tell $ newAgentAo adef 

neighbourAgentIds :: AgentLocalMonad g [AgentId]
neighbourAgentIds = do
    coord <- agentProperty sugAgCoord
    filterNeighbourIds <$> envLift (neighboursM coord False)
  where
    filterNeighbourIds :: [Discrete2dCell SugEnvSite] -> [AgentId]
    filterNeighbourIds ns = map (siteOccupier . snd) $ filter (siteOccupied . snd) ns

sugObservableFromState :: SugAgentState -> SugAgentObservable
sugObservableFromState as = SugAgentObservable
  { sugObsCoord      = sugAgCoord as 
  , sugObsVision     = sugAgVision as
  , sugObsAge        = sugAgAge as 
  , sugObsSugLvl     = sugAgSugarLevel as
  , sugObsSugMetab   = sugAgSugarMetab as
  , sugObsGender     = sugAgGender as
  , sugObsCultureTag = sugAgCultureTag as
  , sugObsTribe      = sugAgTribe as
  , sugObsSpiLvl     = sugAgSpiceLevel as
  , sugObsSpiMetab   = sugAgSpiceMetab as
  , sugObsTrades     = sugAgTrades as
  , sugObsDiseases   = sugAgDiseases as
  }

-- default welfare function, not incorporating change to sugar / spice
agentWelfare :: Double  -- ^ sugar-wealth of agent
             -> Double  -- ^ spice-wealth of agent
             -> Double  -- ^ sugar-metabolism of agent
             -> Double  -- ^ spice-metabolism of agent
             -> Double
agentWelfare = agentWelfareChange 0 0

agentWelfareState :: SugAgentState -> Double
agentWelfareState as = agentWelfareChangeState as 0 0

agentWelfareM :: MonadState SugAgentState m => m Double
agentWelfareM = agentWelfareChangeM 0 0

-- see page 102, internal valuations
mrs :: Double  -- ^ sugar-wealth of agent
    -> Double  -- ^ spice-wealth of agent
    -> Double  -- ^ sugar-metabolism of agent
    -> Double  -- ^ spice-metabolism of agent
    -> Double  -- ^ mrs value: less than 1 the agent values sugar more, and spice otherwise
mrs w1 w2 m1 m2 = (w2 / m2) / (w1 / m1)

mrsM :: AgentLocalMonad g Double
mrsM = do
  s <- agentState
  return $ mrsState s

mrsStateChange :: SugAgentState 
               -> Double
               -> Double
               -> Double
mrsStateChange as sugarChange spiceChange 
    = mrs (w1 + sugarChange) (w2 + spiceChange) m1 m2
  where
    m1 = fromIntegral $ sugAgSugarMetab as
    m2 = fromIntegral $ sugAgSpiceMetab as
    w1 = sugAgSugarLevel as
    w2 = sugAgSpiceLevel as

mrsState :: SugAgentState -> Double
mrsState as = mrsStateChange as 0 0

-- NOTE: this welfare function includes the ability to calculate the changed
-- welfare of an agent when sugar and spice change - is required for determining
-- the best site to move to when spice is enabled
agentWelfareChange :: Double  -- ^ sugar-change in welfare
                   -> Double  -- ^ spice-change in welfare
                   -> Double  -- ^ sugar-wealth of agent
                   -> Double  -- ^ spice-wealth of agent
                   -> Double  -- ^ sugar-metabolism of agent
                   -> Double  -- ^ spice-metabolism of agent
                   -> Double
agentWelfareChange sugarChange spiceChange w1 w2 m1 m2 
{-
    | isNaN wf = error ("invalid welfare change: w1 = " ++ show w1 ++ 
                        ", w2 = " ++ show w2 ++ 
                        ", m1 = " ++ show m1 ++ 
                        ", m2 = " ++ show m2 ++
                        ", sugarchange = " ++ show sugarChange ++ 
                        ", spiceChange = " ++ show spiceChange)
    | otherwise = wf
    -}
    = (w1Diff ** (m1/mT)) * (w2Diff ** (m2/mT))
  where
    mT = m1 + m2
    w1Diff = max (w1 + sugarChange) 0 -- prevent negative wealth, would result in NaN
    w2Diff = max (w2 + spiceChange) 0 -- prevent negative wealth, would result in NaN

agentWelfareChangeState :: SugAgentState
                        -> Double
                        -> Double
                        -> Double
agentWelfareChangeState as sugarChange spiceChange 
    = agentWelfareChange sugarChange spiceChange w1 w2 m1 m2
  where
    m1 = fromIntegral $ sugAgSugarMetab as
    m2 = fromIntegral $ sugAgSpiceMetab as
    w1 = sugAgSugarLevel as
    w2 = sugAgSpiceLevel as

agentWelfareChangeM :: MonadState SugAgentState m 
                    => Double
                    -> Double
                    -> m Double
agentWelfareChangeM sugarChange spiceChange 
  = state (\s -> (agentWelfareChangeState s sugarChange spiceChange, s))

occupier :: AgentId
         -> SugAgentState
         -> SugEnvSiteOccupier
occupier aid as = SugEnvSiteOccupier { 
    sugEnvOccId          = aid
  , sugEnvOccTribe       = sugAgTribe as
  , sugEnvOccSugarWealth = sugAgSugarLevel as
  , sugEnvOccSpiceWealth = sugAgSpiceLevel as
  , sugEnvOccMRS         = mrsState as
  }

occupierLocal :: AgentLocalMonad g SugEnvSiteOccupier
occupierLocal = do
  s   <- lift $ lift get 
  aid <- myId
  return $ occupier aid s

siteOccupier :: SugEnvSite -> AgentId
siteOccupier site = sugEnvOccId $ fromJust $ sugEnvSiteOccupier site

siteOccupied :: SugEnvSite -> Bool
siteOccupied = isJust . sugEnvSiteOccupier

siteUnoccupied :: SugEnvSite -> Bool
siteUnoccupied = not . siteOccupied

unoccupyPosition :: RandomGen g
                 => AgentLocalMonad g ()
unoccupyPosition = do
  (coord, cell) <- agentCellOnCoord
  let cell' = cell { sugEnvSiteOccupier = Nothing }
  envLift $ changeCellAtM coord cell'

updateSiteOccupied :: AgentLocalMonad g ()
updateSiteOccupied = do
  (coord, cell) <- agentCellOnCoord
  occ           <- occupierLocal
  let cell' = cell { sugEnvSiteOccupier = Just occ }
  envLift $ changeCellAtM coord cell'

agentCellOnCoord :: AgentLocalMonad g (Discrete2dCoord, SugEnvSite)
agentCellOnCoord = do
  coord <- agentProperty sugAgCoord
  cell  <- envLift $ cellAtM coord
  return (coord, cell)

randomAgent :: MonadRandom m
            => SugarScapeScenario
            -> (AgentId, Discrete2dCoord)
            -> SugarScapeAgent g
            -> (SugAgentState -> SugAgentState)
            -> m (SugAgentDef g, SugAgentState)
randomAgent sc (agentId, coord) asf f = do
  randSugarMetab     <- getRandomR $ spSugarMetabolismRange sc
  randVision         <- getRandomR $ spVisionRange sc
  randSugarEndowment <- getRandomR $ spSugarEndowmentRange sc
  ageSpan            <- randomAgentAge $ spAgeSpan sc
  randGender         <- randomGender $ spGenderRatio sc
  randFertAgeRange   <- randomFertilityRange sc randGender
  randCultureTag     <- randomCultureTag sc
  randSpiceEndowment <- getRandomR $ spSpiceEndowmentRange sc
  randSpiceMetab     <- getRandomR $ spSpiceMetabolismRange sc
  randImmuneSystem   <- randomImmuneSystem sc
  randDiseases       <- randomDiseases sc

  let initSugar = fromIntegral randSugarEndowment
      initSpice = fromIntegral randSpiceEndowment

  let s = SugAgentState {
    sugAgCoord        = coord
  , sugAgSugarMetab   = randSugarMetab
  , sugAgVision       = randVision
  , sugAgSugarLevel   = initSugar
  , sugAgMaxAge       = ageSpan
  , sugAgAge          = 0
  , sugAgGender       = randGender
  , sugAgFertAgeRange = randFertAgeRange
  , sugAgInitSugEndow = initSugar
  , sugAgChildren     = []
  , sugAgCultureTag   = randCultureTag
  , sugAgTribe        = tagToTribe randCultureTag
  , sugAgSpiceLevel   = initSpice
  , sugAgInitSpiEndow = initSpice
  , sugAgSpiceMetab   = randSpiceMetab
  , sugAgTrades       = []
  , sugAgBorrowed     = []
  , sugAgLent         = []
  , sugAgNetIncome    = 0
  , sugAgImmuneSystem = randImmuneSystem
  , sugAgImSysGeno    = randImmuneSystem
  , sugAgDiseases     = randDiseases
  }

  let s'   = f s
      adef = AgentDef {
    adId      = agentId
  , adSf      = asf sc agentId s'
  , adInitObs = sugObservableFromState s'
  }

  return (adef, s')

randomDiseases :: MonadRandom m
               => SugarScapeScenario
               -> m [Disease]
randomDiseases sc = 
  case spDiseasesEnabled sc of 
    Nothing -> return []
    Just (_, _, _, n, masterList) -> 
      randomElemsM n masterList

randomImmuneSystem :: MonadRandom m
                   => SugarScapeScenario
                   -> m ImmuneSystem
randomImmuneSystem sc = 
  case spDiseasesEnabled sc of 
    Nothing -> return []
    Just (n, _, _, _, _)  -> 
      take n <$> getRandoms

changeToRedTribe :: SugarScapeScenario
                 -> SugAgentState
                 -> SugAgentState
changeToRedTribe sc s = s { sugAgTribe      = tagToTribe redTag
                              , sugAgCultureTag = redTag }
  where             
    redTag = case spCulturalProcess sc of 
              Nothing -> replicate 10 True -- cultural process is deactivated => select default of 10 to generate different Red tribe
              Just n  -> replicate n True

changeToBlueTribe :: SugarScapeScenario
                  -> SugAgentState
                  -> SugAgentState
changeToBlueTribe sc s = s { sugAgTribe     = tagToTribe blueTag
                              , sugAgCultureTag = blueTag }
  where             
    blueTag = case spCulturalProcess sc of 
              Nothing -> replicate 10 False -- cultural process is deactivated => select default of 10 to generate different Red tribe
              Just n  -> replicate n False

tagToTribe :: CultureTag
           -> AgentTribe
tagToTribe tag 
    | zeros > ones = Blue
    | otherwise    = Red
  where
    zeros = length $ filter (==False) tag
    ones  = n - zeros  
    n     = length tag

randomCultureTag :: MonadRandom m
                 => SugarScapeScenario
                 -> m CultureTag
randomCultureTag sc = 
  case spCulturalProcess sc of 
    Nothing -> return []
    Just n  -> 
      take n <$> getRandoms

randomGender :: MonadRandom m
             => Double
             -> m AgentGender
randomGender p = do
  r <- getRandom
  if r >= p
    then return Male
    else return Female

randomFertilityRange :: MonadRandom m
                     => SugarScapeScenario 
                     -> AgentGender
                     -> m (Int, Int)
randomFertilityRange sc Male = do
  from <- getRandomR $ spFertStartRangeMale sc
  to   <- getRandomR $ spFertEndRangeMale sc
  return (from, to)
randomFertilityRange sc Female = do
  from <- getRandomR $ spFertStartRangeFemale sc
  to   <- getRandomR $ spFertEndRangeFemale sc
  return (from, to)

randomAgentAge :: MonadRandom m
               => AgentAgeSpan 
               -> m (Maybe Int)
randomAgentAge Forever         = return Nothing
randomAgentAge (Range from to) = do
  randMaxAge <- getRandomR (from, to)
  return $ Just randMaxAge