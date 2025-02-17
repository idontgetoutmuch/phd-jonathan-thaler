module SugarScape.Model 
  ( SugAgentState (..)
  , SugAgentObservable (..)

  , SugEnvCell (..)

  , SugAgentMonad
  , SugAgentMonadT
  
  , SugEnvironment

  , SugAgent
  , SugAgentDef
  , SugAgentOut

  , AgentAgeSpan (..)

  , SugarScapeParams (..)
  , AgentDistribution (..)
  , SugarRegrow (..)
  , PolutionFormation (..)

  , mkSugarScapeParams

  , mkParamsAnimationII_1
  , mkParamsAnimationII_2
  , mkParamsAnimationII_3
  , mkParamsAnimationII_4
  , mkParamsAnimationII_6
  , mkParamsAnimationII_7
  , mkParamsAnimationII_8
  
  , mkParamsTerracing
  , mkParamsCarryingCapacity
  , mkParamsWealthDistr

  , maxSugarCapacityCell
  , sugarscapeDimensions
  , sugarEnvSpec
  ) where

import Control.Monad.Random

import SugarScape.AgentMonad
import SugarScape.Discrete

------------------------------------------------------------------------------------------------------------------------
-- AGENT-DEFINITIONS
------------------------------------------------------------------------------------------------------------------------

data SugAgentState = SugAgentState 
  { sugAgCoord            :: Discrete2dCoord
  , sugAgSugarMetab       :: Int               -- integer because discrete, otherwise no exact replication possible
  , sugAgVision           :: Int
  , sugAgSugarLevel       :: Double            -- floating point because regrow-rate can be set to floating point values
  , sugAgAge              :: Int
  , sugAgMaxAge           :: Maybe Int
  } deriving (Show, Eq)

data SugAgentObservable = SugAgentObservable
  { sugObsCoord    :: Discrete2dCoord
  , sugObsVision   :: Int
  , sugObsAge      :: Int
  , sugObsSugLvl   :: Double
  , sugObsSugMetab :: Int
  } deriving (Show, Eq)

data SugEnvCell = SugEnvCell 
  { sugEnvCellSugarCapacity :: Double
  , sugEnvCellSugarLevel    :: Double
  , sugEnvCellPolutionLevel :: Double
  , sugEnvCellOccupier      :: Maybe AgentId
  } deriving (Show, Eq)

type SugEnvironment = Discrete2d SugEnvCell

type SugAgentMonad g = Rand g
type SugAgentMonadT g = AgentT (Rand g)

type SugAgent g     = Agent    (SugAgentMonad g) SugAgentObservable SugEnvironment
type SugAgentDef g  = AgentDef (SugAgentMonad g) SugAgentObservable SugEnvironment
type SugAgentOut g  = AgentOut (SugAgentMonad g) SugAgentObservable SugEnvironment
------------------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------------------------------------------------------
-- SUGARSCAPE PARAMETERS
------------------------------------------------------------------------------------------------------------------------
maxSugarCapacityCell :: Int
maxSugarCapacityCell = 4

-- the sugarscape is 51x51 in our implementation
sugarscapeDimensions :: Discrete2dCoord
sugarscapeDimensions = (51, 51)

-- taken from Iain Weaver Sugarscape implementation
-- https://www2.le.ac.uk/departments/interdisciplinary-science/research/replicating-sugarscape
-- http://ccl.northwestern.edu/netlogo/models/community/
sugarEnvSpec :: [String]
sugarEnvSpec =
  [ "111111111111111111111111111112222222222111111111111"
  , "111111111111111111111111111222222222222222111111111"
  , "111111111111111111111111112222222222222222221111111"
  , "111111111111111111111111122222222222222222222211111"
  , "111111111111111111111111222222222222222222222221111"
  , "111110000000111111111111222222222223332222222222111"
  , "111110000000001111111111222222223333333332222222111"
  , "111110000000000111111112222222333333333333222222211"
  , "111110000000000111111112222223333333333333322222211"
  , "111110000000000011111112222223333333333333332222221"
  , "111110000000000011111122222233333344444333333222221"
  , "111110000000000111111122222233333444444433333222221"
  , "111111000000000111111122222333334444444443333222222"
  , "111111000000001111111222222333334444444443333322222"
  , "111111100000011111111222222333334444444443333322222"
  , "111111111001111111111222222333334444444443333322222"
  , "111111111111111111111222222333334444444443333222222"
  , "111111111111111111112222222333333444444433333222222"
  , "111111111111111111112222222233333344444333333222222"
  , "111111111111111111122222222233333333333333332222222"
  , "111111111111111112222222222223333333333333332222222"
  , "111111111111122222222222222223333333333333322222222"
  , "111111111122222222222222222222233333333332222222221"
  , "111111122222222222222222222222222333333222222222221"
  , "111122222222222222222222222222222222222222222222211"
  , "111222222222222222222222222222222222222222222222111"
  , "112222222222222222222222222222222222222222222221111"
  , "122222222222333333222222222222222222222222221111111"
  , "122222222233333333332222222222222222222221111111111"
  , "222222223333333333333322222222222222221111111111111"
  , "222222233333333333333322222222222211111111111111111"
  , "222222233333333333333332222222221111111111111111111"
  , "222222333333444443333332222222211111111111111111111"
  , "222222333334444444333333222222211111111111111111111"
  , "222222333344444444433333222222111111111111111111111"
  , "222223333344444444433333222222111111111100111111111"
  , "222223333344444444433333222222111111110000001111111"
  , "222223333344444444433333222222111111100000000111111"
  , "222222333344444444433333222221111111000000000111111"
  , "122222333334444444333332222221111111000000000011111"
  , "122222333333444443333332222221111110000000000011111"
  , "122222233333333333333322222211111110000000000011111"
  , "112222223333333333333322222211111111000000000011111"
  , "112222222333333333333222222211111111000000000011111"
  , "111222222233333333322222222111111111100000000011111"
  , "111222222222233322222222222111111111111000000011111"
  , "111122222222222222222222222111111111111111111111111"
  , "111112222222222222222222221111111111111111111111111"
  , "111111122222222222222222211111111111111111111111111"
  , "111111111222222222222222111111111111111111111111111"
  , "111111111111222222222211111111111111111111111111111"
  ]

data AgentAgeSpan      = Forever 
                       | Range Int Int deriving (Show, Eq)
data AgentDistribution = Scatter 
                       | Corner Discrete2dCoord deriving (Show, Eq)
data SugarRegrow       = Immediate 
                       | Rate Double 
                       | Season Double Double Int deriving (Show, Eq)
data PolutionFormation = NoPolution 
                       | Polute Double Double deriving (Show, Eq)

data SugarScapeParams = SugarScapeParams 
  { sgAgentCount           :: Int
  , sgAgentDistribution    :: AgentDistribution
  , spSugarRegrow          :: SugarRegrow    -- negative value means G_inf: regrow to max in next step, floating point to allow grow-back of less than 1
  , spSugarEndowmentRange  :: (Int, Int)
  , spSugarMetabolismRange :: (Int, Int)
  , spVisionRange          :: (Int, Int)
  , spReplaceAgents        :: Bool           -- replacement rule R_[a, b] on/off
  , spAgeSpan              :: AgentAgeSpan
  , spPolutionFormation    :: PolutionFormation
  , spPolutionDiffusion    :: Maybe Int
  }

mkSugarScapeParams :: SugarScapeParams
mkSugarScapeParams = SugarScapeParams {
    sgAgentCount           = 0
  , spSugarRegrow          = Immediate
  , sgAgentDistribution    = Scatter
  , spSugarEndowmentRange  = (0, 0)
  , spSugarMetabolismRange = (0, 0)
  , spVisionRange          = (0, 0)
  , spReplaceAgents        = False
  , spAgeSpan              = Forever
  , spPolutionFormation    = NoPolution
  , spPolutionDiffusion    = Nothing
  }

------------------------------------------------------------------------------------------------------------------------
-- CHAPTER II: Life And Death On The Sugarscape
------------------------------------------------------------------------------------------------------------------------
-- Social Evolution with immediate regrow, page 27
mkParamsAnimationII_1 :: SugarScapeParams 
mkParamsAnimationII_1 = SugarScapeParams {
    sgAgentCount           = 400     -- page 28
  , sgAgentDistribution    = Scatter
  , spSugarRegrow          = Immediate -- regrow to max immediately
  , spSugarEndowmentRange  = (5, 25) -- NOTE: this is specified in book page 33 where the initial endowments are set to 5-25
  , spSugarMetabolismRange = (1, 4)  -- NOTE: specified where? 1 - 4
  , spVisionRange          = (1, 6)  -- NOTE: set to 1-6 on page 24
  , spReplaceAgents        = False   -- no replacing of died agents
  , spAgeSpan              = Forever  -- agents dont die of age in this case
  , spPolutionFormation    = NoPolution
  , spPolutionDiffusion    = Nothing  
  }
-- terracing phenomenon as described on page 28
mkParamsTerracing :: SugarScapeParams 
mkParamsTerracing = mkParamsAnimationII_1

-- Social Evolution with regrow rate of 1, page 29
mkParamsAnimationII_2 :: SugarScapeParams
mkParamsAnimationII_2 = SugarScapeParams {
    sgAgentCount           = 400     -- page 28
  , sgAgentDistribution    = Scatter
  , spSugarRegrow          = Rate 1       -- regrow by 1 unit per step
  , spSugarEndowmentRange  = (5, 25) -- NOTE: this is specified in book page 33 where the initial endowments are set to 5-25
  , spSugarMetabolismRange = (1, 4)
  , spVisionRange          = (1, 6)
  , spReplaceAgents        = False        -- no replacing of died agents
  , spAgeSpan              = Forever  -- agents dont die of age in this case
  , spPolutionFormation    = NoPolution
  , spPolutionDiffusion    = Nothing  
  }
-- carrying capacity property as described on page 30
mkParamsCarryingCapacity :: SugarScapeParams
mkParamsCarryingCapacity = mkParamsAnimationII_2

-- Wealth Distribution page 34
mkParamsAnimationII_3 :: SugarScapeParams
mkParamsAnimationII_3 = SugarScapeParams {
    sgAgentCount           = 250        -- page 33
  , sgAgentDistribution    = Scatter
  , spSugarRegrow          = Rate 1          -- page 33
  , spSugarEndowmentRange  = (5, 25)    -- page 33
  , spSugarMetabolismRange = (1, 4)
  , spVisionRange          = (1, 6)
  , spReplaceAgents        = True       -- page 33
  , spAgeSpan              = Range 60 100  -- page 33
  , spPolutionFormation    = NoPolution
  , spPolutionDiffusion    = Nothing  
  }
-- wealth distribution as described on page 32-37
mkParamsAnimationII_4 :: SugarScapeParams
mkParamsAnimationII_4 = mkParamsAnimationII_3 -- same as G_1, M, R_60,100 => same as Animiation II-3
-- wealth distribution as described on page 32-37
mkParamsWealthDistr :: SugarScapeParams
mkParamsWealthDistr = mkParamsAnimationII_3 -- same as G_1, M, R_60,100 => same as Animiation II-3

-- Migration as described on page 42 and 43 in Animation II-6
mkParamsAnimationII_6 :: SugarScapeParams
mkParamsAnimationII_6 = SugarScapeParams {
    sgAgentCount           = 300              -- 300 otherwise no waves, see https://www2.le.ac.uk/departments/interdisciplinary-science/research/replicating-sugarscape
  , sgAgentDistribution    = Corner (20, 20)
  , spSugarRegrow          = Rate 0.5              -- 0.5 otherwise no waves, see https://www2.le.ac.uk/departments/interdisciplinary-science/research/replicating-sugarscape
  , spSugarEndowmentRange  = (5, 25)
  , spSugarMetabolismRange = (1, 4)
  , spVisionRange          = (1, 10)          -- increase vision to 10, see page 42, we suggest to to 15 to make the waves really prominent
  , spReplaceAgents        = False            -- agents in migration experiment are not replaced
  , spAgeSpan              = Forever      -- agents in Migration experiment do not die of age
  , spPolutionFormation    = NoPolution
  , spPolutionDiffusion    = Nothing  
  }

-- Seasonal Migration as described on page 44 and 45 in Animation II-7
mkParamsAnimationII_7 :: SugarScapeParams
mkParamsAnimationII_7 = SugarScapeParams {
    sgAgentCount           = 400              
  , sgAgentDistribution    = Scatter
  , spSugarRegrow          = Season 1 8 50             
  , spSugarEndowmentRange  = (5, 25)
  , spSugarMetabolismRange = (1, 4)
  , spVisionRange          = (1, 6)       
  , spReplaceAgents        = False
  , spAgeSpan              = Forever
  , spPolutionFormation    = NoPolution
  , spPolutionDiffusion    = Nothing  
  }

-- Polution as described on page 45 to 50 in Animation II-8
mkParamsAnimationII_8 :: SugarScapeParams
mkParamsAnimationII_8 = SugarScapeParams {
    sgAgentCount           = 400
  , sgAgentDistribution    = Scatter
  , spSugarRegrow          = Rate 1   
  , spSugarEndowmentRange  = (5, 25)
  , spSugarMetabolismRange = (1, 4)
  , spVisionRange          = (1, 6)       
  , spReplaceAgents        = False
  , spAgeSpan              = Forever
  , spPolutionFormation    = Polute 1 1
  , spPolutionDiffusion    = Just 1
  }
------------------------------------------------------------------------------------------------------------------------