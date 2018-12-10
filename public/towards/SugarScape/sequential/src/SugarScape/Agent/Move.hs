{-# LANGUAGE FlexibleContexts #-}
module SugarScape.Agent.Move 
  ( agentMove
  , handleKilledInCombat

  , selectBestSites   -- for testing purposes
  , sugarSiteMeasure  -- for testing purposes
  ) where

import Data.List
import Data.Maybe

import Control.Monad.Random

import SugarScape.Agent.Common
import SugarScape.Core.Common
import SugarScape.Core.Discrete
import SugarScape.Core.Model
import SugarScape.Core.Random
import SugarScape.Core.Scenario
import SugarScape.Core.Utils

agentMove :: RandomGen g => AgentLocalMonad g Double
agentMove =
  ifThenElseM
    (isNothing . spCombat <$> scenario)
    agentNonCombat
    agentCombat

agentNonCombat :: RandomGen g
               => AgentLocalMonad g Double
agentNonCombat = do
  sitesInSight <- agentLookout
  coord        <- agentProperty sugAgCoord

  let uoc = filter (siteUnoccupied . snd) sitesInSight

  ifThenElse 
    (null uoc)
    (agentHarvestSite coord)
    (do
        -- NOTE included self but this will be always kicked out because self is occupied by self, need to somehow add this
        --       what we want is that in case of same sugar on all fields (including self), the agent does not move because staying is the lowest distance (=0)
        selfCell <- envLift $ cellAtM coord
        myState  <- agentState
        sc       <- scenario

        let uoc' = (coord, selfCell) : uoc
            bf   = selectSiteMeasureFunc sc myState
            bcs  = selectBestSites bf coord uoc'

        (cellCoord, _) <- randLift $ randomElemM bcs
        agentMoveTo cellCoord
        agentHarvestSite cellCoord)

agentCombat :: RandomGen g => AgentLocalMonad g Double
agentCombat = do
    combatReward <- fromJust . spCombat <$> scenario

    myTribe  <- agentProperty sugAgTribe
    mySugLvl <- agentProperty sugAgSugarLevel
    mySpiLvl <- agentProperty sugAgSpiceLevel
    myVis    <- agentProperty sugAgVision

    let myWealth = mySugLvl + mySpiLvl

    -- lookout in 4 directions as far as vision perimts and only consider
    -- sites occuppied by members of different tribe who are less wealthier
    sitesInSight <- agentLookout
    let sites = filter (\(_, site) -> 
                          case sugEnvSiteOccupier site of 
                            Nothing  -> False
                            Just occ -> sugEnvOccTribe occ /= myTribe &&
                                        (sugEnvOccSugarWealth occ + sugEnvOccSpiceWealth occ) < myWealth) sitesInSight

    -- throw out all sites which are vulnerable to retalation:
    nonRetaliationSites <- filterRetaliation myTribe myWealth myVis combatReward sites []

    if null nonRetaliationSites
      then agentNonCombat -- if no sites left for combat, just do a non-combat move
      else do
        myCoord <- agentProperty sugAgCoord
        sc      <- scenario

        let bf   = combatSiteMeasure sc combatReward
            bcs  = selectBestSites bf myCoord nonRetaliationSites

        (siteCoord, site) <- randLift $ randomElemM bcs
        agentMoveTo siteCoord
        harvestAmount <- agentHarvestSite siteCoord

        let victim       = fromJust $ sugEnvSiteOccupier site
            victimWealth = sugEnvOccSugarWealth victim + sugEnvOccSpiceWealth victim
            combatWealth = min victimWealth combatReward

        let victimId = sugEnvOccId (fromJust $ sugEnvSiteOccupier site)
        sendEventTo victimId KilledInCombat

        return (harvestAmount + combatWealth)

  where
    filterRetaliation :: RandomGen g
                      => AgentTribe
                      -> Double
                      -> Int
                      -> Double
                      -> [(Discrete2dCoord, SugEnvSite)]
                      -> [(Discrete2dCoord, SugEnvSite)]
                      -> AgentLocalMonad g [(Discrete2dCoord, SugEnvSite)]
    filterRetaliation _ _ _ _ [] acc = return acc
    filterRetaliation myTribe myWealth myVis combatReward (site@(sc, ss) : sites) acc = do
      futureSites <- envLift $ neighboursInNeumannDistanceM sc myVis False

      let victim        = fromJust $ sugEnvSiteOccupier ss
          victimWealth  = sugEnvOccSugarWealth victim + sugEnvOccSpiceWealth victim
          combatWealth  = min victimWealth combatReward
          sugLvlSite    = sugEnvSiteSugarLevel ss
          futureWealth  = myWealth + combatWealth + sugLvlSite

          filteredSites = filter (\(_, ss') -> 
                          case sugEnvSiteOccupier ss' of 
                            Nothing  -> False
                            Just occ -> sugEnvOccTribe occ /= myTribe &&
                                        (sugEnvOccSugarWealth occ + sugEnvOccSpiceWealth occ) > futureWealth) futureSites

      -- in case sites found with agents more wealthy after this agents combat, then this site is vulnerable to retaliation, 
      if null filteredSites
        then filterRetaliation myTribe myWealth myVis combatReward sites (site : acc) -- add this site, cant be retaliated
        else filterRetaliation myTribe myWealth myVis combatReward sites acc -- ignore this site, its vulnerable to retaliation

handleKilledInCombat :: RandomGen g
                     => AgentId
                     -> AgentLocalMonad g ()
handleKilledInCombat _killerId = kill

agentLookout :: RandomGen g => AgentLocalMonad g [(Discrete2dCoord, SugEnvSite)]
agentLookout = do
  vis   <- agentProperty sugAgVision
  coord <- agentProperty sugAgCoord
  envLift $ neighboursInNeumannDistanceM coord vis False

agentMoveTo :: RandomGen g
            => Discrete2dCoord 
            -> AgentLocalMonad g ()
agentMoveTo cellCoord = do
  unoccupyPosition

  updateAgentState (\s -> s { sugAgCoord = cellCoord })

  occ  <- occupierLocal
  cell <- envLift $ cellAtM cellCoord
  let co = cell { sugEnvSiteOccupier = Just occ }
  envLift $ changeCellAtM cellCoord co 

agentHarvestSite :: RandomGen g
                 => Discrete2dCoord 
                 -> AgentLocalMonad g Double
agentHarvestSite siteCoord = do
  site   <- envLift $ cellAtM siteCoord
  sugLvl <- agentProperty sugAgSugarLevel
  
  let sugLvlSite = sugEnvSiteSugarLevel site

  harvestAmount <- 
    ifThenElseM
      (spSpiceEnabled <$> scenario)
      (do
        spiceLvl <- agentProperty sugAgSpiceLevel
        let spiceLvlSite = sugEnvSiteSpiceLevel site

        updateAgentState (\s -> s { sugAgSugarLevel = sugLvl + sugLvlSite
                                  , sugAgSpiceLevel = spiceLvl + spiceLvlSite})
        return (sugLvlSite + spiceLvlSite))
      (do
        updateAgentState (\s -> s { sugAgSugarLevel = sugLvl + sugLvlSite })
        return sugLvlSite)

  -- NOTE: need to update occupier-info in environment because wealth has (and MRS) changed
  occ <- occupierLocal

  let siteHarvested = site { sugEnvSiteSugarLevel = 0
                           , sugEnvSiteSpiceLevel = 0
                           , sugEnvSiteOccupier   = Just occ }
  envLift $ changeCellAtM siteCoord siteHarvested

  return harvestAmount

type SiteMeasureFunc = SugEnvSite -> Double

-- NOTE: includes polution unconditionally for better maintainability (lower number of functions and cases)
-- polution level will be 0 anyway if polution / diffusion is turned off
sugarSiteMeasure :: SiteMeasureFunc
sugarSiteMeasure site = sug / (1 + pol)
  where
    sug = sugEnvSiteSugarLevel site
    pol = sugEnvSitePolutionLevel site

selectBestSites :: SiteMeasureFunc
                -> Discrete2dCoord
                -> [(Discrete2dCoord, SugEnvSite)]
                -> [(Discrete2dCoord, SugEnvSite)]
selectBestSites measureFunc refCoord cs = bestShortestdistanceManhattanCells
  where
    cellsSortedByMeasure = sortBy (\c1 c2 -> compare (measureFunc $ snd c2) (measureFunc $ snd c1)) cs
    bestCellMeasure = measureFunc $ snd $ head cellsSortedByMeasure
    bestCells = filter ((==bestCellMeasure) . measureFunc . snd) cellsSortedByMeasure

    shortestdistanceManhattanBestCells = sortBy (\c1 c2 -> compare (distanceManhattanDisc2d refCoord (fst c1)) (distanceManhattanDisc2d refCoord (fst c2))) bestCells
    shortestdistanceManhattan = distanceManhattanDisc2d refCoord (fst $ head shortestdistanceManhattanBestCells)
    bestShortestdistanceManhattanCells = filter ((==shortestdistanceManhattan) . (distanceManhattanDisc2d refCoord) . fst) shortestdistanceManhattanBestCells
      
selectSiteMeasureFunc :: SugarScapeScenario -> SugAgentState -> SiteMeasureFunc
selectSiteMeasureFunc params as
  | spSpiceEnabled params = sugarSpiceSiteMeasure as
  | otherwise             = sugarSiteMeasure

-- NOTE: includes polution unconditionally for better maintainability (lower number of functions and cases)
-- polution level will be 0 anyway if polution / diffusion is turned off
combatSiteMeasure :: SugarScapeScenario -> Double -> SiteMeasureFunc
combatSiteMeasure _params combatReward site = combatWealth + sug + spi
  where
    victim       = fromJust $ sugEnvSiteOccupier site
    victimWealth = sugEnvOccSugarWealth victim + sugEnvOccSpiceWealth victim
    combatWealth = min victimWealth combatReward

    pol          = sugEnvSitePolutionLevel site
    sug          = sugEnvSiteSugarLevel site / (1 + pol)
    spi          = sugEnvSiteSpiceLevel site / (1 + pol)

-- See page 97, The Agent Welfare Function and Appendix C (Example makes it quite clear)
-- The agent welfare function itself computes whether the agent requires more 
-- sugar or more spice, depending on the respective metabolisms. 
-- Now we apply this welfare function to compute a measure for the site which means
-- we compute the potential welfare when the agent is on that site, thus we
-- add the sites sugar / spice to the respective parts of the equation.
-- NOTE: includes polution unconditionally for better maintainability (lower number of functions and cases)
-- polution level will be 0 anyway if polution / diffusion is turned off
sugarSpiceSiteMeasure :: SugAgentState -> SiteMeasureFunc
sugarSpiceSiteMeasure as site = agentWelfareChange sug spi w1 w2 m1 m2
  where
    m1 = fromIntegral $ sugAgSugarMetab as
    m2 = fromIntegral $ sugAgSpiceMetab as
    w1 = sugAgSugarLevel as
    w2 = sugAgSpiceLevel as

    pol = sugEnvSitePolutionLevel site
    sug = sugEnvSiteSugarLevel site / (1 + pol)
    spi = sugEnvSiteSpiceLevel site / (1 + pol)