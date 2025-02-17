-- {-# LANGUAGE FlexibleContexts #-}
module SugarScape.Agent.Culture 
  ( agentCultureProcess

  , handleCulturalProcess
  ) where

import Data.Maybe

import Control.Monad.Random

import SugarScape.Agent.Common
import SugarScape.Core.Common
import SugarScape.Core.Model
import SugarScape.Core.Scenario
import SugarScape.Core.Utils

agentCultureProcess :: RandomGen g => AgentLocalMonad g (SugAgentOut g)
agentCultureProcess = 
  ifThenElseM
    ((isNothing . spCulturalProcess) <$> scenario)
    agentObservableM
    (do
      nids <- neighbourAgentIds

      -- no neighbours, ignore cultural process
      if null nids
        then agentObservableM
        else do
          ao         <- agentObservableM
          cultureTag <- agentProperty sugAgCultureTag
          -- simply broadcast to all neighbours, they compute and flip their tags themselves
          broadcastEvent nids (CulturalProcess cultureTag)
          return ao)

handleCulturalProcess :: RandomGen g
                      => AgentId
                      -> CultureTag
                      -> AgentLocalMonad g (SugAgentOut g)
handleCulturalProcess _sender otherTag = do
  myTag <- agentProperty sugAgCultureTag
  
  -- NOTE: assuming length otherTag == length myTag
  idx <- randLift $ getRandomR (0, length myTag - 1)

  -- when disagree, the receiving agents tag is set to the 
  -- sending agents tag => just flip it because its a Bool
  when
    (myTag !! idx /= otherTag !! idx)
    (do
      let myTag' = flipBoolAtIdx idx myTag
      updateAgentState (\s -> s { sugAgCultureTag = myTag'
                                , sugAgTribe      = tagToTribe myTag' })
      -- NOTE: need to update occupier info because tribe might have changed
      updateSiteOccupied)

  agentObservableM