{-# LANGUAGE Arrows           #-}
{-# LANGUAGE FlexibleContexts #-}
module SugarScape.Agent.Loan
  ( agentLoan

  , handleLoanOffer
  , handleLoanPayback
  , handleLoanLenderDied
  , handleLoanInherit

  , splitLoanBorrowed
  , splitLoanLent
  ) where

import Data.List
import Data.Maybe

import Control.Monad.Random
import Data.MonadicStreamFunction

import SugarScape.Agent.Common
import SugarScape.Agent.Utils
import SugarScape.Core.Common
import SugarScape.Core.Discrete
import SugarScape.Core.Model
import SugarScape.Core.Random
import SugarScape.Core.Scenario
import SugarScape.Core.Utils

import Debug.Trace as DBG

agentLoan :: RandomGen g
          => AgentLocalMonad g (Maybe (EventHandler g))
          -> AgentLocalMonad g (Maybe (EventHandler g))
agentLoan cont =
  ifThenElseM
    (isNothing . spLoansEnabled <$> scenario)
    cont
    (do
      checkBorrowedLoans
      offerLending cont)

checkBorrowedLoans :: RandomGen g => AgentLocalMonad g ()
checkBorrowedLoans = do
    cs <- agentProperty sugAgBorrowed

    ret <- mapM checkLoan cs
    let (mcs, sugDebts, spiDebts) = unzip3 ret
        cs'        = catMaybes mcs
        sugDebtSum = sum sugDebts
        spiDebtSum = sum spiDebts

    -- reduce the net-income by the debts payed back from Loans
    updateAgentState (\s -> s { sugAgBorrowed    = cs'
                              , sugAgNetIncome  = sugAgNetIncome s  - (sugDebtSum + spiDebtSum)})
    -- NOTE: need to update occupier-info in environment because wealth (and MRS) might have changed
    updateSiteOccupied
  where
    checkLoan :: RandomGen g
              => Loan
              -> AgentLocalMonad g (Maybe Loan, Double, Double) 
    checkLoan borrowerLoan@(Loan dueDate lender sugarFace spiceFace) = do
      t  <- absStateLift getSimTime

      if dueDate /= t
        then return (Just borrowerLoan, 0, 0) -- Loan not yet due
        else do
          rate <- ((/100) . snd . fromJust . spLoansEnabled) <$> scenario 

          let sugPay = sugarFace + (sugarFace * rate)  -- payback the original face-value + a given percentage (interest)
              spiPay = spiceFace + (spiceFace * rate)  -- payback the original face-value + a given percentage (interest)

          sugLvl <- agentProperty sugAgSugarLevel
          spiLvl <- agentProperty sugAgSpiceLevel
          
          if sugLvl >= sugPay && spiLvl >= spiPay
            then do -- own enough wealth to pay back the Loan fully
              -- NOTE: need to adjust wealth already here otherwise could pay more back than this agent has
              updateAgentState (\s -> s { sugAgSugarLevel = sugAgSugarLevel s - sugPay
                                        , sugAgSpiceLevel = sugAgSpiceLevel s - spiPay})
              -- NOTE: need to update occupier-info in environment because wealth (and MRS) might have changed
              updateSiteOccupied

              sendEventTo lender (LoanPayback borrowerLoan sugPay spiPay)

              aid <- myId
              DBG.trace ("Agent " ++ show aid ++ ": " ++ show borrowerLoan ++ " is now due at t = " ++ show t ++ ", pay FULLY back to lender " ++ show lender)
                return (Nothing, sugPay, spiPay)
            else do -- not enough wealth, just pay back half of wealth and issue new Loan for the remaining face value(s)
              dueDate' <- ((t+) . fst . fromJust . spLoansEnabled) <$> scenario
                  -- prevent negative values in facevalues: if agent has enough sugar/spice but not the other 
                  -- then it could be the case that it pays back one part of the Loan fully
              let sugPay' = min sugarFace sugLvl / 2
                  spiPay' = min spiceFace spiLvl / 2  
                  c'      = Loan dueDate' lender (sugarFace - sugPay') (spiceFace - spiPay')
              
              sendEventTo lender (LoanPayback borrowerLoan sugPay' spiPay')

              -- NOTE: need to adjust wealth already here otherwise could pay more back than this agent has
              updateAgentState (\s -> s { sugAgSugarLevel = sugAgSugarLevel s - sugPay'
                                        , sugAgSpiceLevel = sugAgSpiceLevel s - spiPay'})
              -- NOTE: need to update occupier-info in environment because wealth (and MRS) might have changed
              updateSiteOccupied

              aid <- myId
              DBG.trace ("Agent " ++ show aid ++ ": " ++ show borrowerLoan ++ " is now due at t = " ++ show t ++ ", pays PARTIALLY back with half of its wealth " ++ show (sugPay', spiPay') ++ " to lender " ++ show lender) 
                return (Just c', sugPay', spiPay')

offerLending :: RandomGen g
             => AgentLocalMonad g (Maybe (EventHandler g))
             -> AgentLocalMonad g (Maybe (EventHandler g))
offerLending cont = do
  pl <- potentialLender
  if isNothing pl 
    then cont
    else do
      myCoord <- agentProperty sugAgCoord
      sites   <- envLift $ neighboursM myCoord False
      let ns = mapMaybe (sugEnvSiteOccupier . snd) sites

      if null ns
        then cont
        else do
          t       <- absStateLift getSimTime
          dueDate <- ((t+) . fst . fromJust . spLoansEnabled) <$> scenario

          ns' <- randLift $ randomShuffleM ns
          lendTo cont dueDate ns'

lendTo :: RandomGen g
       => AgentLocalMonad g (Maybe (EventHandler g))
       -> Time
       -> [SugEnvSiteOccupier]
       -> AgentLocalMonad g (Maybe (EventHandler g))
lendTo cont _ []  = cont       -- iterated through all neighbours, finished, quit lendingTo and switch back to globalHandler
lendTo cont dueDate (neighbour : ns) = do 
  pl <- potentialLender -- always check because could have changed while iterating
  case pl of 
    Nothing -> cont -- not potential lender, quit
    Just (sug, spi) -> do
      aid <- myId

      let borrowerId   = sugEnvOccId neighbour
          borrowerLoan = Loan dueDate aid sug spi
          evtHandler   = lendingToHandler cont dueDate ns borrowerLoan

      sendEventTo borrowerId (LoanOffer borrowerLoan)
      return $ Just evtHandler

lendingToHandler :: RandomGen g
                 => AgentLocalMonad g (Maybe (EventHandler g))
                 -> Time
                 -> [SugEnvSiteOccupier]
                 -> Loan
                 -> EventHandler g
lendingToHandler cont0 dueDate ns borrowerLoan@(Loan _ _ sugarFace spiceFace) = 
    continueWithAfter
      (proc evt -> 
        case evt of
          (DomainEvent borrowerId (LoanReply reply)) -> do
            mhdl <- arrM (uncurry (handleLendingReply cont0)) -< (borrowerId, reply)
            returnA -< ((), mhdl)
          _ -> do
            aid <- constM myId -< ()
            returnA -< error $ "Agent " ++ show aid ++ ": received unexpected event " ++ show evt ++ " during lending, terminating simulation!")
  where
    handleLendingReply :: RandomGen g
                       => AgentLocalMonad g (Maybe (EventHandler g))
                       -> AgentId
                       -> LoanReply
                       -> AgentLocalMonad g (Maybe (EventHandler g))
    handleLendingReply cont _ (RefuseLoan _) =  
      -- the sender refuses the Loan-offer, continue with the next neighbour
      lendTo cont dueDate ns
    handleLendingReply cont borrowerId AcceptLoan = do 
      let lenderLoan = Loan dueDate borrowerId sugarFace spiceFace

      -- the sender accepts the Loan-offer, remove Loan-wealth from lender and add borrower
      updateAgentState (\s -> s { sugAgSugarLevel = sugAgSugarLevel s - sugarFace
                                , sugAgSpiceLevel = sugAgSpiceLevel s - spiceFace
                                , sugAgLent       = lenderLoan : sugAgLent s})
      -- NOTE: need to update occupier-info in environment because wealth has (and MRS) changed
      updateSiteOccupied
      -- continue with next neighbour
      aid <- myId
      DBG.trace ("Agent " ++ show aid ++ ": lending " ++ show borrowerLoan ++ " to " ++ show borrowerId ++ " with lenderLoan = " ++ show lenderLoan) 
                  lendTo cont dueDate ns

handleLoanOffer :: RandomGen g
                => Loan
                -> AgentLocalMonad g ()
handleLoanOffer borrowerLoan@(Loan _ lender sugarFace spiceFace) = do
  pb <- potentialBorrower
  case pb of
    Just reason -> 
      sendEventTo lender (LoanReply $ RefuseLoan reason)

    Nothing -> do
      -- the borrower accepts the Loan-offer, increase wealth of borrower
      -- and add to borrowers obligations
      updateAgentState (\s -> s { sugAgSugarLevel = sugAgSugarLevel s + sugarFace
                                , sugAgSpiceLevel = sugAgSpiceLevel s + spiceFace
                                , sugAgBorrowed   = borrowerLoan : sugAgBorrowed s })
      -- NOTE: need to update occupier-info in environment because wealth has (and MRS) changed
      updateSiteOccupied

      aid <- myId
      DBG.trace ("Agent " ++ show aid ++ ": borrowing " ++ show borrowerLoan ++ " from " ++ show lender)
        sendEventTo lender (LoanReply AcceptLoan)

handleLoanPayback :: RandomGen g
                  => AgentId
                  -> Loan
                  -> Double
                  -> Double
                  -> AgentLocalMonad g ()
handleLoanPayback borrower borrowerLoan@(Loan dueDate _ sugarFace spiceFace) sugarBack spiceBack = do
    t          <- absStateLift getSimTime
    newDueDate <- (t+) . fst . fromJust . spLoansEnabled <$> scenario
    rate       <- (/100) . snd . fromJust . spLoansEnabled <$> scenario 

    aid <- myId
    ls  <- processBorrower rate aid newDueDate <$> agentProperty sugAgLent

    updateAgentState (\s -> s { sugAgLent = ls
                              , sugAgSugarLevel = sugAgSugarLevel s + sugarBack
                              , sugAgSpiceLevel = sugAgSpiceLevel s + spiceBack})
    -- NOTE: need to update occupier-info in environment because wealth (and MRS) might have changed
    updateSiteOccupied
  where
    processBorrower :: Double
                    -> AgentId
                    -> Time
                    -> [Loan]
                    -> [Loan]
    processBorrower rate aid newDueDate ls 
        -- TODO: when inheritance is turned on, this error occurs sometimes, no idea why
        | isNothing mxid = error $ "Agent " ++ show aid ++ ": couldn't find " ++ show borrowerLoan ++  
                                   " for borrower payback in my loans " ++ show ls ++ ", exit."
        | otherwise = if fullyPaidBack l
                        then DBG.trace ("Agent " ++ show aid ++ ": received FULL Loan payback of " ++ show l ++ 
                                        " from " ++ show borrower) ls'
                        else DBG.trace("Agent " ++ show aid ++ ": received PARTIAL Loan payback of " ++ show l ++ 
                              " from " ++ show borrower) (l' : ls')
      where
        mxid = findIndex findBorrowerLoan ls 
        idx  = fromJust mxid
        l    = ls !! idx
        l'   = newLoan
        ls'  = removeElemByIdx idx ls

        fullyPaidBack :: Loan -> Bool
        fullyPaidBack (Loan _ _ sugarFace' spiceFace') 
            = fullSugarBack == sugarBack && fullSpiceBack == spiceBack
          where
            -- NOTE: need to include the interest!
            fullSugarBack = sugarFace' + (sugarFace' * rate)
            fullSpiceBack = spiceFace' + (spiceFace' * rate)

        newLoan :: Loan
        newLoan = Loan newDueDate borrower (sugarFace - sugarBack) (spiceFace - spiceBack)

        findBorrowerLoan :: Loan -> Bool
        findBorrowerLoan (Loan dueDate' borrower' sugarFace' spiceFace') 
          = borrower'  == borrower  &&
            dueDate'   == dueDate   &&
            sugarFace' == sugarFace &&
            spiceFace' == spiceFace

handleLoanLenderDied :: RandomGen g
                     => AgentId
                     -> [AgentId]
                     -> AgentLocalMonad g ()
handleLoanLenderDied lender children = do
    oldLs <- agentProperty sugAgBorrowed
    ls    <- processLoans [] <$> agentProperty sugAgBorrowed
    aid   <- myId

    DBG.trace ("Agent " ++ show aid ++ ": lender " ++ show lender ++ " died and inherits its loan to its children " ++ show children ++ 
               "\nold loans = " ++ show oldLs ++ 
               "\nnew loans = " ++ show ls)
      updateAgentState (\s -> s { sugAgBorrowed = ls })
  where
    processLoans :: [Loan]
                 -> [Loan]
                 -> [Loan]
    processLoans acc [] = acc
    processLoans acc (l@(Loan _ lender' _ _) : ls) 
        | lender' /= lender = processLoans (l : acc) ls       -- different lender, keep as it is
        | otherwise         = processLoans (newLs ++ acc) ls  -- died lender, change to new 
      where
        newLs = splitLoanBorrowed children l

handleLoanInherit :: RandomGen g
                  => AgentId
                  -> Loan
                  -> AgentLocalMonad g ()
handleLoanInherit parent loan = do
  aid <- myId
  DBG.trace ("Agent " ++ show aid ++ ": inherited " ++ show loan ++ " from parent " ++ show parent)
    updateAgentState (\s -> s { sugAgLent = loan : sugAgLent s })

potentialLender :: AgentLocalMonad g (Maybe (Double, Double))
potentialLender = do
  age                <- agentProperty sugAgAge
  (fertMin, fertMax) <- agentProperty sugAgFertAgeRange

  sugLvl <- agentProperty sugAgSugarLevel
  spiLvl <- agentProperty sugAgSpiceLevel

  if age > fertMax
    then return $ Just (sugLvl / 2, spiLvl / 2) -- agent is too old to bear children, will lend half of its current wealth
    else -- agent is within child-bearing age 
      if age >= fertMin && age <= fertMax
        then do 
          initSugLvl <- agentProperty sugAgInitSugEndow
          initSpiLvl <- agentProperty sugAgInitSpiEndow

          -- check if this agent has wealth excess
          if sugLvl > initSugLvl && spiLvl > initSpiLvl
            then return $ Just (sugLvl - initSugLvl, spiLvl - initSpiLvl) -- lend excess wealth
            else return Nothing -- no wealth excess
        else return Nothing -- not within child-bearing age, too young, nothing

potentialBorrower :: AgentLocalMonad g (Maybe LoanRefuse)
potentialBorrower = do
  age                <- agentProperty sugAgAge
  (fertMin, fertMax) <- agentProperty sugAgFertAgeRange

  sugLvl <- agentProperty sugAgSugarLevel
  spiLvl <- agentProperty sugAgSpiceLevel

  if age >= fertMin && age <= fertMax
    then do
      initSugLvl <- agentProperty sugAgInitSugEndow
      initSpiLvl <- agentProperty sugAgInitSpiEndow

      if sugLvl > initSugLvl && spiLvl > initSpiLvl
        then return $ Just EnoughWealth -- agent has already enough wealth
        else do
          netInc <- agentProperty sugAgNetIncome

          if netInc <= 0
            then return $ Just NotLoanWorthy -- not Loanworthy: has the agent had a net income in the most recent time-step?
            else return Nothing                    

    else return $ Just NotFertileAge -- agent is not child-bearing age 

splitLoanBorrowed :: [AgentId]
                  -> Loan
                  -> [Loan]
splitLoanBorrowed as (Loan dueDate _ sugarFace spiceFace)
    = map (\a -> Loan dueDate a (sugarFace / n) (spiceFace / n)) as
  where
    n = fromIntegral $ length as

splitLoanLent :: Int
              -> Loan
              -> [Loan]
splitLoanLent n (Loan dueDate borrower sugarFace spiceFace)
    = replicate n (Loan dueDate borrower (sugarFace / n') (spiceFace / n'))
  where
    n' = fromIntegral n