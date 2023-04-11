{-# LANGUAGE Arrows #-}
{-# LANGUAGE Strict #-}
{-# LANGUAGE FlexibleContexts #-}

module Main (main, main', whiteNoise, fallingMassMSF, fallingMass, fallingMass') where

import           Control.Monad.Random
import           Control.Monad.Reader
import           FRP.BearRiver
import           Data.MonadicStreamFunction.InternalCore

import Control.Monad.State
import Data.Void

fallingMass :: Monad m => Double -> Double -> MSF (ClockInfo m) Void Double
fallingMass p0 v0 =
  arr (const (-9.81)) >>>
  integral >>>
  arr (+ v0) >>>
  integral >>>
  arr (+ p0)

fallingMass' :: Monad m => Double -> Double -> MSF (ClockInfo m) Double Double
fallingMass' p0 v0 = proc _ -> do
    v <- arr (+v0) <<< integral -< (-9.8)
    p <- arr (+p0) <<< integral -< v
    returnA -< p

type FallingMassStack g = (StateT Int (RandT g IO))
type FallingMassMSF g = SF (FallingMassStack g) () Double

main' :: IO ()
main' = do
  let g0  = mkStdGen 42
      s0  = 0
      -- msf = fallingMassMSF 0 100
      msf = whiteNoise
  runMSF g0 s0 msf

main :: IO ()
main = do
  let g0  = mkStdGen 1729
      s0  = 0
      msf = brownianMotion
  runMSF g0 s0 msf

arrM_ :: Monad m => m b -> MSF m a b
arrM_ = arrM . const

runMSF :: StdGen -> Int ->
          MSF (ReaderT DTime (StateT Int (RandT StdGen IO))) () Double ->
          IO ()
runMSF g s msf = do
  let msfReaderT = unMSF msf ()
      msfStateT  = runReaderT msfReaderT 0.1
      msfRand    = runStateT msfStateT s
      msfIO      = runRandT msfRand g

  (((_p, msf'), s'), g') <- msfIO

  when (s' <= 1000) (runMSF g' s' msf')

brownianMotion :: MSF (ReaderT DTime (StateT Int (RandT StdGen IO))) () Double
brownianMotion = proc _ -> do
  r1 <- arrM_ (lift $ lift $ getRandomR (0.0, 1.0)) -< ()
  r2 <- arrM_ (lift $ lift $ getRandomR (0.0, 1.0)) -< ()
  (n1, _) <- arr f -< (r1, r2)
  arrM_ (lift $ modify (+1)) -< ()
  bm <- integral -< n1
  arrM (liftIO . putStrLn) -< show bm
  returnA -< bm
  where
    f (u1, u2) = ( sqrt ((-2) * log u1) * (cos (2 * pi * u2))
                 , sqrt ((-2) * log u1) * (sin (2 * pi * u2)))

whiteNoise :: RandomGen g => FallingMassMSF g
whiteNoise = proc _ -> do
  r1 <- arrM_ (lift $ lift $ getRandomR (0.0, 1.0)) -< ()
  r2 <- arrM_ (lift $ lift $ getRandomR (0.0, 1.0)) -< ()
  (n1, _) <- arr f -< (r1, r2)
  _s <- arrM_ (lift get) -< ()
  -- arrM (liftIO . putStrLn) -< "n1 = " ++ show n1 ++ ", s = " ++ show s
  arrM_ (lift $ modify (+1)) -< ()
  bm <- integral -< n1
  arrM (liftIO . putStrLn) -< show bm
  returnA -< bm
  where
    f (u1, u2) = ( sqrt ((-2) * log u1) * (cos (2 * pi * u2))
                 , sqrt ((-2) * log u1) * (sin (2 * pi * u2)))


fallingMassMSF :: RandomGen g
               => Double -> Double -> FallingMassMSF g
fallingMassMSF v0 p0 = proc _ -> do
  r <- arrM_ (lift $ lift $ getRandomR (0, 9.81)) -< ()
  arrM (liftIO . putStrLn) -< "r = " ++ show r

  v <- arr (+v0) <<< integral -< (-r)
  p <- arr (+p0) <<< integral -< v

  arrM_ (lift $ modify (+1)) -< ()

  if p > 0
    then returnA -< p
    else do
      s <- arrM_ (lift get) -< ()
      arrM (liftIO . putStrLn) -< "hit floor with v " ++ show v ++ " after " ++ show s ++ " steps"
      returnA -< p
