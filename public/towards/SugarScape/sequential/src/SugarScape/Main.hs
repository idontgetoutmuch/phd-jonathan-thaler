module Main where

import Data.Char
import Data.List
import System.IO

import Options.Applicative

import SugarScape.Export.ExportRunner
import SugarScape.Visual.GlossRunner
import SugarScape.Visual.Renderer
import SugarScape.Core.Scenario
import SugarScape.Core.Simulation

data Output = Console Int      -- steps
            | File Int String  -- steps, filename
            | Visual Int AgentColoring SiteColoring  -- render-freq, agent vis, site-vis

instance Show Output where
  show (Console steps)     = "CONSOLE " ++ show steps ++ 
                              " (print output after " ++ show steps ++ 
                              " steps to console)"
  show (File steps file)   = "FILE " ++ show steps ++ " " ++ show file ++ 
                              " (write output of " ++ show steps ++ " steps to file " ++ show file ++ ")"
  show (Visual 0 ac sc) = "VISUAL MAX " ++ show ac ++ " " ++ show sc ++
                              " (render as many steps per second possible, Agent-Coloring: " ++ show ac ++ 
                              ", Site-Coloring: " ++ show sc ++ ")"
  show (Visual freq ac sc) = "VISUAL "  ++ show freq ++ " " ++ show ac ++ " " ++ show sc ++
                              " (render " ++ show freq ++ 
                              " steps per second, Agent-Coloring: " ++ show ac ++ 
                              ", Site-Coloring: " ++ show sc ++ ")"

data Options = Options 
  { optScenario :: String
  , optOutput   :: Output
  , optRngSeed  :: Maybe Int
  }

-- RUNNING FROM COMMAND LINE EXAMPLES (using stack)
-- clear & stack exec -- sugarscape -s "Animation II-3" -f 1000 -o export/dynamics.m -r 42
-- clear & stack exec -- sugarscape -s "Animation II-3" -v 0 --ac Default --sc Resource -r 42

-- clear & stack exec -- sugarscape -s "Animation II-8" -f 1000 -o export/dynamics.m -r 42

-- PROFILING
-- clear & stack exec -- sugarscape -s "Animation II-3" -f 1000 -o export/dynamics.m -r 42 +RTS -p -h -s

-- TODOs
-- TESTING: can we add some tests which check for memory-leaks? e.g. running
--  various scenarios and check if memory-consumption is 'normal'? Can we use
--  criterion for that?

-- 100 steps, with last output print to console (Animation II-8)
-- sqeuential 4.73, 4.69, 4.68, 4.7

main :: IO ()
main = do
    hSetBuffering stdout NoBuffering
    o <- execParser opts
    runSugarscape o
  where
    opts = info (parseOptions <**> helper)
      ( fullDesc
     <> progDesc "Full implementation of the famous SugarScape model by J. Epstein and R. Axtell.")

runSugarscape :: Options -> IO ()
runSugarscape opts = do
  let scenarioName = optScenario opts
      ms           = findScenario scenarioName sugarScapeScenarios

  case ms of
    Nothing -> putStrLn $ "Couldn't find scenario " ++ show scenarioName ++ ", exit."
    Just scenario -> do
      let output    = optOutput opts
          rngSeed   = optRngSeed opts

      (initSimState, initOut, scenario') <- initSimulationOpt rngSeed scenario

      putStrLn "Running Sugarscape with... " 
      putStrLn "--------------------------------------------------"
      print scenario'
      putStrLn "--------------------------------------------------"

      putStrLn $ "RNG Seed: \t\t\t" ++ maybe 
                 "N/A - using default global random number initialisation" show rngSeed
      putStrLn $ "Output Type: \t\t\t" ++ show output
      putStrLn "--------------------------------------------------"

      case output of 
        Console steps     -> do
          let (t, _, _, _) = last $ simulateUntil steps initSimState
          print t
        File steps file   -> writeSimulationUntil file steps initSimState
        Visual freq av cv -> runGloss scenario' initSimState initOut freq av cv

      putStrLn "\n--------------------------------------------------\n"

findScenario :: String 
             -> [SugarScapeScenario]
             -> Maybe SugarScapeScenario
findScenario name0 
    = find (\s -> strToLower (sgScenarioName s) == name)
  where
    strToLower = map toLower
    name       = strToLower name0

parseOptions :: Parser Options
parseOptions 
  = Options 
    <$> strOption
      (  long "scenario"
      <> short 's'
      <> metavar "String"
      <> help "SugarScape scenario to run e.g. \"Animation II-2\"")
    <*> parseOutput
    <*> optional (option auto  
      ( long "rng" 
      <> short 'r'
      <> help "Fixing rng seed" 
      <> metavar "Int"))

parseOutput :: Parser Output
parseOutput = fileOut    <|> 
              consoleOut <|> 
              visualOut

consoleOut :: Parser Output
consoleOut = Console <$> option auto
              (  long "consoleout"
              <> short 'c'
              <> help "Print output to console after number of steps"
              <> value 1000
              <> metavar "Int")

fileOut :: Parser Output
fileOut = File 
        <$> option auto
          (  long "fileout"
          <> short 'f'
          <> help "Write each step to output file"
          <> value 1000
          <> metavar "Int" )
        <*> strOption
          (  long "output"
          <> short 'o'
          <> value "export/dynamics.m"
          <> metavar "String"
          <> help "Output file")
      
visualOut :: Parser Output
visualOut = Visual 
         <$> option auto
           (  long "visual"
           <> short 'v'
           <> help "Visual steps calculated per second without upper limit of steps calculated in total (infinitely running)"
           <> value 0
           <> metavar "Int" )
        <*> option auto
           (  long "ac"
           <> help "Coloring of agents"
           <> value Default
           <> metavar "Default | Gender | Culture | Tribe | Welfare | Disease")
        <*> option auto
           (  long "sc"
           <> help "Coloring of sites"
           <> value Resource
           <> metavar "Resource | Polution")