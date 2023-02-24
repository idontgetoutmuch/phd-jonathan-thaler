{ nixpkgs ? import <nixpkgs> { } }:

let

haskellDeps = ps: with ps; [
  array base bearriver containers dunai gloss MonadRandom mtl hvega ihaskell-hvega ];



in

import ./release.nix {
  compiler = "ghc902";
  packages = haskellDeps;
}