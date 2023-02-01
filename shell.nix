let

myHaskellPackageOverlay = self: super: {
  myHaskellPackages = super.haskellPackages.override {
    overrides = hself: hsuper: rec {

    };
  };
};

in

{ nixpkgs ? import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/nixpkgs-22.05-darwin.tar.gz")
  {
    config.allowBroken = true;
    overlays = [ myHaskellPackageOverlay ];
  }
}:

let
  R-with-my-packages = nixpkgs.rWrapper.override{
    packages = with nixpkgs.rPackages; [
    ]; };

  pkgs = nixpkgs;

  haskellDeps = ps: with ps; [
    array base bearriver containers dunai gloss MonadRandom mtl
  ];

in

pkgs.stdenv.mkDerivation {
  name = "whatever";

  buildInputs = [
    pkgs.libintlOrEmpty
    R-with-my-packages
    pkgs.cabal2nix
    (pkgs.myHaskellPackages.ghcWithPackages haskellDeps)
    pkgs.darwin.apple_sdk.frameworks.Cocoa
  ];
}
