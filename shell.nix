let

myHaskellPackageOverlay = self: super: {
  myHaskellPackages = super.haskellPackages.override {
    overrides = hself: hsuper: rec {

      gloss-export = super.haskell.lib.doJailbreak (super.haskell.lib.dontCheck (
        hself.callCabal2nix "gloss-export" (builtins.fetchGit {
          url = "https://gitlab.com/timo-a/gloss-export.git";
          rev = "d2d38aa679f31838ff718ffaac1fd9459b5c925f";
        }) { }));

      ihaskell-diagrams = hself.callCabal2nixWithOptions "ihaskell-diagrams" (builtins.fetchGit {
        url = "https://github.com/IHaskell/IHaskell";
        rev = "725d900414462da0f1859334a482e80c7a9e33d9";
        # rev = "1c22a874ac0c8ed019229f4a0cd5a0bfda017357";
      }) "--subpath ihaskell-display/ihaskell-diagrams" { };

    };
  };
};

in

{ nixpkgs ? import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/refs/tags/22.11.tar.gz")
  {
    config.allowBroken = false;
    overlays = [ myHaskellPackageOverlay ];
  }
}:

let

  pkgs = nixpkgs;

  haskellDeps = ps: with ps; [
    array base bearriver cassava Chart containers diagrams dunai gloss
    MonadRandom mtl diagrams diagrams-cairo hvega ihaskell-hvega
    plots
    ihaskell-juicypixels
    gloss-export
    ihaskell-diagrams
    vector
    gloss-export
    bytestring
  ];

in

pkgs.stdenv.mkDerivation {
  name = "Whatever";

  buildInputs = [
    pkgs.libintlOrEmpty
    (pkgs.myHaskellPackages.ghcWithPackages haskellDeps)
  ];
}
