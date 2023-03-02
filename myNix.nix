let
  overlay = self: super: {
    haskell = super.haskell // {
      packageOverrides = selfH: superH: {

        gloss-export = super.haskell.lib.doJailbreak (super.haskell.lib.dontCheck (
          selfH.callCabal2nix "gloss-export" (builtins.fetchGit {
            url = "https://gitlab.com/timo-a/gloss-export.git";
            rev = "d2d38aa679f31838ff718ffaac1fd9459b5c925f";
          }) { }));

        ihaskell-diagrams = selfH.callCabal2nixWithOptions "ihaskell-diagrams" (builtins.fetchGit {
          url = "https://github.com/IHaskell/IHaskell";
          rev = "725d900414462da0f1859334a482e80c7a9e33d9";
          # rev = "1c22a874ac0c8ed019229f4a0cd5a0bfda017357";
        }) "--subpath ihaskell-display/ihaskell-diagrams" { };

      };
    };
  };
  nixpkgs = import (builtins.fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/refs/tags/22.11.tar.gz";
    sha256 = "11w3wn2yjhaa5pv20gbfbirvjq6i3m7pqrq2msf0g7cv44vijwgw";
  }) { overlays = [ overlay ]; };
in import ./release-9.0.nix {
  nixpkgs = nixpkgs; # equivalent to `inherit nixpkgs;`
  packages = haskellPackages: with haskellPackages; [
    array base bearriver Chart containers diagrams dunai gloss
    MonadRandom mtl diagrams diagrams-cairo hvega ihaskell-hvega
    plots
    ihaskell-juicypixels
    gloss-export
    ihaskell-diagrams
    diagrams-cairo
    diagrams-rasterific
    csv
    cassava
  ];
}