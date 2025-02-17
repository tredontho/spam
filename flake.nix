{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  inputs.poetry2nix.url = "github:nix-community/poetry2nix";

  outputs = { self, nixpkgs, poetry2nix }:
    let
      supportedSystems = [ "x86_64-linux" "x86_64-darwin" "aarch64-linux" "aarch64-darwin" ];
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
      pkgs = forAllSystems (system: nixpkgs.legacyPackages.${system});
    in
    {
      packages = forAllSystems (system: let
        inherit (poetry2nix.lib.mkPoetry2Nix { pkgs = pkgs.${system}; }) mkPoetryApplication;
      in {
        default = mkPoetryApplication { projectDir = self; };
      });

      devShells = forAllSystems (system: let
        inherit (poetry2nix.lib.mkPoetry2Nix { pkgs = pkgs.${system}; }) mkPoetryEnv;
        system_pkgs = pkgs.${system};
      in {
        default = system_pkgs.mkShellNoCC {
          shellHook = ''
            export LD_LIBRARY_PATH=${system_pkgs.lib.makeLibraryPath [
              system_pkgs.stdenv.cc.cc
              system_pkgs.zlib
            ]}
          '';
          packages = with system_pkgs; [
            (mkPoetryEnv { projectDir = self; })
            poetry
          ];
          buildInputs = with system_pkgs; [
            python311Packages.numpy
            stdenv.cc.cc.lib
            zlib
          ];
        };
      });
    };
}
